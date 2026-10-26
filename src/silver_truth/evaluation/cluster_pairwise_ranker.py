from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from silver_truth.evaluation.candidate_ranker import (
    DEFAULT_RANKER_FEATURE_COLUMNS,
    add_cluster_ranker_features,
)


DEFAULT_PAIRWISE_FEATURE_COLUMNS = DEFAULT_RANKER_FEATURE_COLUMNS


@dataclass(frozen=True)
class ClusterPairwiseRankerConfig:
    target_column: str = "true_iou"
    train_splits: tuple[str, ...] = ("train",)
    feature_columns: tuple[str, ...] = DEFAULT_PAIRWISE_FEATURE_COLUMNS
    use_source_identity: bool = False
    cluster_distance: float = 4.0
    min_target_gap: float = 0.03
    max_pairs_per_cluster: int | None = None
    random_state: int = 17
    logistic_c: float = 1.0
    source_identity_blend: float = 1.0


def train_cluster_pairwise_ranker(
    features: pd.DataFrame,
    *,
    output_model: Path | None = None,
    config: ClusterPairwiseRankerConfig = ClusterPairwiseRankerConfig(),
) -> dict[str, Any]:
    augmented = add_cluster_ranker_features(
        features, cluster_distance=config.cluster_distance
    )
    feature_columns = [c for c in config.feature_columns if c in augmented.columns]
    if not feature_columns:
        raise ValueError("No configured pairwise feature columns are present.")

    pairwise = build_pairwise_training_table(
        augmented,
        feature_columns=feature_columns,
        config=config,
    )
    if pairwise.empty or pairwise["target"].nunique() < 2:
        raise ValueError("Pairwise training table does not contain both classes.")
    if not 0.0 <= config.source_identity_blend <= 1.0:
        raise ValueError("source_identity_blend must be between 0 and 1.")

    numeric_columns = [
        f"diff__{column}"
        for column in feature_columns
        if f"diff__{column}" in pairwise.columns
        and pairwise[f"diff__{column}"].notna().any()
    ]
    feature_columns = [column.removeprefix("diff__") for column in numeric_columns]
    if not numeric_columns:
        raise ValueError("Pairwise training table has no non-missing numeric columns.")
    categorical_columns = (
        ["left_source", "right_source"] if config.use_source_identity else []
    )
    model_columns = numeric_columns + categorical_columns
    model = _build_pairwise_model(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        random_state=config.random_state,
        logistic_c=config.logistic_c,
    )
    model.fit(pairwise[model_columns], pairwise["target"].astype(int))
    source_agnostic_model = None
    if categorical_columns and config.source_identity_blend < 1.0:
        source_agnostic_model = _build_pairwise_model(
            numeric_columns=numeric_columns,
            categorical_columns=[],
            random_state=config.random_state,
            logistic_c=config.logistic_c,
        )
        source_agnostic_model.fit(
            pairwise[numeric_columns], pairwise["target"].astype(int)
        )
    payload = {
        "model": model,
        "source_agnostic_model": source_agnostic_model,
        "config": asdict(config),
        "feature_columns": feature_columns,
        "model_columns": model_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "n_pairs": int(len(pairwise)),
    }
    if output_model is not None:
        output_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, output_model)
    return payload


def score_with_cluster_pairwise_ranker(
    features: pd.DataFrame,
    *,
    model_payload: dict[str, Any] | Path,
) -> pd.DataFrame:
    payload = _load_payload(model_payload)
    config = ClusterPairwiseRankerConfig(**payload["config"])
    augmented = add_cluster_ranker_features(
        features, cluster_distance=config.cluster_distance
    )
    feature_columns = list(payload["feature_columns"])
    missing = sorted(set(feature_columns).difference(augmented.columns))
    if missing:
        raise ValueError(f"Feature table is missing pairwise columns: {missing}")

    scored = augmented.copy()
    scores = pd.Series(np.nan, index=scored.index, dtype=float)
    for _, group in scored.groupby("ranker_cluster_id", dropna=False, sort=False):
        if len(group) == 1:
            scores.loc[group.index] = 0.5
            continue
        pair_table, left_indices = build_pairwise_scoring_table(
            group,
            feature_columns=feature_columns,
            use_source_identity=config.use_source_identity,
        )
        probabilities = payload["model"].predict_proba(
            pair_table[payload["model_columns"]]
        )[:, 1]
        source_agnostic_model = payload.get("source_agnostic_model")
        if source_agnostic_model is not None:
            source_blend = float(config.source_identity_blend)
            source_agnostic_probabilities = source_agnostic_model.predict_proba(
                pair_table[payload["numeric_columns"]]
            )[:, 1]
            probabilities = (
                source_blend * probabilities
                + (1.0 - source_blend) * source_agnostic_probabilities
            )
        score_table = pd.DataFrame(
            {"index": left_indices, "win_probability": probabilities}
        )
        scores.loc[group.index] = score_table.groupby("index")[
            "win_probability"
        ].mean()

    scored["pairwise_cluster_score"] = scores.fillna(0.5).clip(0.0, 1.0)
    scored["pairwise_cluster_rank"] = scored.groupby(
        "ranker_cluster_id", dropna=False
    )["pairwise_cluster_score"].rank(method="average", pct=True)
    return scored


def build_pairwise_training_table(
    features: pd.DataFrame,
    *,
    feature_columns: list[str],
    config: ClusterPairwiseRankerConfig,
) -> pd.DataFrame:
    if config.target_column not in features.columns:
        raise ValueError(f"Missing target column: {config.target_column}")

    split_mask = (
        features["split"].astype(str).isin(config.train_splits)
        if "split" in features.columns
        else pd.Series(True, index=features.index)
    )
    train = features[split_mask].copy()
    target = pd.to_numeric(train[config.target_column], errors="coerce")
    train = train[target.notna()].copy()
    if train.empty:
        return pd.DataFrame()

    records = []
    rng = np.random.default_rng(config.random_state)
    for _, group in train.groupby("ranker_cluster_id", dropna=False, sort=False):
        if len(group) < 2:
            continue
        group_records = _pair_records_for_group(
            group,
            feature_columns=feature_columns,
            target_column=config.target_column,
            min_target_gap=config.min_target_gap,
            use_source_identity=config.use_source_identity,
        )
        if (
            config.max_pairs_per_cluster is not None
            and len(group_records) > config.max_pairs_per_cluster
        ):
            selected = rng.choice(
                len(group_records), size=config.max_pairs_per_cluster, replace=False
            )
            group_records = [group_records[index] for index in selected]
        records.extend(group_records)
    return pd.DataFrame.from_records(records)


def build_pairwise_scoring_table(
    group: pd.DataFrame,
    *,
    feature_columns: list[str],
    use_source_identity: bool,
) -> tuple[pd.DataFrame, list[int]]:
    records = []
    left_indices = []
    rows = list(group.iterrows())
    for left_index, left in rows:
        for right_index, right in rows:
            if left_index == right_index:
                continue
            records.append(
                _pair_feature_record(
                    left,
                    right,
                    feature_columns=feature_columns,
                    use_source_identity=use_source_identity,
                )
            )
            left_indices.append(left_index)
    return pd.DataFrame.from_records(records), left_indices


def _pair_records_for_group(
    group: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    min_target_gap: float,
    use_source_identity: bool,
) -> list[dict[str, object]]:
    records = []
    rows = list(group.iterrows())
    for left_pos in range(len(rows)):
        _, left = rows[left_pos]
        left_target = float(left[target_column])
        for right_pos in range(left_pos + 1, len(rows)):
            _, right = rows[right_pos]
            right_target = float(right[target_column])
            gap = left_target - right_target
            if abs(gap) < min_target_gap:
                continue
            record = _pair_feature_record(
                left,
                right,
                feature_columns=feature_columns,
                use_source_identity=use_source_identity,
            )
            record["target"] = 1 if gap > 0 else 0
            records.append(record)
            reverse = _pair_feature_record(
                right,
                left,
                feature_columns=feature_columns,
                use_source_identity=use_source_identity,
            )
            reverse["target"] = 0 if gap > 0 else 1
            records.append(reverse)
    return records


def _pair_feature_record(
    left: pd.Series,
    right: pd.Series,
    *,
    feature_columns: list[str],
    use_source_identity: bool,
) -> dict[str, object]:
    record: dict[str, object] = {}
    for column in feature_columns:
        left_value = pd.to_numeric(pd.Series([left.get(column)]), errors="coerce").iloc[
            0
        ]
        right_value = pd.to_numeric(
            pd.Series([right.get(column)]), errors="coerce"
        ).iloc[0]
        record[f"diff__{column}"] = left_value - right_value
    if use_source_identity:
        record["left_source"] = str(left.get("candidate_source", ""))
        record["right_source"] = str(right.get("candidate_source", ""))
    return record


def _build_pairwise_model(
    *,
    numeric_columns: list[str],
    categorical_columns: list[str],
    random_state: int,
    logistic_c: float,
) -> Any:
    if logistic_c <= 0:
        raise ValueError("logistic_c must be positive.")

    if not categorical_columns:
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=logistic_c,
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
        )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_columns,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
        ]
    )
    return make_pipeline(
        preprocessor,
        LogisticRegression(
            C=logistic_c,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
    )


def _load_payload(model_payload: dict[str, Any] | Path) -> dict[str, Any]:
    if isinstance(model_payload, (str, Path)):
        return joblib.load(model_payload)
    return model_payload


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train or apply a pairwise local-cluster candidate ranker."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("features", type=Path)
    train_parser.add_argument("--output-model", required=True, type=Path)
    train_parser.add_argument("--train-split", action="append", default=None)
    train_parser.add_argument("--target-column", default="true_iou")
    train_parser.add_argument("--use-source-identity", action="store_true")
    train_parser.add_argument("--min-target-gap", type=float, default=0.03)
    train_parser.add_argument("--logistic-c", type=float, default=1.0)
    train_parser.add_argument("--source-identity-blend", type=float, default=1.0)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("features", type=Path)
    score_parser.add_argument("--model", required=True, type=Path)
    score_parser.add_argument("--output", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "train":
        features = _read_table(args.features)
        payload = train_cluster_pairwise_ranker(
            features,
            output_model=args.output_model,
            config=ClusterPairwiseRankerConfig(
                train_splits=tuple(args.train_split or ("train",)),
                target_column=args.target_column,
                use_source_identity=args.use_source_identity,
                min_target_gap=args.min_target_gap,
                logistic_c=args.logistic_c,
                source_identity_blend=args.source_identity_blend,
            ),
        )
        print(
            f"Wrote pairwise ranker to {args.output_model} with "
            f"{payload['n_pairs']} training pairs"
        )
    elif args.command == "score":
        features = _read_table(args.features)
        scored = score_with_cluster_pairwise_ranker(
            features, model_payload=args.model
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".parquet":
            scored.to_parquet(args.output, index=False)
        else:
            scored.to_csv(args.output, index=False)
        print(f"Wrote {len(scored)} pairwise-scored rows to {args.output}")


if __name__ == "__main__":
    main()
