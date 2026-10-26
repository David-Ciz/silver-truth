from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

import matplotlib
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_QA_FOLD_PATHS = {
    "Fold 1": Path(
        "data/paper_runs/qa_results/BF-C2DL-HSC/sz64/"
        "hsc_crop64_qa_predictions_fold-1.xlsx"
    ),
    "Fold 2": Path(
        "data/paper_runs/qa_results/BF-C2DL-HSC/sz64/"
        "hsc_crop64_qa_predictions_fold-2.xlsx"
    ),
}

DEFAULT_MLFLOW_DB_PATH = Path("hpc_run_results/mlflow_export.db")

MUSC_CONTEXT_CROPS = ["sz64", "sz128", "sz192", "sz256", "sz320"]

MUSC_INVALID_DETAIL = "Only 53.6% of MuSC GT boxes fit into 64x64 crops"

MUSC_EXPERIMENT_PATTERN = re.compile(r"ablation-BF-C2DL-MuSC-(musc_crop\d+)-(fold-\d+)")

MUSC_LEARNED_RUNS = {
    "ensemble_only": "test_iou",
    "qa_only__top1": "test_iou",
    "ensemble_qa_t0.75": "test_iou",
    "ensemble_qa_retrained_t0.75": "test_iou",
}

SPLIT_DISPLAY_NAMES = {
    "train": "Training",
    "validation": "Validation",
    "test": "Held-Out Test",
}


def _load_split_sheet(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "QA workbook not found: "
            f"{path}. Sync the March 27, 2026 HSC64 threshold-sweep outputs first "
            "or pass the workbook path explicitly with --qa-fold-1/--qa-fold-2."
        )
    return pd.read_excel(path, sheet_name=split)


def generate_qa_scatter_for_split(
    fold_paths: dict[str, Path], output_path: Path, split: str
) -> None:
    fig, axes = plt.subplots(1, len(fold_paths), figsize=(10, 4), sharex=True, sharey=True)
    if len(fold_paths) == 1:
        axes = [axes]

    for ax, (fold_label, excel_path) in zip(axes, fold_paths.items(), strict=True):
        df = _load_split_sheet(excel_path, split)
        actual = df["Jaccard index"]
        predicted = df["Predicted Jaccard index"]
        pearson_r, _ = pearsonr(actual, predicted)
        spearman_r, _ = spearmanr(actual, predicted)

        ax.scatter(actual, predicted, s=8, alpha=0.35, color="#0f766e", edgecolors="none")
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#64748b")
        ax.set_title(f"{fold_label} (n={len(df)})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Actual IoU")
        ax.text(
            0.04,
            0.95,
            f"Pearson r = {pearson_r:.4f}\nSpearman rho = {spearman_r:.4f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.3"},
        )

    axes[0].set_ylabel("Predicted IoU")
    split_label = SPLIT_DISPLAY_NAMES.get(split, split.title())
    fig.suptitle(f"HSC64 QA Validity on {split_label} Cells", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _connect_readonly_sqlite(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise FileNotFoundError(
            "MLflow export DB not found: "
            f"{db_path}. Pass the correct path with --mlflow-db."
        )
    return sqlite3.connect(f"file:{db_path.as_posix()}?immutable=1", uri=True)


def load_musc_context_points(db_path: Path) -> list[dict[str, object]]:
    connection = _connect_readonly_sqlite(db_path)
    cursor = connection.cursor()
    experiment_rows = cursor.execute(
        """
        select experiment_id, name
        from experiments
        where name like 'ablation-BF-C2DL-MuSC-musc_crop%-fold-%'
        order by name
        """
    ).fetchall()

    best_by_crop_fold: dict[tuple[str, str], tuple[float, str]] = {}
    for experiment_id, experiment_name in experiment_rows:
        match = MUSC_EXPERIMENT_PATTERN.fullmatch(experiment_name)
        if match is None:
            continue
        crop_tag = match.group(1).replace("musc_crop", "sz")
        fold_name = match.group(2)
        if crop_tag not in MUSC_CONTEXT_CROPS:
            continue

        fold_best: tuple[float, str] | None = None
        for run_name, metric_key in MUSC_LEARNED_RUNS.items():
            row = cursor.execute(
                """
                select m.value
                from runs r
                join tags t on r.run_uuid = t.run_uuid and t.key = 'mlflow.runName'
                join metrics m on r.run_uuid = m.run_uuid and m.key = ?
                where r.experiment_id = ? and t.value = ?
                """,
                (metric_key, experiment_id, run_name),
            ).fetchone()
            if row is None:
                continue
            score = float(row[0])
            if fold_best is None or score > fold_best[0]:
                fold_best = (score, run_name)

        if fold_best is None:
            raise ValueError(
                f"No learned MuSC scores found for experiment '{experiment_name}'."
            )
        best_by_crop_fold[(crop_tag, fold_name)] = fold_best

    connection.close()

    points: list[dict[str, object]] = []
    for crop_tag in MUSC_CONTEXT_CROPS:
        fold_1 = best_by_crop_fold.get((crop_tag, "fold-1"))
        fold_2 = best_by_crop_fold.get((crop_tag, "fold-2"))
        if fold_1 is None or fold_2 is None:
            raise ValueError(
                f"Missing MuSC learned sweep result for {crop_tag}: "
                f"fold-1={fold_1 is not None}, fold-2={fold_2 is not None}."
            )
        mean_score = (fold_1[0] + fold_2[0]) / 2.0
        points.append(
            {
                "crop_tag": crop_tag,
                "fold_1_best_iou": fold_1[0],
                "fold_1_winner": fold_1[1],
                "fold_2_best_iou": fold_2[0],
                "fold_2_winner": fold_2[1],
                "best_learned_mean_iou": mean_score,
                "scientifically_valid": crop_tag != "sz64",
            }
        )
    return points


def generate_musc_context_curve(
    output_path: Path,
    context_points: list[dict[str, object]],
    csv_output_path: Path,
) -> None:
    df = pd.DataFrame(context_points)
    labels = df["crop_tag"].tolist()
    values = df["best_learned_mean_iou"].tolist()
    colors = ["#9ca3af" if label == "sz64" else "#d97706" for label in labels]
    edge_colors = ["#6b7280" if label == "sz64" else "#92400e" for label in labels]
    optimum_idx = int(df["best_learned_mean_iou"].idxmax())

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x_positions = list(range(len(labels)))
    ax.plot(x_positions, values, marker="o", linewidth=2, color="#92400e", zorder=2)
    bars = ax.bar(
        x_positions,
        values,
        width=0.55,
        color=colors,
        edgecolor=edge_colors,
        alpha=0.28,
        zorder=1,
    )
    bars[optimum_idx].set_facecolor("#0f766e")
    bars[optimum_idx].set_edgecolor("#115e59")
    bars[optimum_idx].set_alpha(0.35)
    ax.scatter(
        [optimum_idx],
        [values[optimum_idx]],
        s=140,
        marker="*",
        color="#0f766e",
        edgecolors="#134e4a",
        linewidths=0.8,
        zorder=3,
    )

    ax.set_xticks(x_positions, labels)
    ax.set_xlabel("Crop size")
    ax.set_ylabel("Best learned mean IoU")
    ax.set_ylim(0.58, 0.81)
    ax.set_title("MuSC Crop-Context Sweep")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)

    for idx, (label, value) in enumerate(zip(labels, values, strict=True)):
        ax.text(
            idx,
            value + 0.004,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.annotate(
        "Optimum at sz256",
        xy=(optimum_idx, values[optimum_idx]),
        xytext=(optimum_idx + 0.35, values[optimum_idx] + 0.02),
        textcoords="data",
        fontsize=9,
        color="#0f766e",
        ha="left",
        va="bottom",
        arrowprops={"arrowstyle": "->", "color": "#0f766e", "linewidth": 1.1},
    )

    fig.text(
        0.02,
        0.01,
        MUSC_INVALID_DETAIL,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#4b5563",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(csv_output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from saved QA and MuSC outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/paper_runs/figures"),
        help="Directory for the generated figure files.",
    )
    parser.add_argument(
        "--qa-fold-1",
        type=Path,
        default=DEFAULT_QA_FOLD_PATHS["Fold 1"],
        help="Excel file for HSC QA predictions on fold 1.",
    )
    parser.add_argument(
        "--qa-fold-2",
        type=Path,
        default=DEFAULT_QA_FOLD_PATHS["Fold 2"],
        help="Excel file for HSC QA predictions on fold 2.",
    )
    parser.add_argument(
        "--qa-splits",
        nargs="+",
        default=["test", "train", "validation"],
        choices=["train", "validation", "test"],
        help="Workbook sheet(s) to render for the HSC QA scatter figure.",
    )
    parser.add_argument(
        "--mlflow-db",
        type=Path,
        default=DEFAULT_MLFLOW_DB_PATH,
        help="Read-only SQLite export DB used for the MuSC context curve.",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip the HSC QA scatter panels and render only the MuSC context figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_paths = {
        "Fold 1": args.qa_fold_1,
        "Fold 2": args.qa_fold_2,
    }
    if not args.skip_qa:
        for split in args.qa_splits:
            output_name = (
                "hsc_qa_scatter.png"
                if split == "test"
                else f"hsc_qa_scatter_{split}.png"
            )
            generate_qa_scatter_for_split(
                fold_paths,
                args.output_dir / output_name,
                split,
            )
    musc_context_points = load_musc_context_points(args.mlflow_db)
    generate_musc_context_curve(
        args.output_dir / "musc_context_curve.png",
        musc_context_points,
        args.output_dir / "musc_context_curve.csv",
    )


if __name__ == "__main__":
    main()
