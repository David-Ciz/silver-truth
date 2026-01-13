#!/usr/bin/env python3
"""
Minimal Click CLI to load segmentation masks from a Parquet into FiftyOne.

- Uses fo.Segmentation(mask_path=...) exactly as requested
- One GT mask column and any number of competitor mask columns
- Optional base image column for nicer viewing
- Can overwrite or append to an existing FiftyOne dataset
- Optionally launches the FiftyOne App

Install:
    pip install fiftyone click pandas pyarrow

Examples:
    python fo_from_parquet_click_min.py \
      --parquet data/dataset.parquet \
      --dataset-name seg-compare \
      --image-col raw_image \
      --gt-col gt_mask \
      --pred-cols competitor_a,competitor_b \
      --launch

    # Overwrite existing dataset
    python fo_from_parquet_click_min.py \
      --parquet data/dataset.parquet \
      --gt-col gt_mask \
      --pred-cols competitor_a \
      --dataset-name seg-compare \
      --overwrite \
      --launch
"""

import click
import pandas as pd
import fiftyone as fo


def _split_cols(s: str | None) -> list[str]:
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def _get_str(row: pd.Series, col: str | None) -> str | None:
    if not col:
        return None
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    return str(v)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--parquet", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to Parquet file")
@click.option("--filter-only-gt", default=True, show_default=True, help="Create dataset only from rows with GT mask")
@click.option("--dataset-name", default="seg-compare", show_default=True, help="FiftyOne dataset name")
@click.option("--overwrite", is_flag=True, help="Delete existing dataset with same name before import")
@click.option("--image-col", default=None, help="Optional column with base image filepath")
@click.option("--gt-col", required=True, default="gt_image", help="Column with GT mask path")
@click.option("--pred-cols", default=None, help="Comma-separated columns with competitor mask paths")
@click.option("--limit", type=int, default=None, help="Optionally limit number of rows imported")
@click.option("--launch", is_flag=True, help="Launch the FiftyOne App after import")
def main(
    parquet: str,
    filter_only_gt: bool,
    dataset_name: str,
    overwrite: bool,
    image_col: str | None,
    gt_col: str,
    pred_cols: str | None,
    limit: int | None,
    launch: bool,
):
    # Load rows
    df = pd.read_parquet(parquet)
    if limit is not None:
        df = df.head(limit)

    if filter_only_gt:
        df = df[df[gt_col].notna()]


    pred_cols_list = _split_cols(pred_cols)

    # Prepare dataset
    if overwrite and dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    dataset = fo.Dataset(dataset_name) if dataset_name not in fo.list_datasets() else fo.load_dataset(dataset_name)

    samples: list[fo.Sample] = []

    for _, r in df.iterrows():
        # Choose a base filepath so the sample opens in the App
        base = _get_str(r, image_col) or _get_str(r, gt_col)
        if not base:
            # try from first pred col
            for c in pred_cols_list:
                base = _get_str(r, c)
                if base:
                    break
        if not base:
            continue

        s = fo.Sample(filepath=base)

        # GT segmentation
        gt_path = _get_str(r, gt_col)
        if gt_path:
            s["gt"] = fo.Segmentation(mask_path=gt_path)

        # Competitor segmentations
        for c in pred_cols_list:
            p = _get_str(r, c)
            if p:
                s[c] = fo.Segmentation(mask_path=p)

        samples.append(s)

    if not samples:
        click.echo("No samples created. Check your column names and paths.")
        return

    click.echo(f"Adding {len(samples)} samples to dataset '{dataset_name}'...")
    dataset.add_samples(samples)
    try:
        dataset.save()  # ensure schema + state are flushed (safe even if not strictly required)
    except Exception:
        pass

    if launch:
        try:
            session = fo.launch_app(dataset,)
            click.echo(f"FiftyOne App URL: {session.url}")
            # Keep the process alive until the app/session is closed
            session.wait()
        except Exception as e:
            click.echo(f"Failed to launch the FiftyOne App automatically: {e}")
            click.echo("You can try launching it manually:")
            click.echo(f"  fiftyone app launch --dataset {dataset_name}")
    else:
        click.echo("Done. Launch later with:")
        click.echo(f"  fiftyone app launch --dataset {dataset_name}")


if __name__ == "__main__":
    main()