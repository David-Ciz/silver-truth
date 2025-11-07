import click
from pathlib import Path
from typing import Optional

from src.qa.preprocessing import create_qa_dataset
from src.qa.evaluation import run_qa_evaluation
from src.qa.result_conversion import (
    convert_qa_csv_to_detailed_parquet,
    create_parquet_from_qa_results,
)


@click.group()
def cli():
    """A CLI tool for the Quality Assurance (QA) workflow."""
    pass


@cli.command()
@click.argument("dataset_dataframe_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("output_parquet_path", type=click.Path(file_okay=True, dir_okay=False))
@click.option(
    "--crop", default=False, is_flag=True, help="Create crops for the QA dataset"
)
@click.option("--crop-size", default=64, help="Size of the crops for the QA dataset")
@click.option(
    "--fusion-model",
    default=None,
    help="Fusion model used (e.g., 'MAJORITY_FLAT'). Stored in metadata for fused_images.",
)
@click.option(
    "--fusion-threshold",
    type=float,
    default=None,
    help="Fusion threshold used (e.g., 1.0). Stored in metadata for fused_images.",
)
@click.option(
    "--fusion-timepoints",
    default=None,
    help="Fusion timepoints range used (e.g., '0-61'). Stored in metadata for fused_images.",
)
@click.option(
    "--competitors",
    multiple=True,
    help="Specific competitors to process (can be specified multiple times). If not specified, all competitors will be processed. Use 'fused_images' to process only fusion results.",
)
def create_dataset(
    dataset_dataframe_path: str,
    output_dir: str,
    output_parquet_path: str,
    crop: bool = False,
    crop_size: int = 64,
    fusion_model: str = None,
    fusion_threshold: float = None,
    fusion_timepoints: str = None,
    competitors: tuple = None,
) -> None:
    """Creates a QA dataset for cell-level analysis."""
    # Convert tuple to list or None
    competitors_list = list(competitors) if competitors else None
    
    create_qa_dataset(
        dataset_dataframe_path, output_dir, output_parquet_path, crop, crop_size, 
        fusion_model, fusion_threshold, fusion_timepoints, competitors_list
    )


@cli.command()
@click.argument("qa_dataframe_path", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "ground_truth_dataframe_path", type=click.Path(exists=True, path_type=Path)
)
@click.option(
    "--competitor", help="Competitor name to evaluate. If None, evaluate all."
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Path to save results as CSV",
)
@click.option(
    "--visualize",
    "-v",
    is_flag=True,
    help="Generate visualization of results (Placeholder)",
)
def evaluate(
    qa_dataframe_path: Path,
    ground_truth_dataframe_path: Path,
    competitor: Optional[str] = None,
    output: Optional[Path] = None,
    visualize: bool = False,
):
    """Evaluates a QA dataset."""
    run_qa_evaluation(
        qa_dataframe_path,
        ground_truth_dataframe_path,
        competitor,
        output,
        visualize,
    )


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("qa_dataframe_path", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output parquet file path")
def convert_results(
    csv_path: str, qa_dataframe_path: str, output: Optional[str] = None
):
    """
    Convert QA evaluation CSV results to detailed parquet format.
    """
    result = convert_qa_csv_to_detailed_parquet(csv_path, qa_dataframe_path, output)
    if result:
        print(f"✅ Conversion successful: {result}")
    else:
        print("❌ Conversion failed")


@cli.command()
@click.option(
    "--qa-results-dir",
    default=".",
    help="Directory containing QA CSV result files",
)
@click.option(
    "--qa-dataframes-dir",
    default=".",
    help="Directory containing QA dataframe parquet files",
)
@click.option(
    "--output-dir",
    default="detailed_qa_results",
    help="Directory to save converted parquet files",
)
def batch_convert_results(qa_results_dir: str, qa_dataframes_dir: str, output_dir: str):
    """
    Batch convert all QA CSV results to detailed parquet format.
    """
    created_files = create_parquet_from_qa_results(
        qa_results_dir, qa_dataframes_dir, output_dir
    )

    print("\n✅ Batch conversion completed!")
    print(f"📁 Created {len(created_files)} detailed parquet files in: {output_dir}")

    if created_files:
        print("\n📄 Created files:")
        for file in created_files:
            print(f"   - {Path(file).name}")


@cli.command()
@click.argument("crops_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--layer",
    default=1,
    type=int,
    help="Layer index to extract (0=source, 1=GT, 2=tracking, 3=fused, 4+=competitors)",
)
def extract_layer(crops_dir: str, output_dir: str, layer: int):
    """
    Extract a specific layer from multi-layer crop TIFF files created by QA dataset.
    
    Layer indices (from create_qa_dataset):
      0 = Source image (raw/original)
      1 = Competitor/Fusion segmentation mask
      2 = Ground Truth (GT) segmentation mask
    
    Examples:
    
      # Extract source image crops
      python cli_qa.py extract-layer qa_output source_crops --layer 0
      
      # Extract competitor/fusion segmentation crops
      python cli_qa.py extract-layer qa_output segmentation_crops --layer 1
      
      # Extract Ground Truth segmentation crops
      python cli_qa.py extract-layer qa_output gt_crops --layer 2
    """
    import tifffile
    
    crops_path = Path(crops_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    crop_files = list(crops_path.glob("*.tif"))
    
    if not crop_files:
        click.echo(click.style(f"❌ No TIFF files found in {crops_dir}", fg="red"))
        return
    
    layer_names = {
        0: "source image (raw)",
        1: "competitor/fusion segmentation",
        2: "ground truth (GT) segmentation",
    }
    layer_name = layer_names.get(layer, f"unknown layer {layer}")
    
    click.echo(f"📂 Processing {len(crop_files)} crop files...")
    click.echo(f"🎯 Extracting layer {layer} ({layer_name})")
    
    extracted_count = 0
    skipped_count = 0
    
    for crop_file in crop_files:
        try:
            img = tifffile.imread(str(crop_file))
            
            # Extract specified layer
            if len(img.shape) == 3 and img.shape[0] > layer:
                extracted_layer = img[layer]
            elif len(img.shape) == 2:
                # Single layer image
                if layer == 0:
                    extracted_layer = img
                else:
                    skipped_count += 1
                    continue
            else:
                skipped_count += 1
                continue
            
            # Save extracted layer
            output_file = output_path / crop_file.name
            tifffile.imwrite(str(output_file), extracted_layer)
            extracted_count += 1
            
        except Exception as e:
            click.echo(
                click.style(f"❌ Error processing {crop_file.name}: {e}", fg="red")
            )
            skipped_count += 1
    
    click.echo(
        click.style(
            f"\n✅ Extracted {extracted_count} crops successfully",
            fg="green",
            bold=True,
        )
    )
    if skipped_count > 0:
        click.echo(click.style(f"⚠️  Skipped {skipped_count} crops", fg="yellow"))
    click.echo(f"📁 Output directory: {output_path}")


if __name__ == "__main__":
    cli()
