import os
from PIL import Image
import click
import tifffile


def compress_tif_file(file_path, dryrun=False):
    """Compress a TIF file using LZW compression and overwrite the original file."""
    try:
        # Check if file is a TIFF
        if not file_path.lower().endswith((".tif", ".tiff")):
            return False, f"Skipped: {file_path} (not a TIFF file)"

        # Get file size before compression
        original_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

        if dryrun:
            return True, f"Would compress: {file_path} ({original_size:.2f}MB)"

        # Try to read with tifffile first
        try:
            img = tifffile.imread(file_path)

            # Create a temporary file
            temp_file = file_path + ".temp"

            # Save with LZW compression
            tifffile.imwrite(temp_file, img, compression="lzw")

            # Check if operation was successful
            if os.path.exists(temp_file):
                # Replace the original file
                os.replace(temp_file, file_path)
            else:
                return False, f"Failed: {file_path} (temporary file not created)"

        except Exception as e1:
            # Fall back to PIL if tifffile fails
            try:
                img = Image.open(file_path)

                # Create a temporary file
                temp_file = file_path + ".temp"

                # Save with LZW compression
                img.save(temp_file, compression="tiff_lzw")

                # Check if operation was successful
                if os.path.exists(temp_file):
                    # Replace the original file
                    os.replace(temp_file, file_path)
                else:
                    return False, f"Failed: {file_path} (temporary file not created)"

            except Exception as e2:
                return False, f"Failed: {file_path} (errors: {str(e1)} and {str(e2)})"

        # Get file size after compression
        new_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

        return (
            True,
            f"Compressed: {file_path} ({original_size:.2f}MB -> {new_size:.2f}MB, saved {original_size - new_size:.2f}MB)",
        )

    except Exception as e:
        return False, f"Error: {file_path} ({str(e)})"


def process_directory(directory, recursive=True, dryrun=False, verbose=False):
    """Process all TIF files in the given directory and its subdirectories if recursive is True."""
    success_count = 0
    total_count = 0
    total_saved_mb = 0
    errors = []

    # Get all TIFF files in the given directory
    tiff_files = []
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".tif", ".tiff")):
                    tiff_files.append(os.path.join(root, file))
    else:
        tiff_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith((".tif", ".tiff"))
        ]

    if not tiff_files:
        click.echo(f"No TIFF files found in {directory}")
        return

    click.echo(f"Found {len(tiff_files)} TIFF files to process")

    if dryrun:
        click.echo(
            click.style(
                "DRY RUN MODE: No files will be modified", fg="yellow", bold=True
            )
        )

    # Process each TIFF file with a progress bar
    with click.progressbar(tiff_files, label="Compressing TIF files") as bar:
        for file_path in bar:
            total_count += 1

            # Get file size before compression
            original_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

            success, message = compress_tif_file(file_path, dryrun)

            if success:
                success_count += 1
                # Calculate saved space if not in dry run
                if not dryrun:
                    new_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    saved_mb = original_size - new_size
                    total_saved_mb += saved_mb
                if verbose:
                    click.echo(message)
            else:
                errors.append(message)
                if verbose:
                    click.echo(click.style(message, fg="red"))

    # Summary
    click.echo("\nCompression Summary:")
    click.echo(f"Total files processed: {total_count}")
    click.echo(
        f"Successfully {('would be ' if dryrun else '')}compressed: {success_count}"
    )

    if not dryrun:
        click.echo(
            click.style(f"Total space saved: {total_saved_mb:.2f} MB", fg="green")
        )

    if total_count - success_count > 0:
        click.echo(click.style(f"Failed: {total_count - success_count}", fg="red"))

    if errors and verbose:
        click.echo("\nErrors:")
        for error in errors:
            click.echo(f"  {error}")


def compress_tifs_logic(directory, recursive, dryrun, verbose):
    process_directory(directory, recursive, dryrun, verbose)
