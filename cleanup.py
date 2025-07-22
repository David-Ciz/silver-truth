#!/usr/bin/env python3
"""
Cleanup script to remove generated files from the silver-truth pipeline.

This script removes:
- Dataset dataframe files (dataframes/*.parquet or *.parquet)
- Job files (job_files/*_job_file.txt)
- Fusion results (fused_results/*_fused_*.tif)

Usage:
    python cleanup.py [OPTIONS]

Options:
    --dataframes    Only remove dataframe files
    --jobfiles      Only remove job files
    --fusion        Only remove fusion results
    --dry-run       Show what would be deleted without actually deleting
    --confirm       Ask for confirmation before deleting each type
"""

import os
import glob
import argparse
import sys
from pathlib import Path

def find_files_to_delete():
    """Find all files that would be deleted."""
    files_to_delete = {
        'dataframes': [],
        'jobfiles': [],
        'fusion': []
    }
    
    # Find dataframe files (check both root directory and dataframes folder)
    dataframe_patterns = [
        "*_dataset_dataframe.parquet",           # Legacy: root directory
        "dataframes/*_dataset_dataframe.parquet" # New: dataframes folder
    ]
    for pattern in dataframe_patterns:
        files_to_delete['dataframes'].extend(glob.glob(pattern))
    
    # Find job files
    jobfiles_pattern = "job_files/*_job_file.txt"
    files_to_delete['jobfiles'] = glob.glob(jobfiles_pattern)
    
    # Find fusion results
    fusion_pattern = "fused_results/*_fused_*.tif"
    files_to_delete['fusion'] = glob.glob(fusion_pattern)
    
    return files_to_delete

def format_file_size(size_bytes):
    """Format file size in human readable format."""
    if size_bytes == 0:
        return "0 B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def calculate_total_size(file_list):
    """Calculate total size of files in list."""
    total_size = 0
    for file_path in file_list:
        if os.path.exists(file_path):
            total_size += os.path.getsize(file_path)
    return total_size

def print_summary(files_to_delete):
    """Print summary of files to be deleted."""
    print("=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    
    total_files = 0
    total_size = 0
    
    for category, files in files_to_delete.items():
        if files:
            category_size = calculate_total_size(files)
            total_size += category_size
            total_files += len(files)
            
            print(f"\n📁 {category.upper()}:")
            print(f"   Files: {len(files)}")
            print(f"   Size:  {format_file_size(category_size)}")
            
            # Show first few files as examples
            for i, file_path in enumerate(files[:3]):
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                print(f"   - {os.path.basename(file_path)} ({format_file_size(file_size)})")
            
            if len(files) > 3:
                print(f"   ... and {len(files) - 3} more files")
    
    print(f"\n📊 TOTAL:")
    print(f"   Files: {total_files}")
    print(f"   Size:  {format_file_size(total_size)}")
    print("=" * 60)

def delete_files(file_list, category_name, dry_run=False, confirm=False):
    """Delete files in the list."""
    if not file_list:
        print(f"ℹ️  No {category_name} files found.")
        return 0
    
    if confirm:
        response = input(f"🗑️  Delete {len(file_list)} {category_name} files? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print(f"⏭️  Skipping {category_name} files.")
            return 0
    
    deleted_count = 0
    total_size = 0
    
    for file_path in file_list:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            if dry_run:
                print(f"🔍 Would delete: {file_path} ({format_file_size(file_size)})")
            else:
                try:
                    os.remove(file_path)
                    print(f"✅ Deleted: {os.path.basename(file_path)} ({format_file_size(file_size)})")
                    deleted_count += 1
                except OSError as e:
                    print(f"❌ Failed to delete {file_path}: {e}")
        else:
            if dry_run:
                print(f"⚠️  File not found: {file_path}")
    
    if not dry_run and deleted_count > 0:
        print(f"✅ Successfully deleted {deleted_count} {category_name} files ({format_file_size(total_size)})")
    
    return deleted_count

def main():
    parser = argparse.ArgumentParser(
        description="Clean up generated files from silver-truth pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cleanup.py                    # Delete all generated files
  python cleanup.py --dry-run          # Show what would be deleted
  python cleanup.py --dataframes       # Only delete dataframe files
  python cleanup.py --jobfiles         # Only delete job files
  python cleanup.py --fusion           # Only delete fusion results
  python cleanup.py --confirm          # Ask for confirmation
        """
    )
    
    parser.add_argument('--dataframes', action='store_true',
                       help='Only remove dataframe files')
    parser.add_argument('--jobfiles', action='store_true',
                       help='Only remove job files')
    parser.add_argument('--fusion', action='store_true',
                       help='Only remove fusion results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--confirm', action='store_true',
                       help='Ask for confirmation before deleting each type')
    
    args = parser.parse_args()
    
    # Find all files to delete
    files_to_delete = find_files_to_delete()
    
    # Filter based on arguments
    if args.dataframes or args.jobfiles or args.fusion:
        filtered_files = {}
        if args.dataframes:
            filtered_files['dataframes'] = files_to_delete['dataframes']
        if args.jobfiles:
            filtered_files['jobfiles'] = files_to_delete['jobfiles']
        if args.fusion:
            filtered_files['fusion'] = files_to_delete['fusion']
        files_to_delete = filtered_files
    
    # Check if any files found
    total_files = sum(len(files) for files in files_to_delete.values())
    if total_files == 0:
        print("ℹ️  No files found to delete.")
        return
    
    # Print summary
    print_summary(files_to_delete)
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No files will be actually deleted")
        print("=" * 60)
    
    # Delete files
    total_deleted = 0
    for category, files in files_to_delete.items():
        if files:
            deleted = delete_files(files, category, args.dry_run, args.confirm)
            total_deleted += deleted
    
    # Final summary
    if not args.dry_run:
        if total_deleted > 0:
            print(f"\n🎉 Cleanup completed! Deleted {total_deleted} files.")
        else:
            print(f"\n⚠️  No files were deleted.")
    else:
        print(f"\n📋 Dry run completed. Found {total_files} files that would be deleted.")

if __name__ == "__main__":
    main()
