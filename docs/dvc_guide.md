# DVC Guide for Silver Truth Repository

This guide will help you set up DVC (Data Version Control) and download the QA datasets used in this project.

## Prerequisites

- Git repository cloned
- Python virtual environment activated
- Project dependencies installed (see README.md)

## Initial Setup

### 1. Install Project Requirements

First, ensure you have all project dependencies installed:

```bash
# Navigate to project root
cd silver-truth

# Create and activate virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# or .venv\Scripts\activate on Windows

# Install project in editable mode with all dependencies
pip install -e .[dev]
```

This will install DVC along with all other required packages.

### 2. Configure DVC Remote

You need to configure DVC to connect to the HPC storage where the data is stored. Add the following configuration to your local DVC config:

```bash
# Configure the remote storage
dvc remote add hpc_storage ssh://karolina.it4i.cz/mnt/proj1/eu-25-40/innovaite/dvc_store
dvc remote default hpc_storage

# Set cache to be shared across the team
dvc cache dir .dvc/cache
dvc config cache.shared group
```

**Alternative: Manual Configuration**

You can also manually edit `.dvc/config` and add:

```ini
[core]
    remote = hpc_storage
[cache]
    shared = group
['remote "hpc_storage"']
    url = ssh://karolina.it4i.cz/mnt/proj1/eu-25-40/innovaite/dvc_store
```

### 3. SSH Access Setup

Ensure you have SSH access to the HPC storage:

```bash
# Test SSH connection
ssh karolina.it4i.cz

# If you need to set up SSH keys for passwordless access:
ssh-copy-id karolina.it4i.cz
```

## Downloading QA Data

### Understanding the QA Data Structure

The QA datasets are organized by:
- **Dataset name**: `BF-C2DL-HSC`, `BF-C2DL-MuSC`
- **Split type**: `mixed` (70-15-15 train-val-test split)
- **Crop size**: Defined in `params.yaml` (e.g., `sz64` for 64x64 crops)

Each QA dataset contains:
- Cropped images in:
  - `data/qa_crops/{DATASET}/sz{CROP_SIZE}/`
- Split-specific QA parquet files in:
  - `data/dataframes/{DATASET}/qa_crops/*_sz{CROP_SIZE}_qa_dataset.parquet`
  - These contain metadata, `split`, and evaluation labels (`jaccard_score`, `f1_score`).

QA crop TIFF channel layout is `(4, H, W)`:
- `0`: raw image
- `1`: competitor segmentation mask
- `2`: GT mask
- `3`: tracking marker mask

### Pull All Data

To download everything tracked by DVC:

```bash
dvc pull
```

⚠️ **Warning**: This may download a large amount of data. Use selective pulling if you only need specific datasets.

### Pull Specific QA Dataset

To download QA data for a specific dataset:

```bash
# For BF-C2DL-HSC with 64x64 crops (images + dataframes)
dvc pull data/qa_crops/BF-C2DL-HSC/sz64/
dvc pull data/dataframes/BF-C2DL-HSC/qa_crops/mixed_sz64_qa_dataset.parquet

# For BF-C2DL-MuSC with 64x64 crops (images + dataframes)
dvc pull data/qa_crops/BF-C2DL-MuSC/sz64/
dvc pull data/dataframes/BF-C2DL-MuSC/qa_crops/mixed_sz64_qa_dataset.parquet
```

**Note**: Check `params.yaml` to see the configured crop size for each dataset:

```yaml
datasets:
  BF-C2DL-HSC:
    crop_size: 64
  BF-C2DL-MuSC:
    crop_size: 64
```

### Access the Dataset Parquet

After pulling, the parquet file will be available at:

```
data/dataframes/{DATASET_NAME}/qa_crops/mixed_sz{CROP_SIZE}_qa_dataset.parquet
```

Example:
```
data/dataframes/BF-C2DL-HSC/qa_crops/mixed_sz64_qa_dataset.parquet
```

You can load this in Python:

```python
import pandas as pd

# Load the QA dataset
df = pd.read_parquet('data/dataframes/BF-C2DL-HSC/qa_crops/mixed_sz64_qa_dataset.parquet')
print(df.head())
```

### Pull Other Data Files

For other datasets used in the pipeline:

```bash
# Pull synchronized data
dvc pull data/synchronized_data/

# Pull dataset dataframes
dvc pull data/dataframes/

# Pull specific parquet files
dvc pull data/BF_C2DL-HSC_QA_with_fused_split70-15-15_seed42.parquet
```

## Working with DVC Pipelines

### View Pipeline Stages

To see all available DVC pipeline stages:

```bash
dvc dag
```

### Reproduce Pipeline Stages

To regenerate data by running DVC stages:

```bash
# Reproduce everything
dvc repro

# Reproduce specific stage
dvc repro create_qa_crops_split_mixed
```

`create_qa_crops_split_*` stages run both split attachment and QA label generation
(`jaccard_score`, `f1_score`) for cropped data.

### Check Pipeline Status

To see which stages need to be re-run:

```bash
dvc status
```

## Contributing Data

When you generate new datasets, models, or other large files, you need to add them to DVC and share them with the team.

### Adding New Data Files

Use `dvc add` to track large files with DVC instead of Git:

```bash
# Add a single file
dvc add data/new_dataset.parquet

# Add a directory
dvc add data/qa_crops/BF-C2DL-NewDataset/

# Add a trained model
dvc add models/resnet50_new_experiment.pt
```

**What happens when you run `dvc add`:**
1. DVC copies the file to `.dvc/cache`
2. Creates a `.dvc` file (e.g., `new_dataset.parquet.dvc`) with metadata
3. Adds the original file to `.gitignore`

### Committing DVC Files to Git

⚠️ **Important**: After `dvc add`, you MUST commit the `.dvc` file to Git:

```bash
# After dvc add, Git will see the new .dvc file
git status

# Add the .dvc file and updated .gitignore
git add data/new_dataset.parquet.dvc .gitignore

# Commit to Git
git commit -m "Add new QA dataset for BF-C2DL-HSC"
```

### Pushing Data to Remote Storage

After adding files with DVC, push them to the HPC storage so your team can access them:

```bash
# Push all new data
dvc push

# Push specific file
dvc push data/new_dataset.parquet.dvc
```

### Complete Workflow Example

Here is a typical workflow for refreshing QA crops and labels:

```bash
# 1. Reproduce QA split parquet + labels through DVC
dvc repro create_qa_crops_split_mixed

# 2. Push pipeline outputs to DVC remote
dvc push

# 3. Commit pipeline state
git add dvc.lock
git commit -m "Update QA crops split/labels"
git push origin your-branch
```

### Updating Existing Data

If you modify a file that's already tracked by DVC:

```bash
# Modify the file
python scripts/update_dataset.py

# Update DVC tracking (this updates the .dvc file)
dvc add data/existing_dataset.parquet

# Commit the updated .dvc file
git add data/existing_dataset.parquet.dvc
git commit -m "Update dataset with new labels"

# Push both to DVC remote and Git
dvc push
git push
```

### Working with DVC Pipeline Outputs

If you generate data through DVC pipelines (defined in `dvc.yaml`), you don't need to manually `dvc add`:

```bash
# Run the pipeline
dvc repro

# Push pipeline outputs
dvc push

# Commit dvc.lock which tracks pipeline state
git add dvc.lock
git commit -m "Update pipeline outputs"
git push
```

## Common DVC Commands

| Command | Description |
|---------|-------------|
| `dvc pull` | Download data from remote storage |
| `dvc push` | Upload data to remote storage |
| `dvc status` | Check pipeline status |
| `dvc repro` | Reproduce pipeline stages |
| `dvc dag` | Show pipeline dependency graph |
| `dvc list --dvc-only .` | List all DVC-tracked files |

## Troubleshooting

### Permission Denied (SSH)

If you get SSH permission errors:

1. Verify SSH access: `ssh karolina.it4i.cz`
2. Check SSH keys are properly configured
3. Contact your team lead for HPC storage access

### Large Download Times

If downloads are slow:

- Use selective pulling instead of `dvc pull`
- Check your network connection
- Consider using `--jobs` flag: `dvc pull --jobs 4`

### Cache Issues

If you encounter cache corruption:

```bash
# Verify cache integrity
dvc cache verify

# Clear local cache and re-download
rm -rf .dvc/cache
dvc pull
```

### File Not Found

If a file is missing after `dvc pull`:

1. Check expected output paths:
   - `ls data/qa_crops/BF-C2DL-HSC/sz64/`
   - `ls data/dataframes/BF-C2DL-HSC/qa_crops/mixed_sz64_qa_dataset.parquet`
2. Verify remote configuration: `dvc remote list`
3. Check if file exists on remote: Contact team lead

## Best Practices

1. **Don't commit large files to Git**: Always use `dvc add` for data files
2. **Pull before you start**: Run `dvc pull` to ensure you have the latest data
3. **Push your changes**: After running pipelines, use `dvc push` to share results
4. **Track parameters**: Changes to `params.yaml` will trigger pipeline reruns
5. **Use selective pulls**: Only download the data you need for your task

## Next Steps

- Read about [QA Data Preparation](qa_data_preparation.md)
- Learn about [Quality Assurance](quality_assurance.md)
- Explore [Experiment Tracking](experiment_tracking.md)

## Support

If you encounter issues not covered here:

1. Check the [DVC documentation](https://dvc.org/doc)
2. Contact your team lead
3. Open an issue in the repository
