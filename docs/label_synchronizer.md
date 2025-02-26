# Label Synchronization Process

## Overview

This documentation describes the process of synchronizing labels between silver truth, competitors, ground truth, and tracking files using the ImageJ/Fiji plugin 'Annotation Label Sync2'. The synchronization ensures consistent labeling across different data sources and temporal sequences.

## Prerequisites

- [ImageJ/Fiji](https://fiji.sc/) installed
- 'Annotation Label Sync2' plugin installed in ImageJ/Fiji
- Input data organized in the specified structure

## Directory Structure

data/
├── raw/
│   ├── segmentation_results/     # Segmentation data to be synchronized
│   └── tra_markers/             # TRA marker files
└── processed/                   # Output synchronized data
## Input Data Requirements

### Segmentation Results
- Format: [specify format, e.g., TIFF sequences, individual masks]
- Naming convention: [specify if there's a required naming pattern]
- Expected structure: [describe how files should be organized]

### TRA Markers
- Format: [specify format]
- Required files: [list any specific files needed]
- File organization: [describe expected directory structure]

## Running the Synchronization

### Using ImageJ/Fiji GUI

1. Open ImageJ/Fiji
2. Navigate to `Plugins > Annotation Label Sync2`
3. In the dialog:
   - Select input segmentation folder
   - Select TRA markers folder
   - Choose output directory
4. Click "Run"

### Using Command Line (if applicable)

```bash
ImageJ-linux64 --headless --run "Annotation Label Sync2" \
  "input_dir='/path/to/segmentation',tra_dir='/path/to/tra',output_dir='/path/to/output'"
```
### Configuration

Create a config.yml file in the configs/ directory:paths:
  segmentation_dir: "data/raw/segmentation_results"
  tra_markers_dir: "data/raw/tra_markers"
  output_dir: "data/processed/synchronized"

fiji:
  executable_path: "/path/to/ImageJ-linux64"
  plugin_name: "Annotation Label Sync2"

logging:
  level: INFO
  file: "logs/label_sync.log"
Output StructureThe synchronized results will be saved in the specified output directory with the following structure:output_dir/
├── synchronized_masks/
│   ├── t000.tif
│   ├── t001.tif
│   └── ...
└── metadata.json
ValidationTo verify successful synchronization:
Check that the number of output files matches the input
Verify that labels are consistent across temporal sequences
Confirm that the metadata.json file contains expected information
Common Issues and SolutionsIssuePossible CauseSolutionMissing output filesIncorrect input pathsVerify input directory pathsInconsistent labelsTRA markers mismatchCheck TRA marker file formatPlugin errorImageJ version mismatchUpdate ImageJ/FijiProcessing ScriptThe synchronization can be automated using our Python wrapper script:from pathlib import Path
import subprocess
import yaml

def run_label_sync(config_path: str):
    """
    Run the label synchronization process using ImageJ/Fiji.
    
    Args:
        config_path: Path to the configuration YAML file
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Construct ImageJ command
    cmd = [
        config['fiji']['executable_path'],
        '--headless',
        '--run', config['fiji']['plugin_name'],
        f"input_dir='{config['paths']['segmentation_dir']}',"
        f"tra_dir='{config['paths']['tra_markers_dir']}',"
        f"output_dir='{config['paths']['output_dir']}'"
    ]
    
    # Run synchronization
    subprocess.run(cmd, check=True)
Usage Example# From the project root
python src/data_processing/label_synchronizer.py --config configs/label_sync_config.yml
Version HistoryDateVersionChangesAuthor[Current Date]1.0Initial documentation[Your Name]References
ImageJ/Fiji Documentation: [link]
Annotation Label Sync2 Plugin Documentation: [link]
Related Publications: [links]
SupportFor issues with:
The synchronization process: [contact information]
The ImageJ plugin: [plugin maintainer contact]
This documentation: [your contact]
