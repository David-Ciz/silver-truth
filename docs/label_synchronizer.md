# Label Synchronization Process

## Overview

This documentation describes the process of synchronizing labels between silver truth, competitors, ground truth, and tracking files using standalone cell tracking challenge Java application. **Note:** We are no longer using a separate Fiji installation (except for when packaged inside the JAR file). The entire synchronization functionality is now contained within the JAR file.

## Prerequisites

- A Java Runtime Environment (JRE) installed.
- The standalone JAR file for the 'Annotation Label Sync2' plugin, which contains all necessary dependencies.
- Input data organized in the dataset structure.
- Large amount of disk space for the output data, the synchronizer does not compress the data.

## Obtaining the JAR File
To generate the JAR file, follow these steps:

### Step 1: Clone the Repository

Clone the repository containing the source code for the 'Annotation Label Sync2' plugin.

````bash
git clone https://github.com/David-Ciz/label-fusion-ng-fork
````

### Step 2: Build the JAR File

Navigate to the root directory of the cloned repository and run the following command to build the JAR file:

````bash
mvn clean package
````

The JAR file will be generated in the `target` directory.

## Running the Synchronization

### Using python interface

You can use the provided Python wrapper script to run the synchronization process in the preprocessing.py file. The script reads the configuration file and runs the synchronization process using the standalone JAR. 

````bash
python preprocessing.py synchronize_dataset data/inputs-2020-07 data/synchronize_data
````
**Notes:**
- Replace `data/inputs-2020-07` with the path to the input data directory (it should contain folders with datasets).
- Replace `data/synchronize_data` with the path to the output directory.

### Using the Command Line

Since we have moved the synchronization functionality into a standalone JAR file, the synchronization can now be run via a standard Java command without needing a separate Fiji installation. Use the following command:

````bash
java -cp target/LabelSyncer2Runner-1.0-SNAPSHOT-jar-with-dependencies.jar de.mpicbg.ulman.fusion.RunLabelSyncer2 /absolute/path/to/segmentation/results /absolute/path/to/tra/markers /absolute/path/to/output
````

**Notes:**
- Ensure that all paths provided are absolute.
- Replace `/absolute/path/to/segmentation/results`, `/absolute/path/to/tra/markers`, and `/absolute/path/to/output` with the actual paths on your system.
- If you prefer, you can move the JAR to a convenient directory and adjust the command accordingly.
- It is highly recommended to compress the output data after synchronization to save disk space.



### Results

The synchronized data will be saved in the specified output directory with the following structure:

<pre>
output_dir/
├── Dataset1/
│   ├── Competitor1/
│   │   ├── t001.tif
│   ├── Competitor2/
│   │   ├── t001.tif
│   └── ...
</pre>


## Validation

To verify a successful synchronization:
- Check that the number of output files matches the input.
- Verify that labels are consistent across temporal sequences.
- Confirm that the `metadata.json` file contains the expected information.

## Common Issues and Solutions

- **Missing Output Files:**  
  *Possible Cause:* Incorrect input paths.  
  *Solution:* Verify that the input directories are correct and contain valid data.

- **Inconsistent Labels:**  
  *Possible Cause:* TRA markers mismatch.  
  *Solution:* Check that the TRA marker files follow the expected format.

- **Plugin Error:**  
  *Possible Cause:* Java version mismatch or dependency issues.  
  *Solution:* Ensure you are using a compatible Java version and that all dependencies are included in the JAR file.

## Version History

Date       | Version | Changes                              | Author  
-----------|---------|--------------------------------------|---------
2025-03-27 | 1.0     | Initial documentation update         | David Číž

## References

- [Java Documentation](https://docs.oracle.com/)
- [Annotation Label Sync2 Plugin Documentation](#)
- [Related Publications](#)
- https://github.com/CellTrackingChallenge/label-fusion-ng/tree/master

```markdown