# Ensemble

This section presents a study on different processes intended to replace the Fusion step on the silver-truth workflow. 
The objective is to find the best performant process capable of generating an accurate segmentation of an image of a cell given several proposed segmentations.

## Deliverables

## Strategies
Specific startegies are related to different datasets.

- Quantifying the error from learning just from gt;
- Transfer learning, starting from gt;

- A layered sequence of new syntetic (slightly improved) dataset generation for training a new model;
    - this can be done with single segmentation (and a single model) and also normalized segmentations (with multiple models per layer);
    - for inferece, the input will follow through the sequence of trained models;

- Adding gt to the input, while training, for the model to learn not to modify such segmentation;

- Simple competitors normalization;

- Normalizations alongside the raw image;

- Basic segmentation for comparison;

- Use QA and non-QA databanks for comparison;

## Data

### Concepts:
- **Databank** is the folder where the images are gathered and the corresponding parquet file.
- **Dataset** is the specific data structure used for training models.

### Dataset versions
- **A1**: 
    - gt -> gt [1 input-> 1 output], ground truth to ground truth;
    - can be used for an initial training stage before transfer to another dataset;
    - does this strategy offer benefits?
- **A2**:
    - gt&raw -> gt [2->1], ground truth along side raw image to ground truth;
    - for other [2->1] models.
- **B1**:
    - seg -> gt [1->1], segmentation to ground truth;
    - can we improve the images? 
        - If so, it may be used to create a syntetic dataset which can be used to train a new model.
 **B2**:
    - seg&raw -> gt [2->1], segmentation along side raw image to ground truth;
- **B3**: 
    - seg+gt -> gt [1->1], segmentation and ground truth to ground truth;
    - does adding some ground truths improve the results, compared with dataset B?
- **C1**: 
    - norm_seg -> gt [1->1], competitors normalized segmentation to ground truth.
- **C2**: 
    - norm_seg&raw -> gt [2->1], competitors normalized segmentation along side raw image to ground truth;
    - same as above but with an additional input;
    - may have some issues in cases where the raw images contain additional cells;
    - does it improve performance upon dataset D?
- **D1**: 
    - raw -> gt [1->1], raw image to ground truth;
    - may have more trouble with additional cells in the images - will have to learn to ignore everything around the centered structure;
    - what's the center of the image? This external information is already given by centering the image in the intended cell (so it's not a completely raw dataset); 
    - for ablation study to understand how much competitors segmentations improve our models.

### Steps:
1. Create QA and non-QA (for ablation study) databanks:
    - each databank folder has its corresponding parquet file on the same directory;
    - each parquet file contains, among other information:
        - a checksum for each file/image;
        - the name of each file.
    - the parquet files are saved in github (?).

2. Create Ensemble databank:
    - for each ground truth, create the input series (a sequence of images corresponding to the same cell: norm_seg, seg1, seg2...):
        - normalized competitors segmentations with ground truth, suffix: "_normseg";
            - layers RGB: [norm_seg, gt, raw];
        - single competitors segmentation with ground truth, suffix: "_seg[N]";
            - layers RGB: [seg, gt, raw];
    - each segmentation/raw image is centered according to the normalized segmentation image of the same series;
    - the parquet file also contains the name and checksum of each file;
    - the parquet files are saved in github (?).

3. Compute a split of the Ensemble parquet indices according to the different datasets:
    - each resulting set (train, val, test) must contain the same input series as the other sets of the same type;
        - any set of Dataset [X], whether with multiple files with the same ground truth or not, have direct correspondance to any set of the same type of a different Dataset.
            - example 1: all sets of Dataset A1 contain the same files as the Dataset C1;
            - example 2: all sets of Dataset B1 contains multiple files with the same ground truth, and in a direct correspondance with all sets of dataset A1.
    - steps:
        - I) load Ensemble parquet to a dataframe;
        - II) select only norm_seg images;
        - III) do a random split of the indices, according to a given seed value;
            - training (70%), validation (15%) and test(15%);
            - log the split details;
        - IV) create 3 dataframe with the content of the splitted indices;
            - these dataframes are used for the different sets of Dataset A1 and C1;
        - V) use this dataframe to create the other sets, following the same input series location;

4. Instantiate the different datasets sets, with the corresponding dataframe versions.
    - calculate the checksum for each image and assert with the corresponding checksum in the dataframe;



## Models

## Training

### Data augmentation

## Results
