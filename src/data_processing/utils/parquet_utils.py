import math
import numpy as np
import tifffile
import random
import src.data_processing.utils.dataset_dataframe_creation as ddc
from src.ensemble import utils

# Constants
SPLITS_COLUMN = "split" # column containing "train", "validation" or "test"
CELL_ID_COLUMNS = ["gt_image", "label",] # columns that uniquely indentify a cell from a QA parquet
COMMON_COLUMN = "crop_size" # column common to both QA and Ensemble parquets


def add_split_type(parquet_path: str, build_opt: dict) -> str:
    """
    Create a parquet with an added column of random train/validation/test splits, with the given seed.
    Returns the path to the newly created file.
    """
    # must sum to 1
    assert (math.fsum(build_opt["split_sets"]) == 1.0)
    # must have 3 floats for train/val/test
    assert (len(build_opt["split_sets"]) == 3)

    train_percent, val_percent, test_percent = build_opt["split_sets"]
    
    # load dataframe
    df = ddc.load_dataframe_from_parquet_with_metadata(parquet_path)

    assert(COMMON_COLUMN in df.columns) # must be a parquet from either QA ot Ensemble
    is_ensemple = not set(CELL_ID_COLUMNS).issubset(df.columns)

    # find how many "groups" (competitors segmentation of each cell)
    groups_count = df.shape[0] if is_ensemple else df[CELL_ID_COLUMNS].drop_duplicates().shape[0]
    train_count = round(groups_count * train_percent)
    val_count = round(groups_count * val_percent)
    test_count = groups_count - train_count - val_count
    # make sure the result makes sense
    assert(train_count > 0 and val_count > 0 and test_count > 0)
    assert(train_count > val_count + test_count)

    # split values
    splits = (["train"] * train_count) + (["validation"] * val_count) + (["test"] * test_count)
    # suffle
    random.Random(build_opt["split_seed"]).shuffle(splits)

    if is_ensemple:
        df[SPLITS_COLUMN] = splits
    else:
        # create a mapping between each unique "cell group" and the corresponding split value
        uniques_col = ",".join(CELL_ID_COLUMNS)
        df[uniques_col] = df[CELL_ID_COLUMNS[0]] + df[CELL_ID_COLUMNS[1]].astype(str)
        mappings = dict(zip(df[uniques_col].drop_duplicates().to_numpy(), splits))
        df[SPLITS_COLUMN] = df[uniques_col].map(mappings)
        df.drop(columns=[uniques_col])

    # save parquet
    parquet_suffix = "_split" + utils.get_splits_name(build_opt)
    filetype_pos = parquet_path.rfind(".")
    assert(filetype_pos >= 0) # the '.' char must exist in the filename
    output_parquet_path = parquet_path[:filetype_pos] + parquet_suffix + parquet_path[filetype_pos:]
    df.to_parquet(output_parquet_path)
    return output_parquet_path


def _get_uniques(parquet_path) -> np.ndarray:
    # load dataframe
    df = ddc.load_dataframe_from_parquet_with_metadata(parquet_path)
    uniques_col = ",".join(CELL_ID_COLUMNS)
    df[uniques_col] = df[CELL_ID_COLUMNS[0]] + df[CELL_ID_COLUMNS[1]].apply(lambda x: "%04d" % (x))
    uniques = df[[uniques_col, SPLITS_COLUMN]].drop_duplicates().sort_values(by=uniques_col)
    return uniques[SPLITS_COLUMN].to_numpy()


def same_splits(parquet1_path: str, parquet2_path: str) -> bool:
    """
    Check if the 2 parquets have the same splits.
    """
    uniques1 =_get_uniques(parquet1_path)
    uniques2 =_get_uniques(parquet2_path)

    return bool((uniques1==uniques2).all())
