import math
import tifffile
import random
import src.data_processing.utils.dataset_dataframe_creation as ddc


def create_parquet_with_split_column(parquet_path: str, seed: int, set_splits: list[float], group_by: str=""):
    # must sum to 1
    assert (math.fsum(set_splits) == 1.0)
    # must have 3 floats for train/val/test
    assert (len(set_splits) == 3)

    train_percent, val_percent, test_percent = set_splits
    
    # load dataframe
    df = ddc.load_dataframe_from_parquet_with_metadata(parquet_path)

    no_group = group_by == ""

    # check that the group_by is empty or a name of a column
    assert(no_group or group_by in df.columns)

    # find how many "groups"
    groups_count = df.shape[0] if no_group else df[group_by].nunique()
    train_count = round(groups_count * train_percent)
    val_count = round(groups_count * val_percent)
    test_count = groups_count - train_count - val_count
    # make sure the result makes sense
    assert(train_count > 0 and val_count > 0 and test_count > 0)
    assert(train_count > val_count + test_count)

    # split values
    splits = (["train"] * train_count) + (["validation"] * val_count) + (["test"] * test_count)
    # suffle
    random.Random(seed).shuffle(splits)

    split_col_name = "split"
    if no_group:
        df[split_col_name] = splits
    else:
        # create a mapping with unique gt file name and the corresponding split value
        mappings = dict(zip(df[group_by].unique(), splits))
        df[split_col_name] = df[group_by].map(mappings)    

    # save parquet
    parquet_suffix = f"_split{int(train_percent*100)}-{int(val_percent*100)}-{int(test_percent*100)}_seed{seed}"
    filetype_pos = parquet_path.rfind(".")
    assert(filetype_pos >= 0) # the '.' char must exist in the filename
    output_parquet_path = parquet_path[:filetype_pos] + parquet_suffix + parquet_path[filetype_pos:]
    df.to_parquet(output_parquet_path)



#TODO: a method that receives 2 parquet and checks if they have the same split (each gt and related segmentations must have direct correspondence)