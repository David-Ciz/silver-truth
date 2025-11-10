import math
import tifffile
import random
import src.data_processing.utils.dataset_dataframe_creation as ddc


def add_split_column_to_parquet(parquet_path: str, seed: int, set_splits: list[float], group_by: str=""):
    # must sum to 1
    assert (math.fsum(set_splits))
    # must have 3 floats for train/val/test
    assert (len(set_splits) == 3)

    train_percent, val_percent, _ = set_splits
    
    # load dataframe
    df = ddc.load_dataframe_from_parquet_with_metadata(parquet_path)

    do_group_by = group_by != ""

    # check that the group_by is empty or a name of a column
    assert(not do_group_by or group_by in df.columns)

    # find how many "groups"
    groups_count = df[group_by].nunique() if do_group_by else df.shape[0]
    train_count = int(groups_count * train_percent)
    val_count = int(groups_count * val_percent)
    test_count = groups_count - train_count - val_count
    # make sure the result makes sense
    assert(train_count > 0 and val_count > 0 and test_count > 0)
    assert(train_count > val_count + test_count)

    splits = (["train"] * train_count) + (["validation"] * val_count) + (["test"] * test_count)
    # suffle
    random.Random(seed).shuffle(splits)

    split_cpl_name = "split"
    if not do_group_by:
        df[split_cpl_name] = splits
    else:
        pass
        
        #for index, row in enumerate(df.itertuples()):
        #    # load the image
        #    composed_image = tifffile.imread(row.image_path)  # type: ignore