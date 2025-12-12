import src.data_processing.compression as dp_comp
import src.qa.preprocessing as  qa_pp
import src.data_processing.utils.dataset_dataframe_creation as ddc


def compress_images(image_folder, recursive=True):
    dp_comp.compress_tifs_logic(image_folder, recursive, False, False)


def load_parquet(input_path: str):
    return ddc.load_dataframe_from_parquet_with_metadata(input_path)