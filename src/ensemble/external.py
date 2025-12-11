import src.data_processing.compression as dp_comp
import src.qa.preprocessing as  qa_pp
import src.data_processing.utils.dataset_dataframe_creation as ddc


def compress_images(image_folder, recursive=True):
    dp_comp.compress_tifs_logic(image_folder, recursive, False, False)


def build_qa_dataset(
    dataset_dataframe_path: str,
    output_dir: str,
    output_parquet_path: str,
):
    qa_pp.create_qa_dataset(dataset_dataframe_path, 
                            output_dir,
                            output_parquet_path, 
                            crop=True, 
                            crop_size=64)


def load_parquet(input_path: str):
    return ddc.load_dataframe_from_parquet_with_metadata(input_path)