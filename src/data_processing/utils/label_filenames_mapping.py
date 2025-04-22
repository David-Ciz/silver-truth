import logging
import pathlib
import re


def map_label_file_names(first_list: list[str], second_list: list[str]):
    """
    This function maps two lists of file names containing labels to each other.
    """

    # Check if the lists are of the same length
    if len(first_list) != len(second_list):
        logging.warning(
            f"The two lists are not of the same length, first list contains {len(first_list)} and second list {len(second_list)} number of filenames."
        )
        smaller_list = first_list if len(first_list) < len(second_list) else second_list
    else:
        smaller_list = first_list
        logging.info(f"Both lists contain {len(first_list)} number of filenames.")

    # strip the file extensions and chars from the file names
    first_list_numbers_mapping = get_filename_numbers_mapping(first_list)
    second_list_numbers_mapping = get_filename_numbers_mapping(second_list)

    label_mappings = {}
    # Compare the two lists
    for key in first_list_numbers_mapping:
        if key in second_list_numbers_mapping:
            logging.info(
                f"File {first_list_numbers_mapping[key]} found in the second list."
            )
            label_mappings[first_list_numbers_mapping[key]] = (
                second_list_numbers_mapping[key]
            )
        if key not in second_list_numbers_mapping:
            logging.warning(
                f"File {first_list_numbers_mapping[key]} not found in the second list."
            )

    return label_mappings


def get_filename_numbers_mapping(filename_list: list[str]) -> dict:
    """
    This function extracts the numbers from the filenames and maps them to the filenames.
    """
    list_mapping = {}
    for file in filename_list:
        filename = pathlib.Path(file).stem
        file_number = re.sub(r"[\D\s]", "", file)
        list_mapping[file_number] = filename
    return list_mapping