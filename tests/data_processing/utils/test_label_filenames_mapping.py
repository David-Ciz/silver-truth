from src.data_processing.utils.label_filenames_mapping import (
    map_label_file_names,
    get_filename_numbers_mapping,
)


def test_get_filename_numbers_mapping_basic():
    """Test basic functionality of get_filename_numbers_mapping"""
    test_files = ["file1.txt", "file2.txt", "file3.txt"]
    expected = {"1": "file1", "2": "file2", "3": "file3"}
    assert get_filename_numbers_mapping(test_files) == expected


def test_get_filename_numbers_mapping_complex():
    """Test get_filename_numbers_mapping with more complex filenames"""
    test_files = ["label_123_test.txt", "image_456_sample.jpg", "data_789_final.png"]
    expected = {
        "123": "label_123_test",
        "456": "image_456_sample",
        "789": "data_789_final",
    }
    assert get_filename_numbers_mapping(test_files) == expected


def test_get_filename_numbers_mapping_empty():
    """Test get_filename_numbers_mapping with empty list"""
    assert get_filename_numbers_mapping([]) == {}


def test_map_label_file_names_equal_length():
    """Test mapping when lists are of equal length"""
    first_list = ["file1.txt", "file2.txt", "file3.txt"]
    second_list = ["label1.txt", "label2.txt", "label3.txt"]
    expected = {"file1": "label1", "file2": "label2", "file3": "label3"}
    assert map_label_file_names(first_list, second_list) == expected


def test_map_label_file_names_unequal_length():
    """Test mapping when lists are of unequal length"""
    first_list = ["file1.txt", "file2.txt"]
    second_list = ["label1.txt", "label2.txt", "label3.txt"]
    expected = {"file1": "label1", "file2": "label2"}
    assert map_label_file_names(first_list, second_list) == expected


def test_map_label_file_names_no_matches():
    """Test mapping when there are no matching numbers"""
    first_list = ["file1.txt", "file2.txt"]
    second_list = ["label3.txt", "label4.txt"]
    assert map_label_file_names(first_list, second_list) == {}


def test_map_label_file_names_complex_names():
    """Test mapping with complex file names"""
    first_list = ["image_001_rgb.jpg", "image_002_rgb.jpg"]
    second_list = ["label_001_seg.png", "label_002_seg.png"]
    expected = {"image_001_rgb": "label_001_seg", "image_002_rgb": "label_002_seg"}
    assert map_label_file_names(first_list, second_list) == expected


def test_map_label_file_names_empty_lists():
    """Test mapping with empty lists"""
    assert map_label_file_names([], []) == {}


def test_map_label_file_names_partial_matches():
    """Test mapping with partial matches"""
    first_list = ["file1.txt", "file2.txt", "file3.txt"]
    second_list = ["label2.txt", "label4.txt"]
    expected = {"file2": "label2"}
    assert map_label_file_names(first_list, second_list) == expected


def test_get_filename_numbers_mapping_no_numbers():
    """Test get_filename_numbers_mapping with filenames containing no numbers"""
    test_files = ["file_a.txt", "file_b.txt"]
    expected = {"": "file_b"}
    assert get_filename_numbers_mapping(test_files) == expected
