from pathlib import Path

import numpy as np
import pytest

from silver_truth.data_processing.compression import write_tiff_lossless


tifffile = pytest.importorskip("tifffile")


def test_write_tiff_lossless_preserves_array_and_uses_lzw(tmp_path: Path) -> None:
    image = np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8)
    output_path = tmp_path / "stacked.tif"

    write_tiff_lossless(output_path, image)

    loaded = tifffile.imread(output_path)
    assert loaded.shape == image.shape
    assert loaded.dtype == image.dtype
    np.testing.assert_array_equal(loaded, image)

    with tifffile.TiffFile(output_path) as tif:
        assert tif.pages[0].compression.name == "LZW"
