"""
Unit tests for src/io_utils.py - Image loading, EXIF parsing, and sorting.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
import pytest

from src.io_utils import (
    IMAGE_EXTENSIONS,
    ImageInfo,
    extract_exif_data,
    get_image_dimensions,
    get_image_files,
    get_image_info,
    load_image,
    load_images,
    parse_exif_datetime,
    save_image,
    sort_images_robustly,
)


class TestImageExtensions:
    """Tests for supported image extensions."""
    
    def test_common_extensions_supported(self):
        """Test that common image formats are supported."""
        assert '.jpg' in IMAGE_EXTENSIONS
        assert '.jpeg' in IMAGE_EXTENSIONS
        assert '.png' in IMAGE_EXTENSIONS
        assert '.tiff' in IMAGE_EXTENSIONS
        assert '.bmp' in IMAGE_EXTENSIONS


class TestGetImageFiles:
    """Tests for get_image_files function."""
    
    def test_finds_image_files(self, test_images_dir):
        """Test finding image files in directory."""
        files = get_image_files(test_images_dir)
        assert len(files) > 0
        assert all(f.suffix.lower() in IMAGE_EXTENSIONS for f in files)
    
    def test_raises_on_missing_directory(self, temp_dir):
        """Test error on non-existent directory."""
        with pytest.raises(FileNotFoundError):
            get_image_files(temp_dir / "nonexistent")
    
    def test_raises_on_empty_directory(self, temp_dir):
        """Test error on directory with no images."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No image files"):
            get_image_files(empty_dir)
    
    def test_ignores_non_image_files(self, temp_dir):
        """Test that non-image files are ignored."""
        # Create mixed files - only non-image extensions
        (temp_dir / "text.txt").write_text("hello")
        (temp_dir / "data.json").write_text("{}")
        (temp_dir / "script.py").write_text("# python")
        
        # Create a real image for the test
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(temp_dir / "real.jpg"), img)
        
        files = get_image_files(temp_dir)
        assert len(files) == 1
        assert files[0].name == "real.jpg"


class TestParseExifDatetime:
    """Tests for parse_exif_datetime function."""
    
    def test_parses_standard_datetime(self):
        """Test parsing standard EXIF datetime format."""
        tags = {'EXIF DateTimeOriginal': MagicMock(__str__=lambda x: '2024:06:15 14:30:45')}
        dt = parse_exif_datetime(tags)
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 6
        assert dt.day == 15
        assert dt.hour == 14
        assert dt.minute == 30
        assert dt.second == 45
    
    def test_handles_subseconds(self):
        """Test parsing datetime with subseconds."""
        tags = {
            'EXIF DateTimeOriginal': MagicMock(__str__=lambda x: '2024:06:15 14:30:45'),
            'EXIF SubSecTimeOriginal': MagicMock(__str__=lambda x: '123')
        }
        dt = parse_exif_datetime(tags)
        assert dt is not None
        assert dt.microsecond > 0
    
    def test_returns_none_on_missing(self):
        """Test returns None when no datetime tag."""
        tags = {}
        dt = parse_exif_datetime(tags)
        assert dt is None
    
    def test_handles_invalid_format(self):
        """Test handling invalid datetime format."""
        tags = {'EXIF DateTimeOriginal': MagicMock(__str__=lambda x: 'invalid')}
        dt = parse_exif_datetime(tags)
        assert dt is None


class TestImageInfo:
    """Tests for ImageInfo dataclass."""
    
    def test_sort_key_with_timestamp(self, temp_dir):
        """Test sort key prioritizes EXIF timestamp."""
        info = ImageInfo(
            path=temp_dir / "test.jpg",
            timestamp=datetime(2024, 6, 15, 14, 30, 0),
            mtime=1000000.0
        )
        key = info.sort_key
        assert key[0] == 0  # Priority for EXIF
    
    def test_sort_key_with_mtime(self, temp_dir):
        """Test sort key uses mtime when no timestamp."""
        info = ImageInfo(
            path=temp_dir / "test.jpg",
            timestamp=None,
            mtime=1000000.0
        )
        key = info.sort_key
        assert key[0] == 1  # Priority for mtime
    
    def test_sort_key_fallback(self, temp_dir):
        """Test sort key fallback when no timestamp or mtime."""
        info = ImageInfo(
            path=temp_dir / "test.jpg",
            timestamp=None,
            mtime=None
        )
        key = info.sort_key
        assert key[0] == 2  # Lowest priority


class TestLoadImage:
    """Tests for load_image function."""
    
    def test_loads_image(self, test_images_dir):
        """Test loading an image file."""
        files = list(test_images_dir.glob("*.jpg"))
        img = load_image(files[0])
        assert img is not None
        assert len(img.shape) == 3
        assert img.shape[2] == 3  # BGR
    
    def test_downscales_image(self, test_images_dir):
        """Test image downscaling."""
        files = list(test_images_dir.glob("*.jpg"))
        img_full = load_image(files[0])
        img_scaled = load_image(files[0], max_width=400)
        
        assert img_scaled.shape[1] == 400
        assert img_scaled.shape[1] < img_full.shape[1]
    
    def test_raises_on_invalid_file(self, temp_dir):
        """Test error on invalid image file."""
        invalid_path = temp_dir / "invalid.jpg"
        invalid_path.write_text("not an image")
        
        with pytest.raises(ValueError, match="Could not load"):
            load_image(invalid_path)


class TestLoadImages:
    """Tests for load_images function."""
    
    def test_loads_multiple_images(self, test_images_dir):
        """Test loading multiple images."""
        files = list(test_images_dir.glob("*.jpg"))
        infos = [ImageInfo(path=f) for f in files]
        
        images = load_images(infos)
        assert len(images) == len(files)
        assert all(img is not None for img in images)


class TestGetImageDimensions:
    """Tests for get_image_dimensions function."""
    
    def test_gets_dimensions(self, test_images_dir):
        """Test getting image dimensions without loading."""
        files = list(test_images_dir.glob("*.jpg"))
        width, height = get_image_dimensions(files[0])
        
        # Load and compare
        img = cv2.imread(str(files[0]))
        assert width == img.shape[1]
        assert height == img.shape[0]


class TestSaveImage:
    """Tests for save_image function."""
    
    def test_saves_jpg(self, temp_dir, sample_image):
        """Test saving JPEG image."""
        output_path = temp_dir / "output.jpg"
        save_image(sample_image, output_path)
        
        assert output_path.exists()
        loaded = cv2.imread(str(output_path))
        assert loaded is not None
    
    def test_saves_png(self, temp_dir, sample_image):
        """Test saving PNG image."""
        output_path = temp_dir / "output.png"
        save_image(sample_image, output_path)
        
        assert output_path.exists()
        loaded = cv2.imread(str(output_path))
        assert loaded is not None
    
    def test_creates_parent_directories(self, temp_dir, sample_image):
        """Test automatic parent directory creation."""
        output_path = temp_dir / "nested" / "dir" / "output.jpg"
        save_image(sample_image, output_path)
        
        assert output_path.exists()


class TestSortImagesRobustly:
    """Tests for sort_images_robustly function."""
    
    def test_sorts_by_filename(self, test_images_dir):
        """Test sorting by filename when no EXIF."""
        files = list(test_images_dir.glob("*.jpg"))
        sorted_infos, method = sort_images_robustly(files)
        
        # Should use filename sorting for synthetic images
        assert method in ["filename_natural", "mtime"]
        assert len(sorted_infos) == len(files)
    
    def test_returns_image_infos(self, test_images_dir):
        """Test that results are ImageInfo objects."""
        files = list(test_images_dir.glob("*.jpg"))
        sorted_infos, _ = sort_images_robustly(files)
        
        assert all(isinstance(info, ImageInfo) for info in sorted_infos)
    
    def test_natural_sort_order(self, temp_dir):
        """Test natural sorting (image_2 before image_10)."""
        # Create test images with numeric names
        for name in ["image_1.jpg", "image_2.jpg", "image_10.jpg", "image_20.jpg"]:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(temp_dir / name), img)
        
        files = list(temp_dir.glob("*.jpg"))
        sorted_infos, _ = sort_images_robustly(files)
        
        # Natural sort should give: 1, 2, 10, 20
        names = [info.path.name for info in sorted_infos]
        assert names == ["image_1.jpg", "image_2.jpg", "image_10.jpg", "image_20.jpg"]

