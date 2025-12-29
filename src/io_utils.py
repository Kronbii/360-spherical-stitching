"""
I/O utilities for image loading, EXIF parsing, and robust image sorting.
"""

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import exifread
import numpy as np
from natsort import natsorted
from PIL import Image
from PIL.ExifTags import TAGS

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}


@dataclass
class ImageInfo:
    """Information about an image file."""
    path: Path
    timestamp: Optional[datetime] = None
    mtime: Optional[float] = None
    exif_data: Optional[dict] = None
    sort_method: str = "unknown"
    
    @property
    def sort_key(self) -> Tuple:
        """Return sort key for ordering images."""
        # Priority: EXIF timestamp > mtime > filename
        if self.timestamp:
            return (0, self.timestamp, str(self.path))
        elif self.mtime:
            return (1, datetime.fromtimestamp(self.mtime), str(self.path))
        else:
            return (2, datetime.min, str(self.path))


def get_image_files(input_dir: Path) -> List[Path]:
    """
    Get all image files from a directory.
    
    Args:
        input_dir: Directory containing images.
        
    Returns:
        List of image file paths.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    images = []
    for file in input_dir.iterdir():
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)
    
    if not images:
        raise ValueError(f"No image files found in {input_dir}")
    
    logger.info(f"Found {len(images)} image files in {input_dir}")
    return images


def parse_exif_datetime(exif_tags: dict) -> Optional[datetime]:
    """
    Parse datetime from EXIF tags with subsecond precision if available.
    
    Args:
        exif_tags: Dictionary of EXIF tags from exifread.
        
    Returns:
        Parsed datetime or None.
    """
    # Try DateTimeOriginal first (when photo was taken)
    datetime_tag = None
    for tag_name in ['EXIF DateTimeOriginal', 'EXIF CreateDate', 'Image DateTime']:
        if tag_name in exif_tags:
            datetime_tag = str(exif_tags[tag_name])
            break
    
    if not datetime_tag:
        return None
    
    # Parse main datetime (format: "YYYY:MM:DD HH:MM:SS")
    try:
        dt = datetime.strptime(datetime_tag, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        logger.warning(f"Could not parse EXIF datetime: {datetime_tag}")
        return None
    
    # Try to add subsecond precision
    subsec = None
    for tag_name in ['EXIF SubSecTimeOriginal', 'EXIF SubSecTime']:
        if tag_name in exif_tags:
            subsec = str(exif_tags[tag_name])
            break
    
    if subsec:
        try:
            # Convert to microseconds (subsec is typically decimal fraction)
            subsec_float = float(f"0.{subsec}")
            dt = dt.replace(microsecond=int(subsec_float * 1_000_000))
        except ValueError:
            pass
    
    return dt


def extract_exif_data(image_path: Path) -> dict:
    """
    Extract relevant EXIF data from an image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Dictionary with relevant EXIF fields.
    """
    exif_data = {}
    
    # Use exifread for comprehensive EXIF parsing
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)
            
            # Extract relevant tags
            tag_mappings = {
                'EXIF FocalLengthIn35mmFilm': 'FocalLengthIn35mmFilm',
                'EXIF FocalLength': 'FocalLength',
                'EXIF ExifImageWidth': 'ExifImageWidth',
                'EXIF ExifImageLength': 'ExifImageHeight',
                'Image ImageWidth': 'ImageWidth',
                'Image ImageLength': 'ImageHeight',
                'EXIF DateTimeOriginal': 'DateTimeOriginal',
                'EXIF SubSecTimeOriginal': 'SubSecTimeOriginal',
            }
            
            for exif_tag, key in tag_mappings.items():
                if exif_tag in tags:
                    value = tags[exif_tag]
                    # Convert to appropriate type
                    try:
                        if 'FocalLength' in key and 'Film' not in key:
                            # FocalLength is a ratio
                            if hasattr(value, 'values') and value.values:
                                exif_data[key] = float(value.values[0])
                            else:
                                exif_data[key] = float(str(value).split('/')[0]) / float(str(value).split('/')[1]) if '/' in str(value) else float(value)
                        elif 'Width' in key or 'Height' in key or '35mm' in key:
                            exif_data[key] = int(str(value))
                        else:
                            exif_data[key] = str(value)
                    except (ValueError, IndexError, ZeroDivisionError):
                        exif_data[key] = str(value)
            
            exif_data['_raw_tags'] = tags
            
    except Exception as e:
        logger.debug(f"Could not read EXIF with exifread from {image_path}: {e}")
    
    # Also try Pillow as fallback for some tags
    try:
        with Image.open(image_path) as img:
            pillow_exif = img._getexif()
            if pillow_exif:
                for tag_id, value in pillow_exif.items():
                    tag_name = TAGS.get(tag_id, tag_id)
                    if tag_name == 'FocalLengthIn35mmFilm' and 'FocalLengthIn35mmFilm' not in exif_data:
                        exif_data['FocalLengthIn35mmFilm'] = int(value)
                    elif tag_name == 'FocalLength' and 'FocalLength' not in exif_data:
                        if hasattr(value, 'numerator'):
                            exif_data['FocalLength'] = value.numerator / value.denominator
                        else:
                            exif_data['FocalLength'] = float(value)
    except Exception as e:
        logger.debug(f"Could not read EXIF with Pillow from {image_path}: {e}")
    
    return exif_data


def get_image_info(image_path: Path) -> ImageInfo:
    """
    Get comprehensive information about an image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        ImageInfo dataclass with all available information.
    """
    exif_data = extract_exif_data(image_path)
    
    # Try to get EXIF timestamp
    timestamp = None
    if '_raw_tags' in exif_data:
        timestamp = parse_exif_datetime(exif_data['_raw_tags'])
    
    # Get file modification time
    mtime = os.path.getmtime(image_path)
    
    # Determine sort method
    if timestamp:
        sort_method = "exif_datetime"
    else:
        sort_method = "mtime"
    
    return ImageInfo(
        path=image_path,
        timestamp=timestamp,
        mtime=mtime,
        exif_data=exif_data,
        sort_method=sort_method,
    )


def sort_images_robustly(image_paths: List[Path]) -> Tuple[List[ImageInfo], str]:
    """
    Sort images robustly using EXIF timestamps, mtime, or natural filename sort.
    
    Strategy:
    1. Try to use EXIF DateTimeOriginal (with SubSecTimeOriginal if available)
    2. If EXIF missing for some images, fall back to mtime
    3. If mtime is ambiguous (all same), use natural sort on filename
    
    Args:
        image_paths: List of image file paths.
        
    Returns:
        Tuple of (sorted ImageInfo list, ordering method used).
    """
    # Get info for all images
    image_infos = [get_image_info(p) for p in image_paths]
    
    # Check how many have EXIF timestamps
    with_exif = sum(1 for info in image_infos if info.timestamp is not None)
    
    if with_exif == len(image_infos):
        # All images have EXIF timestamps - use them
        sorted_infos = sorted(image_infos, key=lambda x: (x.timestamp, str(x.path)))
        method = "exif_datetime"
        logger.info(f"Sorted {len(image_infos)} images by EXIF DateTimeOriginal")
    elif with_exif > 0:
        # Some have EXIF, some don't - use combined sort key
        sorted_infos = sorted(image_infos, key=lambda x: x.sort_key)
        method = "mixed_exif_mtime"
        logger.warning(f"Mixed sorting: {with_exif}/{len(image_infos)} images have EXIF timestamps")
    else:
        # No EXIF timestamps - check if mtime varies
        mtimes = [info.mtime for info in image_infos]
        mtime_range = max(mtimes) - min(mtimes) if mtimes else 0
        
        if mtime_range > 1.0:  # More than 1 second difference
            sorted_infos = sorted(image_infos, key=lambda x: (x.mtime, str(x.path)))
            method = "mtime"
            logger.info(f"Sorted {len(image_infos)} images by modification time")
        else:
            # mtime is too similar, use natural sort on filename
            sorted_paths = natsorted(image_paths, key=lambda p: p.name)
            path_to_info = {info.path: info for info in image_infos}
            sorted_infos = [path_to_info[p] for p in sorted_paths]
            for info in sorted_infos:
                info.sort_method = "filename"
            method = "filename_natural"
            logger.info(f"Sorted {len(image_infos)} images by natural filename order")
    
    
    return sorted_infos, method


def load_image(image_path: Path, max_width: Optional[int] = None) -> np.ndarray:
    """
    Load an image using OpenCV.
    
    Args:
        image_path: Path to the image file.
        max_width: If set, downscale image to this width (maintaining aspect ratio).
        
    Returns:
        Image as numpy array (BGR format).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    if max_width and img.shape[1] > max_width:
        scale = max_width / img.shape[1]
        new_height = int(img.shape[0] * scale)
        img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
    
    return img


def load_images(image_infos: List[ImageInfo], max_width: Optional[int] = None) -> List[np.ndarray]:
    """
    Load all images.
    
    Args:
        image_infos: List of ImageInfo objects.
        max_width: Optional maximum width for downscaling.
        
    Returns:
        List of images as numpy arrays.
    """
    images = []
    for i, info in enumerate(image_infos):
        # Progress update: every image for small batches, every 10 for large batches
        if len(image_infos) > 50:
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Loading image {i+1}/{len(image_infos)}: {info.path.name}...")
        else:
            logger.info(f"Loading image {i+1}/{len(image_infos)}: {info.path.name}...")
        
        img = load_image(info.path, max_width)
        images.append(img)
        logger.debug(f"Loaded {info.path.name}: {img.shape[1]}x{img.shape[0]}")
    
    logger.info(f"Loaded {len(images)} images" + (f" (scaled to max width {max_width})" if max_width else ""))
    return images


def get_image_dimensions(image_path: Path) -> Tuple[int, int]:
    """
    Get image dimensions without loading the full image.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        Tuple of (width, height).
    """
    with Image.open(image_path) as img:
        return img.size


def save_image(image: np.ndarray, output_path: Path, quality: int = 95) -> None:
    """
    Save an image to disk.
    
    Args:
        image: Image as numpy array (BGR format).
        output_path: Output file path.
        quality: JPEG quality (1-100).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif output_path.suffix.lower() == '.png':
        cv2.imwrite(str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    else:
        cv2.imwrite(str(output_path), image)
    
    logger.info(f"Saved image to {output_path}")

