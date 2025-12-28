#!/usr/bin/env python3
"""
360° Spherical Panorama Stitching

Creates equirectangular panoramas from a sequence of phone photos
captured on a tripod with pure rotation.

Usage:
    python run.py --input_dir ./photos --output_dir ./output
    python run.py --input_dir ./photos --output_dir ./output --pano_width 8192 --debug
    python run.py --input_dir ./photos --output_dir ./output --blend feather --clahe
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    BlendingConfig,
    DebugConfig,
    IntrinsicsConfig,
    MatchingConfig,
    OutputConfig,
    PipelineConfig,
)
from src.pipeline import PanoramaStitchingError, run_pipeline


def setup_logging(debug: bool = False) -> None:
    """Configure logging with appropriate format and level."""
    level = logging.DEBUG if debug else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(console_handler)
    
    # Suppress verbose logging from external libraries
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('exifread').setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Create 360° spherical panoramas from phone photos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Basic usage:
    python run.py --input_dir ./photos --output_dir ./output

  High resolution with debug output:
    python run.py --input_dir ./photos --output_dir ./output --pano_width 8192 --debug

  Fast blending mode:
    python run.py --input_dir ./photos --output_dir ./output --blend feather

  With calibration file:
    python run.py --input_dir ./photos --output_dir ./output --calib_json ./calib.json

Tips for capturing good panoramas:
  - Use a tripod or stable surface
  - Rotate camera around its entrance pupil
  - Ensure 30-50% overlap between shots
  - Lock exposure and white balance
  - Avoid moving objects in the scene
        '''
    )
    
    # Required arguments
    parser.add_argument(
        '--input_dir', '-i',
        type=Path,
        required=True,
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        required=True,
        help='Directory for output panorama and viewer'
    )
    
    # Output settings
    parser.add_argument(
        '--pano_width',
        type=int,
        default=4096,
        help='Output panorama width in pixels (height = width/2). Default: 4096'
    )
    parser.add_argument(
        '--output_format',
        choices=['jpg', 'png'],
        default='jpg',
        help='Output image format. Default: jpg'
    )
    
    # Matching settings
    parser.add_argument(
        '--match_width',
        type=int,
        default=1600,
        help='Width for downscaled images during feature matching. Default: 1600'
    )
    parser.add_argument(
        '--min_inliers',
        type=int,
        default=60,
        help='Minimum RANSAC inliers required for a valid match. Default: 60'
    )
    parser.add_argument(
        '--clahe',
        action='store_true',
        help='Apply CLAHE for feature extraction (helps with low contrast images)'
    )
    
    # Intrinsics settings
    parser.add_argument(
        '--hfov_deg',
        type=float,
        default=65.0,
        help='Horizontal field of view in degrees (used if EXIF unavailable). Default: 65'
    )
    parser.add_argument(
        '--calib_json',
        type=Path,
        default=None,
        help='Path to camera calibration JSON file (optional)'
    )
    
    # Blending settings
    parser.add_argument(
        '--blend',
        choices=['multiband', 'feather'],
        default='multiband',
        help='Blending method. Default: multiband'
    )
    parser.add_argument(
        '--blend_levels',
        type=int,
        default=5,
        help='Number of pyramid levels for multiband blending. Default: 5'
    )
    
    # Debug settings
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (saves intermediate visualizations)'
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    setup_logging(args.debug)
    logger = logging.getLogger(__name__)
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    # Build configuration
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        matching=MatchingConfig(
            match_width=args.match_width,
            min_inliers=args.min_inliers,
            use_clahe=args.clahe,
        ),
        intrinsics=IntrinsicsConfig(
            hfov_deg=args.hfov_deg,
            calib_json=args.calib_json,
        ),
        blending=BlendingConfig(
            method=args.blend,
            multiband_levels=args.blend_levels,
        ),
        output=OutputConfig(
            pano_width=args.pano_width,
            output_format=args.output_format,
        ),
        debug=DebugConfig(
            enabled=args.debug,
        ),
    )
    
    try:
        # Run pipeline
        panorama_path = run_pipeline(config)
        
        # Final success message (not using logging)
        print(f"\n✓ Panorama created successfully: {panorama_path}")
        print(f"✓ Open viewer at: file://{config.output_dir.absolute() / 'viewer' / 'index.html'}")
        
        return 0
        
    except PanoramaStitchingError as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
        
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

