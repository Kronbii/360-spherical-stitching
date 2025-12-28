#!/usr/bin/env python3
"""
360° Spherical Panorama Stitching

Creates equirectangular panoramas from a sequence of phone photos
captured on a tripod with pure rotation.

Usage:
    python run.py --input_dir ./photos --output_dir ./output
    python run.py --video ./video.mov --output_dir ./output
    python run.py --video ./video.mov --output_dir ./output --num_frames 50
    python run.py --input_dir ./photos --output_dir ./output --pano_width 8192 --debug
"""

import argparse
import logging
import sys
import tempfile
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
from src.video_utils import extract_frames_from_video, get_video_info


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
        description='Create 360° spherical panoramas from phone photos or video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  From images:
    python run.py --input_dir ./photos --output_dir ./output

  From video (recommended for phone recordings):
    python run.py --video ./video.mov --output_dir ./output
    python run.py --video ./video.mov --output_dir ./output --num_frames 50
    python run.py --video ./video.mov --output_dir ./output --extract_method motion

  High resolution with debug output:
    python run.py --input_dir ./photos --output_dir ./output --pano_width 8192 --debug

  Fast blending mode:
    python run.py --input_dir ./photos --output_dir ./output --blend feather

Video extraction methods:
  uniform  - Extract N frames uniformly distributed (default)
  interval - Extract every Nth frame
  fps      - Extract at target frames per second
  motion   - Smart extraction based on camera motion

Tips for capturing good panoramas:
  - Use a tripod or stable surface
  - Rotate camera SLOWLY and steadily
  - Ensure 30-50% overlap between shots
  - Lock exposure and white balance
  - Avoid moving objects in the scene
        '''
    )
    
    # Input source (either --input_dir OR --video)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir', '-i',
        type=Path,
        help='Directory containing input images'
    )
    input_group.add_argument(
        '--video', '-v',
        type=Path,
        help='Input video file (.mov, .mp4, etc.)'
    )
    
    # Output directory
    parser.add_argument(
        '--output_dir', '-o',
        type=Path,
        required=True,
        help='Directory for output panorama and viewer'
    )
    
    # Video extraction settings
    parser.add_argument(
        '--num_frames',
        type=int,
        default=40,
        help='Number of frames to extract from video (for uniform method). Default: 40'
    )
    parser.add_argument(
        '--extract_method',
        choices=['uniform', 'interval', 'fps', 'motion'],
        default='uniform',
        help='Video frame extraction method. Default: uniform'
    )
    parser.add_argument(
        '--extract_fps',
        type=float,
        default=2.0,
        help='Target FPS for fps extraction method. Default: 2.0'
    )
    parser.add_argument(
        '--frame_interval',
        type=int,
        default=15,
        help='Frame interval for interval extraction method. Default: 15'
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
        help='Width for downscaled images during feature matching. Default: 1600. Use --match_full_res to disable downscaling'
    )
    parser.add_argument(
        '--match_full_res',
        action='store_true',
        help='Use full resolution images for ORB feature matching (no downscaling). Slower but potentially more accurate'
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
        choices=['multiband', 'feather', 'none'],
        default='multiband',
        help='Blending method. Default: multiband. "none" = no blending (sharpest, may show seams)'
    )
    parser.add_argument(
        '--blend_levels',
        type=int,
        default=5,
        help='Number of pyramid levels for multiband blending. Default: 5'
    )
    parser.add_argument(
        '--multiband_sigma',
        type=float,
        default=30.0,
        help='Gaussian blur sigma for multiband weight computation (lower = sharper). Default: 30.0'
    )
    parser.add_argument(
        '--feather_sigma',
        type=float,
        default=50.0,
        help='Gaussian blur sigma for feather blending (lower = sharper). Default: 50.0'
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
    
    # Handle video input - extract frames first
    input_dir = args.input_dir
    frames_dir = None  # Track if we created a frames directory
    
    if args.video:
        if not args.video.exists():
            logger.error(f"Video file does not exist: {args.video}")
            return 1
        
        logger.info(f"Processing video: {args.video}")
        
        # Get video info
        try:
            info = get_video_info(args.video)
        except Exception as e:
            logger.error(f"Could not read video: {e}")
            return 1
        
        # Create frames directory inside output
        frames_dir = args.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting frames using '{args.extract_method}' method...")
        
        try:
            extracted = extract_frames_from_video(
                video_path=args.video,
                output_dir=frames_dir,
                method=args.extract_method,
                num_frames=args.num_frames,
                frame_interval=args.frame_interval,
                target_fps=args.extract_fps,
            )
            logger.info(f"Extracted {len(extracted)} frames")
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return 1
        
        input_dir = frames_dir
    else:
        # Validate input directory
        if not args.input_dir.exists():
            logger.error(f"Input directory does not exist: {args.input_dir}")
            return 1
    
    # Build configuration
    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=args.output_dir,
        matching=MatchingConfig(
            match_width=None if args.match_full_res else args.match_width,
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
            multiband_sigma=args.multiband_sigma,
            feather_sigma=args.feather_sigma,
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

