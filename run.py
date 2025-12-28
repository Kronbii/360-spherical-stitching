#!/usr/bin/env python3
"""
360° Spherical Panorama Stitching

Creates equirectangular panoramas from a sequence of phone photos
captured on a tripod with pure rotation.

Usage:
    python run.py config.yaml
"""

import logging
import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import load_config_from_yaml
from src.pipeline import PanoramaStitchingError, run_pipeline
from src.video_utils import extract_frames_from_video, get_video_info


def setup_logging(debug: bool = False, verbose: bool = True) -> None:
    """Configure logging with appropriate format and level."""
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
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


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python run.py <config.yaml>", file=sys.stderr)
        print("\nExample:")
        print("  python run.py config.yaml", file=sys.stderr)
        return 1
    
    config_path = Path(sys.argv[1])
    
    # Load configuration from YAML file
    try:
        config = load_config_from_yaml(config_path)
        setup_logging(config.debug.enabled, config.debug.verbose_logging)
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded configuration from: {config_path}")
    except FileNotFoundError as e:
        print(f"Error: Config file not found: {config_path}", file=sys.stderr)
        return 1
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to load config file: {e}")
        return 1
    
    # Handle video input if specified in config
    video_path = getattr(config, '_video_path', None)
    
    if video_path:
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return 1
        
        logger.info(f"Processing video: {video_path}")
        
        # Get video info
        try:
            info = get_video_info(video_path)
        except Exception as e:
            logger.error(f"Could not read video: {e}")
            return 1
        
        # Create frames directory inside output (clear old frames if they exist)
        frames_dir = config.output_dir / "frames"
        if frames_dir.exists():
            logger.info(f"Clearing old frames from {frames_dir}...")
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting frames using '{config.video_extraction.method}' method...")
        
        try:
            extracted = extract_frames_from_video(
                video_path=video_path,
                output_dir=frames_dir,
                method=config.video_extraction.method,
                num_frames=config.video_extraction.num_frames,
                frame_interval=config.video_extraction.frame_interval,
                target_fps=config.video_extraction.extract_fps,
                min_frames=config.video_extraction.min_frames,
                max_frames=config.video_extraction.max_frames,
                motion_threshold=config.video_extraction.motion_threshold,
            )
            logger.info(f"Extracted {len(extracted)} frames")
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return 1
        
        config.input_dir = frames_dir
    else:
        # Validate input directory
        if not config.input_dir.exists():
            logger.error(f"Input directory does not exist: {config.input_dir}")
            return 1
    
    try:
        # Run pipeline
        panorama_path = run_pipeline(config)
        
        # Final success message
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
