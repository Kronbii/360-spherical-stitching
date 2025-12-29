"""
Video frame extraction utilities for panorama stitching.

Extracts frames from video files with various strategies:
- Uniform time-based extraction
- Frame interval extraction
- Motion-based keyframe detection
"""

import logging
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    return shutil.which('ffmpeg') is not None


def get_video_info(video_path: Path) -> dict:
    """
    Get video information using OpenCV.
    
    Args:
        video_path: Path to video file.
        
    Returns:
        Dictionary with video properties.
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration_sec': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
    }
    
    cap.release()
    
    logger.info(f"Video info: {info['width']}x{info['height']}, "
                f"{info['fps']:.1f} fps, {info['frame_count']} frames, "
                f"{info['duration_sec']:.1f} seconds")
    
    return info


def extract_frames_uniform(
    video_path: Path,
    output_dir: Path,
    num_frames: int = 30,
    quality: int = 95
) -> List[Path]:
    """
    Extract frames uniformly distributed across the video duration.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save extracted frames.
        num_frames: Number of frames to extract.
        quality: JPEG quality (1-100).
        
    Returns:
        List of paths to extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames < num_frames:
        num_frames = total_frames
        logger.warning(f"Video has only {total_frames} frames, extracting all")
    
    # Calculate frame indices to extract (uniformly distributed)
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    logger.info(f"Extracting {num_frames} frames uniformly from {total_frames} total frames...")
    
    extracted_paths = []
    for i, frame_idx in enumerate(frame_indices):
        # Progress update: every frame for small batches, every 10 for large batches
        if num_frames > 50:
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Extracting frame {i+1}/{num_frames}...")
        else:
            logger.info(f"Extracting frame {i+1}/{num_frames}...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            logger.warning(f"Could not read frame {frame_idx}")
            continue
        
        # Save frame
        frame_path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        extracted_paths.append(frame_path)
        
        time_sec = frame_idx / fps
        logger.debug(f"Extracted frame {i+1}/{num_frames} (frame {frame_idx}, t={time_sec:.2f}s)")
    
    cap.release()
    
    logger.info(f"Extracted {len(extracted_paths)} frames to {output_dir}")
    return extracted_paths


def extract_frames_interval(
    video_path: Path,
    output_dir: Path,
    frame_interval: int = 15,
    quality: int = 95
) -> List[Path]:
    """
    Extract every Nth frame from the video.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save extracted frames.
        frame_interval: Extract every Nth frame.
        quality: JPEG quality (1-100).
        
    Returns:
        List of paths to extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    expected_frames = total_frames // frame_interval
    logger.info(f"Extracting every {frame_interval}th frame (~{expected_frames} frames)...")
    
    extracted_paths = []
    frame_idx = 0
    output_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Progress update: every 10 frames for large batches
            if expected_frames > 50:
                if output_idx % 10 == 0:
                    logger.info(f"Extracting frame {output_idx+1}/{expected_frames}...")
            else:
                logger.info(f"Extracting frame {output_idx+1}/{expected_frames}...")
            
            frame_path = output_dir / f"frame_{output_idx:04d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            extracted_paths.append(frame_path)
            
            time_sec = frame_idx / fps
            logger.debug(f"Extracted frame {output_idx+1} (frame {frame_idx}, t={time_sec:.2f}s)")
            output_idx += 1
        
        frame_idx += 1
    
    cap.release()
    
    logger.info(f"Extracted {len(extracted_paths)} frames to {output_dir}")
    return extracted_paths


def extract_frames_fps(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 2.0,
    quality: int = 95
) -> List[Path]:
    """
    Extract frames at a target frames-per-second rate.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save extracted frames.
        target_fps: Target extraction rate in frames per second.
        quality: JPEG quality (1-100).
        
    Returns:
        List of paths to extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    
    # Calculate frame interval (round to nearest integer for best approximation)
    frame_interval = max(1, round(video_fps / target_fps))
    expected_frames = int(duration * target_fps)
    
    # Calculate actual extraction FPS for logging
    actual_extraction_fps = video_fps / frame_interval
    
    logger.info(f"Extracting at {target_fps} fps target (every {frame_interval} frames, actual: ~{actual_extraction_fps:.1f} fps, ~{expected_frames} total)...")
    
    cap.release()
    
    return extract_frames_interval(video_path, output_dir, frame_interval, quality)


def extract_keyframes_motion(
    video_path: Path,
    output_dir: Path,
    min_frames: int = 20,
    max_frames: int = 100,
    motion_threshold: float = 0.02,
    quality: int = 95
) -> List[Path]:
    """
    Extract keyframes based on motion detection.
    
    Selects frames where significant motion/rotation occurred.
    Good for panorama videos where camera rotation varies in speed.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save extracted frames.
        min_frames: Minimum number of frames to extract.
        max_frames: Maximum number of frames to extract.
        motion_threshold: Motion threshold for keyframe detection.
        quality: JPEG quality (1-100).
        
    Returns:
        List of paths to extracted frames.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    logger.info(f"Analyzing video for motion-based keyframes ({total_frames} frames)...")
    
    # First pass: compute motion scores
    motion_scores = []
    prev_gray = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Progress update during analysis: every 100 frames
        if frame_idx % 100 == 0 and frame_idx > 0:
            logger.info(f"Analyzing motion: frame {frame_idx}/{total_frames}...")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))  # Downscale for speed
        
        if prev_gray is not None:
            # Compute absolute difference
            diff = cv2.absdiff(gray, prev_gray)
            motion_score = np.mean(diff) / 255.0
            motion_scores.append((frame_idx, motion_score))
        
        prev_gray = gray
        frame_idx += 1
    
    cap.release()
    
    if not motion_scores:
        logger.warning("No motion detected, falling back to uniform extraction")
        return extract_frames_uniform(video_path, output_dir, min_frames, quality)
    
    # Cumulative motion to select evenly-spaced keyframes
    cumulative_motion = np.cumsum([s[1] for s in motion_scores])
    total_motion = cumulative_motion[-1]
    
    # Calculate number of frames to extract based on motion distribution
    # Use max_frames as the upper bound, but try to select frames based on motion density
    # Instead of using total_motion/motion_threshold (which can be huge), 
    # we'll select frames evenly spaced in cumulative motion space, capped by max_frames
    num_frames = min(max_frames, max(min_frames, len(motion_scores) // 10))
    motion_intervals = np.linspace(0, total_motion, num_frames)
    
    selected_indices = []
    for target_motion in motion_intervals:
        idx = np.searchsorted(cumulative_motion, target_motion)
        idx = min(idx, len(motion_scores) - 1)
        frame_idx = motion_scores[idx][0]
        if frame_idx not in selected_indices:
            selected_indices.append(frame_idx)
    
    # Re-open video and extract selected frames
    logger.info(f"Extracting {len(selected_indices)} keyframes...")
    cap = cv2.VideoCapture(str(video_path))
    extracted_paths = []
    sorted_indices = sorted(selected_indices)
    
    for i, frame_idx in enumerate(sorted_indices):
        # Progress update: every frame for small batches, every 10 for large batches
        if len(sorted_indices) > 50:
            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"Extracting keyframe {i+1}/{len(sorted_indices)}...")
        else:
            logger.info(f"Extracting keyframe {i+1}/{len(sorted_indices)}...")
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_path = output_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        extracted_paths.append(frame_path)
        
        time_sec = frame_idx / fps
        logger.debug(f"Extracted keyframe {i+1}/{len(sorted_indices)} (t={time_sec:.2f}s)")
    
    cap.release()
    
    logger.info(f"Extracted {len(extracted_paths)} motion-based keyframes to {output_dir}")
    return extracted_paths


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    method: str = "uniform",
    num_frames: int = 30,
    frame_interval: int = 15,
    target_fps: float = 2.0,
    min_frames: int = 20,
    max_frames: int = 100,
    motion_threshold: float = 0.02,
    quality: int = 95
) -> List[Path]:
    """
    Extract frames from video using specified method.
    
    Args:
        video_path: Path to video file.
        output_dir: Directory to save extracted frames.
        method: Extraction method ("uniform", "interval", "fps", "motion").
        num_frames: Number of frames for uniform method.
        frame_interval: Frame interval for interval method.
        target_fps: Target FPS for fps method.
        min_frames: Minimum frames for motion method.
        max_frames: Maximum frames for motion method.
        motion_threshold: Motion threshold for motion method (0-1).
        quality: JPEG quality (1-100).
        
    Returns:
        List of paths to extracted frames.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video info
    info = get_video_info(video_path)
    
    # Extract based on method
    if method == "uniform":
        return extract_frames_uniform(video_path, output_dir, num_frames, quality)
    elif method == "interval":
        return extract_frames_interval(video_path, output_dir, frame_interval, quality)
    elif method == "fps":
        return extract_frames_fps(video_path, output_dir, target_fps, quality)
    elif method == "motion":
        return extract_keyframes_motion(
            video_path, output_dir,
            min_frames=min_frames,
            max_frames=max_frames,
            motion_threshold=motion_threshold,
            quality=quality
        )
    else:
        raise ValueError(f"Unknown extraction method: {method}")


