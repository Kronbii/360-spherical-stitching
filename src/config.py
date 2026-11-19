"""
Configuration module with dataclasses for pipeline settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import yaml


@dataclass
class MatchingConfig:
    """Configuration for ORB feature matching."""
    match_width: Optional[int] = 1600  # Width to downscale images for matching (None = use full resolution)
    orb_nfeatures: int = 3000  # Number of ORB features to detect
    ratio_test_threshold: float = 0.75  # Lowe's ratio test threshold (lower = stricter, 0.75 = balanced)
    ransac_reproj_threshold: float = 3.0  # RANSAC reprojection threshold (scaled by image size)
    min_inliers: int = 60  # Minimum number of inliers required (lower = more lenient)
    use_clahe: bool = False  # Use CLAHE for feature extraction
    disable_circular_closure: bool = False  # Disable circular closure detection (use if video doesn't loop)
    symmetric_matching: bool = False  # Use symmetric matching (cross-check) - can be too strict for some datasets
    ransac_refinement: bool = True  # Refine homography with inliers after initial RANSAC
    ransac_max_iters: int = 3000  # Maximum RANSAC iterations (higher = better but slower)
    rotation_smoothing_window: int = 3  # Temporal smoothing window size for rotations (odd, 3-15 recommended, larger = straighter lines but may oversmooth)


@dataclass
class IntrinsicsConfig:
    """Configuration for camera intrinsics."""
    hfov_deg: float = 65.0  # Default horizontal FOV in degrees (fallback)
    calib_json: Optional[Path] = None  # Path to calibration JSON file


@dataclass
class BlendingConfig:
    """Configuration for image blending."""
    method: str = "multiband"  # 'multiband', 'feather', 'sharp', or 'none'
    multiband_levels: int = 5  # Number of pyramid levels for multiband
    multiband_sigma: float = 30.0  # Gaussian blur sigma for multiband weight computation
    feather_sigma: float = 50.0  # Gaussian blur sigma for feather blending
    sharp_blend_width: float = 2.0  # Blend zone width in pixels for sharp blending (1-3 recommended)


@dataclass
class OutputConfig:
    """Configuration for output panorama."""
    pano_width: int = 4096  # Output panorama width
    output_format: str = "jpg"  # Output format: 'jpg' or 'png'
    jpg_quality: int = 95  # JPEG quality (1-100)
    warp_workers: int = 0  # Threads for warping (0 = auto, 1 = serial). Each holds one warped frame.
    
    @property
    def pano_height(self) -> int:
        """Equirectangular panorama height is half the width."""
        return self.pano_width // 2


@dataclass
class DebugConfig:
    """Configuration for debug outputs."""
    enabled: bool = False  # Enable debug mode
    save_matches: bool = True  # Save feature match visualizations
    save_warped_frames: int = 0  # Number of warped frames to save (0 = none)
    save_seams: bool = True  # Save blending masks
    verbose_logging: bool = True  # Enable verbose logging


@dataclass
class OrientationConfig:
    """How frame orientations are estimated."""
    method: str = "bundle"  # 'bundle' (loop closures, rotation averaging, bundle adjustment) or 'chain' (v1)
    optimize_focal: bool = True  # bundle: solve one shared focal length along with the rotations
    ransac_threshold_px: float = 2.0  # bundle: rotation-only RANSAC threshold at match resolution
    min_inliers: int = 30  # bundle: inliers a pair needs to be used
    max_pair_gap: int = 3  # bundle: always match (i, i+1) .. (i, i+gap)
    long_pairs_per_frame: int = 1  # bundle: extra pairs ~half a field of view apart, per frame
    loop_pairs_per_frame: int = 2  # bundle: loop-closure pairs per frame
    chain_drift_deg_per_frame: float = 0.1  # bundle without gyro: assumed chain drift, widens search
    gyro_drift_deg_per_s: float = 0.5  # bundle with gyro: assumed gyro drift, widens search


@dataclass
class ImuConfig:
    """Optional gyroscope log, used as the prior for method 'bundle'."""
    gyro_csv: Optional[Path] = None  # CSV with a time column and three angular-rate columns
    gyro_units: str = "rad/s"  # 'rad/s' or 'deg/s'
    frame_times_csv: Optional[Path] = None  # file,time_s per frame; written automatically for video input
    time_offset_s: Optional[float] = None  # gyro_time = video_time + offset; None = estimate from the video
    axes: Optional[list] = None  # gyro axis per camera axis, e.g. [x, -y, -z]; None = estimate


@dataclass
class VideoExtractionConfig:
    """Configuration for video frame extraction."""
    method: str = "uniform"  # 'uniform', 'interval', 'fps', or 'motion'
    num_frames: int = 40  # Number of frames for uniform method
    frame_interval: int = 15  # Frame interval for interval method
    extract_fps: float = 2.0  # Target FPS for fps method
    min_frames: int = 20  # Minimum frames for motion method
    max_frames: int = 100  # Maximum frames for motion method
    motion_threshold: float = 0.02  # Motion threshold for motion method (0-1)


@dataclass
class PipelineConfig:
    """Main configuration for the entire pipeline."""
    input_dir: Path
    output_dir: Path
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    intrinsics: IntrinsicsConfig = field(default_factory=IntrinsicsConfig)
    blending: BlendingConfig = field(default_factory=BlendingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    video_extraction: VideoExtractionConfig = field(default_factory=VideoExtractionConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    
    def __post_init__(self):
        """Convert paths to Path objects if needed."""
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.intrinsics.calib_json and isinstance(self.intrinsics.calib_json, str):
            self.intrinsics.calib_json = Path(self.intrinsics.calib_json)
        for name in ("gyro_csv", "frame_times_csv"):
            value = getattr(self.imu, name)
            if value and isinstance(value, str):
                setattr(self.imu, name, Path(value))
        if self.orientation.method not in ("chain", "bundle"):
            raise ValueError(f"orientation.method must be 'chain' or 'bundle', got {self.orientation.method!r}")
        if self.imu.gyro_csv and self.orientation.method != "bundle":
            raise ValueError("A gyro log is only used by orientation.method 'bundle'")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging/saving."""
        return {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "matching": {
                "match_width": self.matching.match_width,
                "orb_nfeatures": self.matching.orb_nfeatures,
                "ratio_test_threshold": self.matching.ratio_test_threshold,
                "ransac_reproj_threshold": self.matching.ransac_reproj_threshold,
                "min_inliers": self.matching.min_inliers,
                "use_clahe": self.matching.use_clahe,
                "disable_circular_closure": self.matching.disable_circular_closure,
                "symmetric_matching": self.matching.symmetric_matching,
                "ransac_refinement": self.matching.ransac_refinement,
                "ransac_max_iters": self.matching.ransac_max_iters,
                "rotation_smoothing_window": self.matching.rotation_smoothing_window,
            },
            "intrinsics": {
                "hfov_deg": self.intrinsics.hfov_deg,
                "calib_json": str(self.intrinsics.calib_json) if self.intrinsics.calib_json else None,
            },
            "blending": {
                "method": self.blending.method,
                "multiband_levels": self.blending.multiband_levels,
                "multiband_sigma": self.blending.multiband_sigma,
                "feather_sigma": self.blending.feather_sigma,
                "sharp_blend_width": self.blending.sharp_blend_width,
            },
            "output": {
                "pano_width": self.output.pano_width,
                "pano_height": self.output.pano_height,
                "output_format": self.output.output_format,
                "jpg_quality": self.output.jpg_quality,
                "warp_workers": self.output.warp_workers,
            },
            "debug": {
                "enabled": self.debug.enabled,
            },
            "video_extraction": {
                "method": self.video_extraction.method,
                "num_frames": self.video_extraction.num_frames,
                "frame_interval": self.video_extraction.frame_interval,
                "extract_fps": self.video_extraction.extract_fps,
                "min_frames": self.video_extraction.min_frames,
                "max_frames": self.video_extraction.max_frames,
                "motion_threshold": self.video_extraction.motion_threshold,
            },
            "orientation": {
                "method": self.orientation.method,
                "optimize_focal": self.orientation.optimize_focal,
                "ransac_threshold_px": self.orientation.ransac_threshold_px,
                "min_inliers": self.orientation.min_inliers,
                "max_pair_gap": self.orientation.max_pair_gap,
                "long_pairs_per_frame": self.orientation.long_pairs_per_frame,
                "loop_pairs_per_frame": self.orientation.loop_pairs_per_frame,
                "chain_drift_deg_per_frame": self.orientation.chain_drift_deg_per_frame,
                "gyro_drift_deg_per_s": self.orientation.gyro_drift_deg_per_s,
            },
            "imu": {
                "gyro_csv": str(self.imu.gyro_csv) if self.imu.gyro_csv else None,
                "gyro_units": self.imu.gyro_units,
                "frame_times_csv": str(self.imu.frame_times_csv) if self.imu.frame_times_csv else None,
                "time_offset_s": self.imu.time_offset_s,
                "axes": self.imu.axes,
            },
        }


def load_config_from_yaml(config_path: Path) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        PipelineConfig object.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config file is invalid.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML dictionary, got {type(data)}")
    
    # Extract top-level paths
    input_dir_str = data.get('input_dir')
    video_path = data.get('video')
    output_dir = Path(data.get('output_dir', ''))
    
    if not output_dir or str(output_dir) == '.':
        raise ValueError("'output_dir' is required in config file")
    
    # Check if input_dir was actually specified (not just empty string)
    input_dir = Path(input_dir_str) if input_dir_str else None
    
    if not input_dir and not video_path:
        raise ValueError("Either 'input_dir' or 'video' must be specified in config file")
    
    if input_dir and video_path:
        raise ValueError("Cannot specify both 'input_dir' and 'video' in config file")
    
    # If video is specified, input_dir will be set later (after frame extraction)
    # For now, we'll use a placeholder that gets replaced
    if video_path:
        input_dir = Path(video_path)  # Temporary, will be replaced with frames dir
    else:
        # Ensure input_dir is a Path object
        input_dir = Path(input_dir)
    
    # Extract nested configs
    matching_data = data.get('matching', {})
    intrinsics_data = data.get('intrinsics', {})
    blending_data = data.get('blending', {})
    output_data = data.get('output', {})
    debug_data = data.get('debug', {})
    video_extraction_data = data.get('video_extraction', {})
    orientation_data = data.get('orientation', {}) or {}
    imu_data = data.get('imu', {}) or {}
    
    # Build config objects
    matching = MatchingConfig(
        match_width=None if matching_data.get('match_full_res', False) else matching_data.get('match_width', 1600),
        orb_nfeatures=matching_data.get('orb_nfeatures', 3000),
        ratio_test_threshold=matching_data.get('ratio_test_threshold', 0.75),
        ransac_reproj_threshold=matching_data.get('ransac_reproj_threshold', 3.0),
        min_inliers=matching_data.get('min_inliers', 60),
        use_clahe=matching_data.get('use_clahe', False),
        disable_circular_closure=matching_data.get('disable_circular_closure', False),
        symmetric_matching=matching_data.get('symmetric_matching', False),
        ransac_refinement=matching_data.get('ransac_refinement', True),
        ransac_max_iters=matching_data.get('ransac_max_iters', 3000),
        rotation_smoothing_window=matching_data.get('rotation_smoothing_window', 3),
    )
    
    calib_json_str = intrinsics_data.get('calib_json')
    intrinsics = IntrinsicsConfig(
        hfov_deg=intrinsics_data.get('hfov_deg', 65.0),
        calib_json=Path(calib_json_str) if calib_json_str else None,
    )
    
    blending = BlendingConfig(
        method=blending_data.get('method', 'multiband'),
        multiband_levels=blending_data.get('multiband_levels', 5),
        multiband_sigma=blending_data.get('multiband_sigma', 30.0),
        feather_sigma=blending_data.get('feather_sigma', 50.0),
        sharp_blend_width=blending_data.get('sharp_blend_width', 2.0),
    )
    
    output = OutputConfig(
        pano_width=output_data.get('pano_width', 4096),
        output_format=output_data.get('output_format', 'jpg'),
        jpg_quality=output_data.get('jpg_quality', 95),
        warp_workers=output_data.get('warp_workers', 0),
    )
    
    debug = DebugConfig(
        enabled=debug_data.get('enabled', False),
        save_matches=debug_data.get('save_matches', True),
        save_warped_frames=debug_data.get('save_warped_frames', 0),
        save_seams=debug_data.get('save_seams', True),
        verbose_logging=debug_data.get('verbose_logging', True),
    )
    
    video_extraction = VideoExtractionConfig(
        method=video_extraction_data.get('method', 'uniform'),
        num_frames=video_extraction_data.get('num_frames', 40),
        frame_interval=video_extraction_data.get('frame_interval', 15),
        extract_fps=video_extraction_data.get('extract_fps', 2.0),
        min_frames=video_extraction_data.get('min_frames', 20),
        max_frames=video_extraction_data.get('max_frames', 100),
        motion_threshold=video_extraction_data.get('motion_threshold', 0.02),
    )
    
    defaults = OrientationConfig()
    orientation = OrientationConfig(**{
        name: orientation_data.get(name, getattr(defaults, name))
        for name in OrientationConfig.__dataclass_fields__
    })
    unknown = set(orientation_data) - set(OrientationConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown orientation option(s): {', '.join(sorted(unknown))}")

    offset = imu_data.get('time_offset_s')
    if isinstance(offset, str) and offset.strip().lower() == 'auto':
        offset = None
    axes = imu_data.get('axes')
    if isinstance(axes, str):
        axes = None if axes.strip().lower() == 'auto' else [a.strip() for a in axes.split(',')]
    imu = ImuConfig(
        gyro_csv=Path(imu_data['gyro_csv']) if imu_data.get('gyro_csv') else None,
        gyro_units=imu_data.get('gyro_units', 'rad/s'),
        frame_times_csv=Path(imu_data['frame_times_csv']) if imu_data.get('frame_times_csv') else None,
        time_offset_s=float(offset) if offset is not None else None,
        axes=list(axes) if axes else None,
    )

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        matching=matching,
        intrinsics=intrinsics,
        blending=blending,
        output=output,
        debug=debug,
        video_extraction=video_extraction,
        orientation=orientation,
        imu=imu,
    )
    
    # Store video path if specified (for later use)
    if video_path:
        config._video_path = Path(video_path)
    else:
        config._video_path = None
    
    return config


@dataclass
class CalibrationData:
    """Camera calibration data loaded from JSON or estimated."""
    fx: float  # Focal length in pixels (x)
    fy: float  # Focal length in pixels (y)
    cx: float  # Principal point (x)
    cy: float  # Principal point (y)
    dist_coeffs: Optional[list] = None  # Distortion coefficients [k1, k2, p1, p2, k3]
    source: str = "unspecified"  # Human-readable provenance for reports and diagnostics
    
    @property
    def K(self) -> "np.ndarray":  # type: ignore
        """Camera intrinsic matrix."""
        import numpy as np
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

    def to_dict(self) -> dict:
        """Return a JSON-serializable calibration record."""
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "dist_coeffs": self.dist_coeffs,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationData":
        """Load calibration data from a flat record or a saved intrinsics report."""
        path = Path(path)
        with path.open("r") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Calibration file must contain a JSON object: {path}")

        # ``save_intrinsics_report`` wraps the values in an ``intrinsics`` object,
        # while user-authored calibration files commonly use the flat form.
        if "intrinsics" in data:
            values = data["intrinsics"]
            if not isinstance(values, dict):
                raise ValueError(f"'intrinsics' must be a JSON object: {path}")
            source = data.get("estimation_source", values.get("source"))
        else:
            values = data
            source = values.get("source")

        try:
            return cls(
                fx=float(values["fx"]),
                fy=float(values["fy"]),
                cx=float(values["cx"]),
                cy=float(values["cy"]),
                dist_coeffs=values.get("dist_coeffs"),
                source=str(source) if source is not None else f"calibration JSON: {path}",
            )
        except KeyError as exc:
            raise ValueError(f"Calibration file is missing required field {exc.args[0]!r}: {path}") from exc
