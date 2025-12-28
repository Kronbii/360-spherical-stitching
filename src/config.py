"""
Configuration module with dataclasses for pipeline settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class MatchingConfig:
    """Configuration for ORB feature matching."""
    match_width: int = 1600  # Width to downscale images for matching
    orb_nfeatures: int = 3000  # Number of ORB features to detect
    ratio_test_threshold: float = 0.75  # Lowe's ratio test threshold
    ransac_reproj_threshold: float = 3.0  # RANSAC reprojection threshold
    min_inliers: int = 60  # Minimum number of inliers required
    use_clahe: bool = False  # Use CLAHE for feature extraction


@dataclass
class IntrinsicsConfig:
    """Configuration for camera intrinsics."""
    hfov_deg: float = 65.0  # Default horizontal FOV in degrees (fallback)
    calib_json: Optional[Path] = None  # Path to calibration JSON file


@dataclass
class BlendingConfig:
    """Configuration for image blending."""
    method: str = "multiband"  # 'multiband' or 'feather'
    multiband_levels: int = 5  # Number of pyramid levels for multiband
    feather_sigma: float = 50.0  # Gaussian blur sigma for feather blending


@dataclass
class OutputConfig:
    """Configuration for output panorama."""
    pano_width: int = 4096  # Output panorama width
    output_format: str = "jpg"  # Output format: 'jpg' or 'png'
    jpg_quality: int = 95  # JPEG quality (1-100)
    
    @property
    def pano_height(self) -> int:
        """Equirectangular panorama height is half the width."""
        return self.pano_width // 2


@dataclass
class DebugConfig:
    """Configuration for debug outputs."""
    enabled: bool = False
    save_matches: bool = True  # Save match visualizations
    save_warped_frames: int = 3  # Number of warped frames to save (0 to disable)
    save_masks: bool = True  # Save blending masks
    verbose_logging: bool = True  # Enable verbose logging


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
    
    def __post_init__(self):
        """Convert paths to Path objects if needed."""
        if isinstance(self.input_dir, str):
            self.input_dir = Path(self.input_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if self.intrinsics.calib_json and isinstance(self.intrinsics.calib_json, str):
            self.intrinsics.calib_json = Path(self.intrinsics.calib_json)
    
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
            },
            "intrinsics": {
                "hfov_deg": self.intrinsics.hfov_deg,
                "calib_json": str(self.intrinsics.calib_json) if self.intrinsics.calib_json else None,
            },
            "blending": {
                "method": self.blending.method,
                "multiband_levels": self.blending.multiband_levels,
                "feather_sigma": self.blending.feather_sigma,
            },
            "output": {
                "pano_width": self.output.pano_width,
                "pano_height": self.output.pano_height,
                "output_format": self.output.output_format,
                "jpg_quality": self.output.jpg_quality,
            },
            "debug": {
                "enabled": self.debug.enabled,
            },
        }


@dataclass
class CalibrationData:
    """Camera calibration data loaded from JSON or estimated."""
    fx: float  # Focal length in pixels (x)
    fy: float  # Focal length in pixels (y)
    cx: float  # Principal point x
    cy: float  # Principal point y
    dist_coeffs: Optional[list] = None  # Distortion coefficients [k1, k2, p1, p2, k3]
    source: str = "unknown"  # How calibration was determined
    
    @property
    def K(self):
        """Return camera matrix as numpy array."""
        import numpy as np
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
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
        """Load calibration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            fx=data["fx"],
            fy=data["fy"],
            cx=data["cx"],
            cy=data["cy"],
            dist_coeffs=data.get("dist_coeffs"),
            source=f"calibration file: {path.name}"
        )

