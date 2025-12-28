# 360° Spherical Panorama Stitching

Create stunning 360° equirectangular panoramas from a sequence of phone photos captured on a tripod with pure rotation.

![Panorama Example](docs/example.jpg)

## Features

- **Robust image ordering**: Automatically sorts images using EXIF timestamps, file modification time, or natural filename sorting
- **EXIF-based intrinsics**: Estimates camera focal length from EXIF data (FocalLengthIn35mmFilm) with HFOV fallback
- **ORB feature matching**: Uses ORB features with BFMatcher, Lowe's ratio test, and RANSAC for robust matching
- **Pure rotation model**: Extracts rotation matrices from homographies, chains them into global rotations
- **Spherical projection**: Uses inverse mapping with cv2.remap for efficient equirectangular warping
- **Advanced blending**: Multiband (Laplacian pyramid) and feather blending options
- **Interactive viewer**: Generates an HTML 360° viewer using Three.js
- **Debug outputs**: Optional visualization of matches, warped frames, and seam regions

## Installation

### Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy, Pillow, exifread, natsort

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/360-spherical-stitching.git
cd 360-spherical-stitching

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The pipeline uses a YAML configuration file for all settings. Create a `config.yaml` file (see `config.yaml` in the repository for an example) and run:

```bash
python run.py config.yaml
```

### Configuration File

The configuration file supports all pipeline parameters. See `config.yaml` for a complete example with all options documented.

**Basic example (`config.yaml`):**
```yaml
# Input source (choose one)
input_dir: ./photos  # Directory containing input images
# OR use video:
# video: ./video.mov

# Required: Output directory
output_dir: ./output

# Output settings
output:
  pano_width: 4096  # Panorama width in pixels (height = width/2)
  output_format: jpg  # Options: jpg, png
  jpg_quality: 95  # JPEG quality (1-100)

# Feature matching settings
matching:
  match_width: 1600  # Width for downscaled images during matching (or null for full resolution)
  match_full_res: false  # If true, use full resolution (match_width is ignored)
  orb_nfeatures: 3000  # Number of ORB features to detect
  ratio_test_threshold: 0.75  # Lowe's ratio test threshold
  ransac_reproj_threshold: 3.0  # RANSAC reprojection threshold
  min_inliers: 60  # Minimum RANSAC inliers required
  use_clahe: false  # Apply CLAHE for feature extraction
  disable_circular_closure: false  # Set to true if video doesn't loop

# Camera intrinsics
intrinsics:
  hfov_deg: 65.0  # Horizontal field of view in degrees (fallback if EXIF unavailable)
  calib_json: null  # Path to calibration JSON file (optional)

# Blending settings
blending:
  method: multiband  # Options: multiband, feather, none
  multiband_levels: 5  # Number of pyramid levels for multiband blending
  multiband_sigma: 30.0  # Gaussian blur sigma for multiband (lower = sharper)
  feather_sigma: 50.0  # Gaussian blur sigma for feather blending (lower = sharper)

# Video extraction (only used if 'video' is specified)
video_extraction:
  method: uniform  # Options: uniform, interval, fps, motion
  num_frames: 40  # For 'uniform' method
  frame_interval: 15  # For 'interval' method
  extract_fps: 2.0  # For 'fps' method

# Debug settings
debug:
  enabled: false  # Enable debug mode (saves intermediate visualizations)
  save_matches: true  # Save feature match visualizations
  save_warped_frames: 0  # Number of warped frames to save (0 = none)
  save_seams: true  # Save blending masks
  verbose_logging: true  # Enable verbose logging
```

### Examples

**High resolution panorama:**
```yaml
output:
  pano_width: 8192
```

**Fast processing with feather blending:**
```yaml
matching:
  match_width: 1200
blending:
  method: feather
```

**Debug mode with CLAHE (for low contrast images):**
```yaml
matching:
  use_clahe: true
debug:
  enabled: true
```

**With custom calibration:**
```yaml
intrinsics:
  calib_json: ./camera_calib.json
```

**Processing a video:**
```yaml
video: ./video.mov
video_extraction:
  method: fps
  extract_fps: 5.0
```

## Tips for Capturing Good Panoramas

### Camera Setup

1. **Use a tripod or stable surface**: Minimizes parallax errors
2. **Rotate around the entrance pupil**: The optical center of the lens, not the camera body
3. **Lock exposure and white balance**: Prevents brightness/color variations between shots
4. **Use manual focus**: Ensures consistent focus across all images

### Shooting

1. **Overlap**: Ensure 30-50% overlap between consecutive shots
2. **Rotation**: Rotate smoothly in one direction (left-to-right or right-to-left)
3. **Coverage**: For full 360°, shoot approximately 360°/HFOV + 2 images
4. **Avoid**: Moving objects, extreme lighting changes, nearby objects (parallax)

### Image Count Guidelines

| Phone HFOV | Images for 360° |
|------------|-----------------|
| ~65° | 8-10 images |
| ~75° | 7-8 images |
| ~85° | 6-7 images |

### Typical Phone FOVs

- **iPhone 15 Pro** (main): ~75° HFOV
- **Samsung Galaxy S24**: ~80° HFOV  
- **Google Pixel 8**: ~77° HFOV
- **iPhone 15** (standard): ~75° HFOV

## Output Files

After running, you'll find:

```
output_dir/
├── panorama.jpg           # Final equirectangular panorama
├── config.json            # Pipeline configuration used
├── intrinsics.json        # Camera intrinsics report
├── viewer/
│   ├── index.html         # Interactive 360° viewer
│   └── panorama.jpg       # Copy of panorama
└── debug/                 # (if debug.enabled: true in config)
    ├── matches/           # Feature match visualizations
    ├── warped/            # Warped frame previews
    └── seams.jpg          # Seam visualization
```

## Viewing the Panorama

### Local Viewer

Open the viewer directly in your browser:

```bash
# The path is printed after running
file:///path/to/output/viewer/index.html
```

**Note**: Some browsers block local file access. If the panorama doesn't load:
- Use Firefox (usually works)
- Start a local server: `python -m http.server 8000` then open `http://localhost:8000/output/viewer/`

### Online Viewers

Upload your panorama to:
- [Pannellum](https://pannellum.org/documentation/examples/simple-example/)
- [Marzipano](https://www.marzipano.net/tool/)
- [A-Frame VR](https://aframe.io/)

## Calibration File Format

If you have camera calibration data, create a JSON file:

```json
{
    "fx": 1500.0,
    "fy": 1500.0,
    "cx": 2000.0,
    "cy": 1500.0,
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

- `fx`, `fy`: Focal length in pixels
- `cx`, `cy`: Principal point (usually image center)
- `dist_coeffs`: [k1, k2, p1, p2, k3] distortion coefficients (optional)

## Technical Details

### Pipeline Overview

1. **Load & Sort**: Images sorted by EXIF timestamp → mtime → filename
2. **Intrinsics**: K estimated from EXIF FocalLengthIn35mmFilm or HFOV fallback
3. **Feature Matching**: ORB + BFMatcher + kNN ratio test + RANSAC homography
4. **Rotation Extraction**: R = K⁻¹ @ H @ K, orthonormalized with SVD
5. **Rotation Chaining**: R_global[i+1] = R_rel[i] @ R_global[i]
6. **Spherical Warp**: Inverse mapping from equirectangular → image coordinates
7. **Blending**: Multiband (Laplacian pyramid) or feather blending
8. **Output**: Equirectangular JPG/PNG + HTML viewer

### Coordinate Systems

- **Equirectangular**: θ ∈ [-π, π] (azimuth), φ ∈ [-π/2, π/2] (elevation)
- **World direction**: r = [sin(θ)cos(φ), sin(φ), cos(θ)cos(φ)]
- **Camera transform**: r_cam = R^T @ r_world
- **Projection**: u = fx·(x/z) + cx, v = fy·(y/z) + cy

### Known Limitations

1. **Parallax**: Close objects may show ghosting if rotation center isn't at entrance pupil
2. **Moving objects**: Will appear as ghosts or artifacts
3. **360° loops**: Doesn't explicitly handle loop closure (first/last image matching)
4. **Vertical coverage**: Limited by camera FOV; typically captures horizontal band only
5. **Extreme rotations**: Very large rotations between shots may fail matching

## Troubleshooting

### "Matching failed" Error

- **Cause**: Insufficient feature matches between images
- **Solutions**:
  - Ensure 30-50% overlap between shots
  - Enable `use_clahe: true` in config for low contrast scenes
  - Reduce `min_inliers` (e.g., 40) in config for difficult scenes
  - Check image order (may be misordered)

### Visible Seams

- **Cause**: Exposure differences or insufficient blending
- **Solutions**:
  - Lock camera exposure before shooting
  - Use `method: multiband` in blending config (default)
  - Increase `multiband_levels` (e.g., 6 or 7) in config

### Panorama Not Loading in Viewer

- **Cause**: Browser security restrictions on local files
- **Solutions**:
  - Use Firefox
  - Run local server: `python -m http.server 8000`
  - Upload to web hosting

### Ghosting/Artifacts

- **Cause**: Parallax or camera translation
- **Solutions**:
  - Use tripod with proper rotation point
  - Avoid close foreground objects
  - Reshoot with careful rotation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- OpenCV for computer vision algorithms
- Three.js for the 360° viewer
- Inspired by classic panorama stitching papers and projects

