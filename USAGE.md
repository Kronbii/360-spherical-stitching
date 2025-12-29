# Usage Guide

Complete guide to using the 360° Spherical Panorama Stitching pipeline.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Configuration File](#configuration-file)
4. [Configuration Reference](#configuration-reference)
5. [Capturing Good Panoramas](#capturing-good-panoramas)
6. [Examples](#examples)
7. [Output Files](#output-files)
8. [Viewing Panoramas](#viewing-panoramas)
9. [Troubleshooting](#troubleshooting)
10. [Tips and Best Practices](#tips-and-best-practices)

## Installation

### Requirements

- Python 3.8 or higher
- OpenCV 4.8+
- NumPy, Pillow, exifread, natsort, pyyaml

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

## Quick Start

1. **Create a configuration file** (copy `config.yaml` and edit it):

```yaml
# Input source (choose one)
input_dir: ./photos  # Directory with your images
# OR
video: ./video.mov   # Video file

# Output directory
output_dir: ./output

# Output settings
output:
  pano_width: 4096
  output_format: jpg
  jpg_quality: 95
```

2. **Run the pipeline**:

```bash
python run.py config.yaml
```

3. **View your panorama**:

Open `output/viewer/index.html` in a web browser.

## Configuration File

All settings are controlled through a YAML configuration file. The file is self-documenting - see `config.yaml` for a complete example with all options.

### Basic Structure

```yaml
# Input (choose one)
input_dir: ./photos        # Directory with images
# OR
video: ./video.mov         # Video file

# Required
output_dir: ./output       # Where to save results

# Output settings
output:
  pano_width: 4096
  output_format: jpg
  jpg_quality: 95

# Feature matching
matching:
  match_width: 1600
  min_inliers: 30
  # ... more options

# Camera intrinsics
intrinsics:
  hfov_deg: 65.0
  calib_json: null

# Blending
blending:
  method: multiband
  # ... more options

# Video extraction (only if using video)
video_extraction:
  method: uniform
  num_frames: 40

# Debug
debug:
  enabled: false
  save_matches: true
```

## Configuration Reference

### Input Source

Choose **one** of the following:

- **`input_dir`**: Path to directory containing images (JPG, PNG, etc.)
- **`video`**: Path to video file (MOV, MP4, etc.)

**Example:**
```yaml
input_dir: ./photos
# OR
video: ./IMG_1480.MOV
```

### Output Settings

```yaml
output:
  pano_width: 4096      # Panorama width in pixels (height = width/2)
  output_format: jpg    # Options: jpg, png
  jpg_quality: 95       # JPEG quality (1-100, higher = better quality)
```

**Recommended values:**
- **Standard**: 4096 (good balance)
- **High quality**: 8192 (large file size)
- **Web/quick**: 2048 (faster processing)

### Feature Matching

```yaml
matching:
  match_width: 1600              # Width for matching (None = full resolution)
  match_full_res: false          # If true, use full resolution (match_width ignored)
  orb_nfeatures: 3000            # Number of ORB features to detect
  ratio_test_threshold: 0.75     # Lowe's ratio test (lower = stricter, 0.7-0.8 recommended)
  min_inliers: 30                # Minimum RANSAC inliers (lower = more lenient)
  symmetric_matching: false      # Cross-check matches (can be too strict)
  ransac_refinement: true        # Refine homography with all inliers (recommended)
  ransac_max_iters: 3000         # Max RANSAC iterations (higher = better, slower)
  use_clahe: false               # Apply CLAHE for low-contrast scenes
  disable_circular_closure: false # Set true if video doesn't loop
  rotation_smoothing_window: 3   # Temporal smoothing (3-15, larger = straighter lines)
```

**Key parameters:**
- **`match_width`**: Lower = faster matching, but may miss features. 1600 is a good default.
- **`min_inliers`**: Lower this (e.g., 20-30) if matches are failing. Higher (50-100) for better quality.
- **`rotation_smoothing_window`**: Increase to 7-11 for video sequences to straighten lines

### Camera Intrinsics

```yaml
intrinsics:
  hfov_deg: 65.0              # Horizontal field of view in degrees (fallback)
  calib_json: null            # Path to calibration JSON file (optional)
```

**HFOV guidelines:**
- iPhone 15 Pro: ~75°
- Samsung Galaxy S24: ~80°
- Google Pixel 8: ~77°
- Most phones: 65-85°

The pipeline tries to extract FOV from EXIF first, only uses `hfov_deg` as fallback.

**Custom calibration file:**
```json
{
    "fx": 1500.0,
    "fy": 1500.0,
    "cx": 2000.0,
    "cy": 1500.0,
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
}
```

### Blending

```yaml
blending:
  method: multiband           # Options: multiband, feather, sharp, none
  multiband_levels: 5         # Pyramid levels (5-7 recommended)
  multiband_sigma: 30.0       # Gaussian blur sigma (lower = sharper)
  feather_sigma: 50.0         # Gaussian blur sigma for feather blending
  sharp_blend_width: 2.0      # Blend zone width for sharp blending (1-3 pixels)
```

**Blending methods:**
- **`multiband`**: Best quality, handles exposure differences well (default)
- **`feather`**: Faster, smooth transitions
- **`sharp`**: Minimal blur, good for many close frames
- **`none`**: Hard seams, sharpest but may show seams

### Video Extraction

Only used when `video:` is specified in config.

```yaml
video_extraction:
  method: uniform             # Options: uniform, interval, fps, motion
  num_frames: 40              # For 'uniform' method
  frame_interval: 15          # For 'interval' method (every Nth frame)
  extract_fps: 2.0            # For 'fps' method (target FPS)
  min_frames: 20              # For 'motion' method (minimum frames)
  max_frames: 100             # For 'motion' method (maximum frames)
  motion_threshold: 0.02      # For 'motion' method (sensitivity)
```

**Methods:**
- **`uniform`**: Evenly distributed frames (good for constant rotation speed)
- **`interval`**: Every Nth frame (simple and predictable)
- **`fps`**: Extract at target FPS (e.g., 2 fps from 30 fps video)
- **`motion`**: Motion-based keyframes (adapts to variable rotation speed)

### Debug Settings

```yaml
debug:
  enabled: false              # Enable debug mode
  save_matches: true          # Save feature match visualizations
  save_warped_frames: 0       # Number of warped frames to save (0 = none)
  save_seams: true            # Save blending mask visualization
  verbose_logging: true       # Detailed logging
```

Enable debug mode to inspect:
- Feature matches between image pairs
- Individual warped frames
- Blending seams and weights

## Capturing Good Panoramas

### Camera Setup

1. **Use a tripod** (or stable surface): Minimizes parallax errors
2. **Rotate around optical center**: The lens center, not the camera body
3. **Lock exposure and white balance**: Prevents brightness/color variations
4. **Use manual focus**: Ensures consistent focus across shots

### Shooting Tips

1. **Overlap**: Ensure 30-50% overlap between consecutive shots
2. **Smooth rotation**: Rotate steadily in one direction
3. **Coverage**: For full 360°, shoot approximately `360°/HFOV + 2` images
4. **Avoid**: 
   - Moving objects (will cause ghosting)
   - Extreme lighting changes
   - Close foreground objects (parallax issues)

### Image Count Guidelines

| Phone HFOV | Images for 360° |
|------------|-----------------|
| ~65°       | 8-10 images     |
| ~75°       | 7-8 images      |
| ~85°       | 6-7 images      |

### Video Capture

For video input:
- Record at 30 fps or higher
- Rotate smoothly and steadily
- Ensure good overlap (slower rotation = more overlap)
- Use motion-based extraction for variable rotation speed

## Examples

### High Resolution Panorama

```yaml
output:
  pano_width: 8192
  jpg_quality: 98

matching:
  match_full_res: true  # Use full resolution for matching
```

### Fast Processing

```yaml
matching:
  match_width: 1200     # Smaller for faster matching

blending:
  method: feather       # Faster than multiband
```

### Low Contrast Scenes

```yaml
matching:
  use_clahe: true       # Enhance contrast for feature detection
  min_inliers: 25       # Lower threshold for difficult scenes
```

### Video with Motion-Based Extraction

```yaml
video: ./video.mov

video_extraction:
  method: motion
  min_frames: 30
  max_frames: 100
  motion_threshold: 0.02
```

### Minimal Blur (Sharp Blending)

```yaml
blending:
  method: sharp
  sharp_blend_width: 1.0  # Very sharp, minimal blending
```

### Straighter Lines (Video Sequences)

```yaml
matching:
  rotation_smoothing_window: 9  # Larger window = straighter lines

blending:
  method: none  # Or sharp with width 1.0
```

### Debug Mode

```yaml
debug:
  enabled: true
  save_matches: true
  save_warped_frames: 5
  save_seams: true
```

## Output Files

After running, you'll find in `output_dir/`:

```
output/
├── panorama.jpg           # Final equirectangular panorama
├── config.json            # Pipeline configuration used
├── intrinsics.json        # Camera intrinsics report
├── viewer/
│   ├── index.html         # Interactive 360° viewer
│   └── panorama.jpg       # Copy of panorama
└── debug/                 # (if debug.enabled: true)
    ├── matches/           # Feature match visualizations
    │   ├── match_0_1.jpg
    │   ├── match_1_2.jpg
    │   └── ...
    ├── warped/            # Warped frame previews
    │   ├── warped_0000.jpg
    │   └── ...
    └── seams.jpg          # Blending mask visualization
```

### intrinsics.json

Contains camera parameters:
- Focal length (fx, fy)
- Principal point (cx, cy)
- Distortion coefficients (if any)
- Estimated HFOV
- EXIF data used

### config.json

Complete configuration used for the run (useful for reproducibility).

## Viewing Panoramas

### Local Viewer

The pipeline automatically generates an HTML viewer using Three.js.

**Opening:**
```bash
# Path is printed after pipeline completes
# Open in browser:
file:///path/to/output/viewer/index.html
```

**Note**: Some browsers block local file access. If it doesn't load:
- Use Firefox (usually works)
- Start a local server:
  ```bash
  cd output
  python -m http.server 8000
  # Then open: http://localhost:8000/viewer/
  ```

### Online Viewers

Upload your panorama to:
- **YouTube**: Upload as 360° video (use video format)
- **Facebook**: Upload as 360° photo
- **Pannellum**: [pannellum.org](https://pannellum.org/)
- **Marzipano**: [marzipano.net](https://www.marzipano.net/)

### VR Headsets

Panoramas can be viewed in:
- **Oculus Quest/Go**: Via web browser or VR apps
- **Google Cardboard**: Using compatible apps
- **SteamVR**: Via compatible viewers

## Troubleshooting

### "Matching failed" Error

**Symptoms**: Pipeline stops with matching error

**Causes & Solutions:**
- **Insufficient overlap**: Ensure 30-50% overlap between images
- **Low contrast**: Enable `use_clahe: true` in config
- **Too strict matching**: Lower `min_inliers` (e.g., 20-30)
- **Wrong image order**: Check that images are in correct sequence
- **Extreme rotations**: Reduce rotation between shots

### Visible Seams

**Symptoms**: Visible boundaries between images in panorama

**Solutions:**
- Use `method: multiband` in blending config (default)
- Increase `multiband_levels` (e.g., 6 or 7)
- Lock camera exposure before shooting
- Use `method: sharp` with small `sharp_blend_width` for minimal blur

### Panorama Not Loading in Viewer

**Symptoms**: Blank viewer or error message

**Solutions:**
- Use Firefox browser
- Start local server (see [Viewing Panoramas](#viewing-panoramas))
- Check browser console for errors
- Verify panorama.jpg exists in viewer directory

### Ghosting/Artifacts

**Symptoms**: Duplicate objects or blurry areas

**Causes:**
- **Parallax**: Camera not rotating around optical center
- **Moving objects**: People/cars in scene
- **Close foreground**: Objects too close to camera

**Solutions:**
- Use tripod with proper rotation point
- Avoid close foreground objects
- Remove moving objects from scene
- Reshoot with careful rotation

### Excessive Blur

**Symptoms**: Panorama looks blurry despite good source images

**Solutions:**
- Use `method: sharp` or `method: none` for minimal blur
- Reduce `multiband_sigma` or `feather_sigma`
- Increase `rotation_smoothing_window` to reduce jitter (straighter = less perceived blur)

### Lines Not Straight

**Symptoms**: Vertical/horizontal lines appear curved

**Solutions:**
- Increase `rotation_smoothing_window` (e.g., 7-15)
- Use `match_full_res: true` for better alignment
- Enable `ransac_refinement: true`
- Check camera calibration/distortion correction

### Out of Memory (OOM)

**Symptoms**: Pipeline crashes with memory error

**Solutions:**
- Use `blending.method: none` (memory-efficient sequential processing)
- Reduce `pano_width` (e.g., 2048 instead of 4096)
- Reduce `match_width` or number of images
- Process smaller batches

### Poor Alignment

**Symptoms**: Images don't align correctly

**Solutions:**
- Increase `ransac_max_iters` (e.g., 5000-10000)
- Use `match_full_res: true`
- Lower `ratio_test_threshold` (stricter matching)
- Enable `symmetric_matching: true` (if you have many features)
- Check HFOV setting in intrinsics

## Tips and Best Practices

### For Best Quality

1. **Use tripod** with proper rotation point
2. **Lock camera settings** (exposure, white balance, focus)
3. **Good overlap** (30-50%)
4. **Stable lighting** (avoid harsh shadows/sun)
5. **High resolution** output (4096+ width)

### For Speed

1. **Lower matching resolution** (`match_width: 1200`)
2. **Feather blending** instead of multiband
3. **Smaller panorama** (2048 width)
4. **Disable debug mode**

### For Video Input

1. **Record at 30+ fps** for smooth motion
2. **Use motion extraction** for variable rotation speed
3. **Increase smoothing window** (7-11) for straighter lines
4. **Longer videos** = more frames = smoother result

### For Difficult Scenes

1. **Enable CLAHE** (`use_clahe: true`) for low contrast
2. **Lower min_inliers** (20-30) for challenging matches
3. **Use debug mode** to inspect matches
4. **Check image order** and overlap

### Configuration Tips

- **Start with defaults**, then adjust based on results
- **Save config.json** from successful runs for reference
- **Use debug mode** to understand what's happening
- **Experiment with blending methods** to find your preference

---

For technical details, see [TECHNICAL.md](TECHNICAL.md).

