   j# 360° Spherical Panorama Stitching

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

### Basic Usage

```bash
python run.py --input_dir ./photos --output_dir ./output
```

### Full Options

```bash
python run.py \
    --input_dir ./photos \
    --output_dir ./output \
    --pano_width 4096 \
    --match_width 1600 \
    --blend multiband \
    --hfov_deg 65 \
    --debug
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir`, `-i` | *required* | Directory containing input images |
| `--output_dir`, `-o` | *required* | Output directory for panorama and viewer |
| `--pano_width` | 4096 | Output panorama width (height = width/2) |
| `--match_width` | 1600 | Image width for feature matching |
| `--blend` | multiband | Blending method: `multiband` or `feather` |
| `--blend_levels` | 5 | Pyramid levels for multiband blending |
| `--hfov_deg` | 65 | Horizontal FOV fallback (if no EXIF) |
| `--calib_json` | None | Camera calibration JSON file |
| `--clahe` | False | Apply CLAHE for feature extraction |
| `--min_inliers` | 60 | Minimum RANSAC inliers required |
| `--output_format` | jpg | Output format: `jpg` or `png` |
| `--debug` | False | Save debug visualizations |

### Examples

**High resolution panorama:**
```bash
python run.py -i ./photos -o ./output --pano_width 8192
```

**Fast processing with feather blending:**
```bash
python run.py -i ./photos -o ./output --blend feather --match_width 1200
```

**Debug mode with CLAHE (for low contrast images):**
```bash
python run.py -i ./photos -o ./output --clahe --debug
```

**With custom calibration:**
```bash
python run.py -i ./photos -o ./output --calib_json ./camera_calib.json
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
└── debug/                 # (if --debug enabled)
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
  - Try `--clahe` flag for low contrast scenes
  - Reduce `--min_inliers` (e.g., 40) for difficult scenes
  - Check image order (may be misordered)

### Visible Seams

- **Cause**: Exposure differences or insufficient blending
- **Solutions**:
  - Lock camera exposure before shooting
  - Use `--blend multiband` (default)
  - Increase `--blend_levels` (e.g., 6 or 7)

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

