# Command Line Reference for run.py

Complete documentation of all command-line arguments and options for the 360° panorama stitching pipeline.

## Required Arguments

### Input Source (choose one)
- **`--input_dir`** / **`-i`**  
  Type: Path  
  Description: Directory containing input images (JPEG/PNG)  
  Example: `--input_dir ./photos`

- **`--video`** / **`-v`**  
  Type: Path  
  Description: Input video file (.mov, .mp4, etc.)  
  Example: `--video ./video.mov`  
  Note: Use this when processing video instead of individual images

- **`--output_dir`** / **`-o`**  
  Type: Path  
  Required: Yes  
  Description: Directory where panorama and viewer will be saved  
  Example: `--output_dir ./output`

---

## Video Extraction Options

These options only apply when using `--video`:

### `--extract_method`
Type: Choice  
Default: `uniform`  
Choices: `uniform`, `interval`, `fps`, `motion`  
Description: Method for extracting frames from video

- **`uniform`**: Extract N frames uniformly distributed across video duration
  - Use with: `--num_frames`
- **`interval`**: Extract every Nth frame
  - Use with: `--frame_interval`
- **`fps`**: Extract at target frames per second rate
  - Use with: `--extract_fps`
- **`motion`**: Smart extraction based on camera motion (keyframes)

Example: `--extract_method fps`

### `--num_frames`
Type: Integer  
Default: `40`  
Description: Number of frames to extract (for `uniform` method)  
Example: `--num_frames 50`

### `--extract_fps`
Type: Float  
Default: `2.0`  
Description: Target frames per second for `fps` extraction method  
Example: `--extract_fps 5.0` (extracts 5 frames per second)

### `--frame_interval`
Type: Integer  
Default: `15`  
Description: Extract every Nth frame (for `interval` method)  
Example: `--frame_interval 10` (every 10th frame)

---

## Output Settings

### `--pano_width`
Type: Integer  
Default: `4096`  
Description: Output panorama width in pixels (height = width/2 for equirectangular)  
Example: `--pano_width 2048` (creates 2048×1024 panorama)  
Note: Larger values = higher quality but more memory usage

### `--output_format`
Type: Choice  
Default: `jpg`  
Choices: `jpg`, `png`  
Description: Output image format  
Example: `--output_format png`

---

## Feature Matching Settings

### `--match_width`
Type: Integer  
Default: `1600`  
Description: Width for downscaled images during feature matching (speeds up matching)  
Example: `--match_width 800`  
Note: Lower values = faster matching, but may reduce matching quality

### `--min_inliers`
Type: Integer  
Default: `60`  
Description: Minimum RANSAC inliers required for a valid match between image pairs  
Example: `--min_inliers 40`  
Note: Lower values = more lenient matching (may include bad matches)

### `--clahe`
Type: Flag (no value)  
Default: `False`  
Description: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for feature extraction  
Use when: Images have low contrast or poor lighting  
Example: `--clahe`

---

## Camera Intrinsics Settings

### `--hfov_deg`
Type: Float  
Default: `65.0`  
Description: Horizontal field of view in degrees (used when EXIF data unavailable)  
Example: `--hfov_deg 42.0`  
Notes:
- Phone cameras: ~42-45° for portrait mode, ~65-70° for landscape
- If EXIF focal length is available, it takes precedence
- Incorrect HFOV causes poor stitching alignment

### `--calib_json`
Type: Path  
Default: `None`  
Description: Path to camera calibration JSON file with intrinsics matrix  
Example: `--calib_json ./camera_calib.json`  
Format: JSON with `camera_matrix_K` (3×3) and optional `dist_coeffs`

---

## Blending Settings

### `--blend`
Type: Choice  
Default: `multiband`  
Choices: `multiband`, `feather`  
Description: Blending method for combining warped images

- **`multiband`**: Laplacian pyramid blending (higher quality, more memory)
- **`feather`**: Gaussian blur-based feather blending (faster, less memory)

Example: `--blend feather`

### `--blend_levels`
Type: Integer  
Default: `5`  
Description: Number of pyramid levels for multiband blending  
Example: `--blend_levels 4`  
Note: Only applies to multiband blending

---

## Debug Options

### `--debug`
Type: Flag (no value)  
Default: `False`  
Description: Enable debug mode (saves intermediate visualizations to `output/debug/`)  
Example: `--debug`  
Saves:
- Feature match visualizations
- Warped images and masks
- Blending intermediate results

---

## Complete Examples

### Basic Usage (Images)
```bash
python run.py --input_dir ./photos --output_dir ./output
```

### Video with FPS Extraction
```bash
python run.py --video ./video.mov --output_dir ./output \
  --extract_method fps --extract_fps 5.0
```

### High Quality with Custom Settings
```bash
python run.py --input_dir ./photos --output_dir ./output \
  --pano_width 8192 --blend multiband --debug
```

### Low Memory / Fast Processing
```bash
python run.py --video ./video.mov --output_dir ./output \
  --pano_width 2048 --blend feather --match_width 1200
```

### Portrait Mode Phone Camera
```bash
python run.py --video ./video.mov --output_dir ./output \
  --hfov_deg 42.0 --extract_method fps --extract_fps 3.0
```

### With Camera Calibration
```bash
python run.py --input_dir ./photos --output_dir ./output \
  --calib_json ./my_camera_calib.json
```

### Low Contrast Images (with CLAHE)
```bash
python run.py --input_dir ./photos --output_dir ./output \
  --clahe --min_inliers 40
```

---

## Tips for Best Results

1. **Frame Rate**: For video, start with `--extract_fps 2.0` to `5.0`
2. **HFOV**: Use `--hfov_deg 42.0` for portrait mode, `65.0` for landscape
3. **Memory**: Use `--blend feather` if you get out-of-memory errors
4. **Quality**: Lower `--pano_width` for faster processing and less memory
5. **Matching**: Lower `--min_inliers` if matching fails, but may reduce quality

