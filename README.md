# 360° Spherical Panorama Stitching

Turns a handheld phone sweep into an equirectangular 360° panorama. The geometry
comes entirely from image features and a pure-rotation camera model — no gyroscope
data, no tripod, no stitching app.

**[Live demo](https://360.ramikronbi.com)** · [Usage guide](USAGE.md) · [Method](TECHNICAL.md) · [Smoothing notes](TEMPORAL_SMOOTHING.md)

![Equirectangular panorama of a living room, stitched from 309 phone frames](showcase/panorama.jpg)

## How it works

Give it a video or a folder of stills. Each stage below is a module in [src/](src/):

1. **Frames** — pulled from video at a fixed interval, uniform count, target fps, or
   by motion, then sorted by EXIF timestamp with a filename fallback
   ([video_utils.py](src/video_utils.py), [io_utils.py](src/io_utils.py)).
2. **Intrinsics** — focal length from EXIF where available, otherwise from a stated
   horizontal field of view or a calibration JSON ([intrinsics.py](src/intrinsics.py)).
3. **Matching** — ORB keypoints, Lowe's ratio test, then RANSAC for a homography per
   adjacent pair. If a pair fails, it retries across the gap at (i, i+2) and
   interpolates ([features.py](src/features.py)).
4. **Rotation** — each homography becomes a rotation, `R = K⁻¹HK`, orthonormalized by
   SVD so it stays a true rotation, then chained into global orientations. Pairs that
   never matched borrow from their neighbours ([rotation.py](src/rotation.py)).
5. **Smoothing** — an optional moving average over the chained rotations. It exists
   because independent per-pair estimates leave a few tenths of a degree of pitch and
   roll error on every frame, which shows up as a staircase along any long straight
   edge. On the sample run it cuts pitch wobble 13.5× and roll wobble 5.2×.
6. **Warp** — inverse-mapped to equirectangular: for each output pixel, take the world
   direction, rotate into each camera's frame, project, sample
   ([warp_sphere.py](src/warp_sphere.py)).
7. **Blend and fill** — `multiband`, `feather`, `sharp`, or `none` across the overlaps,
   then inpainting for whatever the sweep never saw ([blend.py](src/blend.py)).
8. **Viewer** — a self-contained Three.js page written next to the panorama.

## Install

```bash
git clone https://github.com/Kronbii/360-spherical-stitching.git
cd 360-spherical-stitching
pip install -r requirements.txt
```

Python 3.12, OpenCV, NumPy, Pillow, exifread, natsort. CPU only — there is no GPU path
and none is needed.

## Usage

Everything is driven by one YAML file:

```yaml
video: ./IMG_1480_2.MOV          # or: input_dir: ./photos
output_dir: ./output/livingroom

video_extraction:
  method: interval               # uniform · interval · fps · motion
  frame_interval: 2

matching:
  match_full_res: true
  min_inliers: 200
  rotation_smoothing_window: 17  # moving average over chained rotations

intrinsics:
  hfov_deg: 42                   # fallback when EXIF has no focal length

blending:
  method: none                   # multiband · feather · sharp · none

output:
  pano_width: 4096               # height is always width / 2
```

```bash
python run.py config.yaml
```

The run prints a rotation summary as it goes — recovered sweep, per-step angle, and any
pair whose homography looked suspicious. Output lands in `output_dir`:

```
output/livingroom/
├── panorama.jpg          # the equirectangular result
├── frames/               # extracted frames, when the input was video
├── intrinsics.json       # what the camera model resolved to
├── config.json           # the exact settings this run used
└── viewer/index.html     # drag-to-look viewer, open it in a browser
```

Some browsers refuse to load the panorama over `file://`. Serve it instead:

```bash
cd output/livingroom && python -m http.server 8000   # then open localhost:8000/viewer/
```

[USAGE.md](USAGE.md) documents every option and its default.

## What one run looks like

A 309-frame handheld sweep of a living room, every 2nd frame of a phone video, matched
at full 1080×1920 resolution on a Ryzen 7 5800H:

| | |
|---|---|
| Frames in | 309 |
| Recovered sweep | 333° |
| RANSAC inliers per pair | 921 median |
| Pairs recovered by interpolation | 15 of 308 |
| Sphere actually imaged | 36.9%, spanning +31° to −37° elevation |
| Output | 4096 × 2048 |

| Stage | Time |
|---|---|
| Feature matching, 308 pairs at full resolution | 19 s |
| Warping 309 frames to 4096 × 2048 | 96 s |
| Gap fill | 4 s |
| **Total** | **~2 min** |

The demo site walks through the same run stage by stage, including a real ORB match
visualization and a wipe comparison of smoothing on versus off:
**[360.ramikronbi.com](https://360.ramikronbi.com)**

## v2: gyro prior and bundle adjustment (validated in simulation)

A second orientation path for captures that come with a gyroscope log. It is validated
on simulated data only so far; `run.py` still uses the v1 path for real videos.

1. **Gyro prior** — the gyro is integrated between frame timestamps to give every frame
   a dead-reckoned orientation ([imu.py](src/imu.py)).
2. **Pair selection** — adjacent and nearby pairs, plus any pair the prior says should
   overlap. After a full turn that includes the last frames against the first ones: the
   loop closures chaining never sees.
3. **Guided matching** — ORB matches are only searched near where the prior predicts
   each keypoint lands, and a rotation-only model (3 degrees of freedom, two-point
   RANSAC on bearing vectors) replaces the homography ([guided_match.py](src/guided_match.py)).
4. **Bundle adjustment** — every rotation and one shared focal length solved together
   over all pairs, with a soft-L1 loss, starting from the gyro prior
   ([bundle_adjust.py](src/bundle_adjust.py)).
5. **Warp and blend** — unchanged from v1.

The simulator ([simulate.py](src/simulate.py)) renders pinhole frames from an existing
equirectangular image for a camera turning about its centre, and a 200 Hz gyro with
bias, bias random walk, scale error and noise. Ground truth is exact, so each stage can
be scored:

```bash
python run_sim.py --world showcase/panorama.jpg --out output/sim
python scripts/make_v2_media.py --sim output/sim     # figures and a short video
```

On the default run (73 frames, 396° of yaw, starting focal length 3.5% short):

| Orientation source              | Mean error | Last frame | Reprojection RMS |
| ------------------------------- | ---------: | ---------: | ---------------: |
| Gyro only                       |      2.01° |      4.27° |          19.0 px |
| Pairwise chain, v1 homography   |      1.35° |      2.09° |          14.8 px |
| Pairwise chain, v2 rotation     |      6.13° |     12.21° |         100.1 px |
| Bundle adjusted                 |      0.04° |      0.03° |          0.99 px |

Bundle adjustment also recovers the focal length to within 0.02%. The rotation-only chain
is the most sensitive to a wrong focal length, which is why the global solve matters
more than the pairwise model: with the focal length exact, the same chain ends 0.12° off
and bundle adjustment 0.02°.

Not simulated: parallax from camera translation, lens distortion, rolling shutter, motion
blur, camera–gyro time offset and misalignment. Real captures will score worse.

## Assumptions and limits

- **Rotation only.** The camera is assumed to turn about its optical centre. Translate
  while sweeping and nearby objects will not line up — that is parallax, and a
  homography cannot represent it.
- **A phone-height sweep does not cover a sphere.** One horizontal pass images roughly a
  third of it; the rest is inpainted from its surroundings and will look smeared. Tilt up
  and down across several passes if you want real pole coverage.
- **Texture drives the estimate.** Blank walls, blown highlights, and repeating patterns
  produce few usable inliers. The pipeline carries on by borrowing a neighbour's rotation
  rather than dropping the frame.
- **No bundle adjustment on the v1 path, and no exposure matching.** Error accumulates
  along the chain instead of being distributed around the loop, and frames keep their own
  exposure. The v2 path above adds bundle adjustment, but needs a gyro log.

## Tests

```bash
python run_tests.py          # or: pytest -q
```

More than 200 tests cover config, IO, intrinsics, feature matching, rotation, blending,
and warping.

## Repository layout

```
run.py            entry point: load config, extract frames, run the pipeline
run_sim.py        v2 validation on a simulated sweep with ground truth
scripts/          media for the v2 simulation (figures, video)
src/pipeline.py   the eight stages, in order, with logging
src/              one module per stage (see "How it works")
tests/            pytest suite
docs/             the demo site (deployed on Vercel)
showcase/         sample panorama and capture video
```

## License

MIT — see [LICENSE](LICENSE). Contributions welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).
