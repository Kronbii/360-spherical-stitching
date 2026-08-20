"""
Main pipeline module for 360° spherical panorama stitching.

Orchestrates the complete workflow:
1. Load and sort images
2. (Optional) Undistort images
3. Estimate camera intrinsics
4. Match features between adjacent images
5. Extract rotations from homographies
6. Chain global rotations
7. Warp images to equirectangular
8. Blend into final panorama
9. Create viewer
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .blend import blend_panorama, fill_gaps, create_seam_visualization
from .config import (
    CalibrationData,
    PipelineConfig,
)
from .features import (
    check_matching_quality,
    match_sequential_pairs,
    save_match_visualization,
)
from .intrinsics import (
    estimate_intrinsics,
    save_intrinsics_report,
    undistort_images,
)
from .io_utils import (
    get_image_files,
    load_image,
    load_images,
    save_image,
    sort_images_robustly,
    get_image_dimensions,
    ImageInfo,
)
from .rotation import (
    chain_rotations,
    compute_relative_rotations,
    log_rotation_summary,
)
from .circular_match import (
    detect_circular_closure,
    enforce_open_sweep,
    head_tail_overlap,
    trim_excess_frames,
)
from .warp_sphere import (
    estimate_panorama_coverage,
    save_debug_warps,
    warp_all_images,
    warp_and_blend_sequential,
)

logger = logging.getLogger(__name__)


class PanoramaStitchingError(Exception):
    """Custom exception for panorama stitching errors."""
    pass


def setup_output_directory(config: PipelineConfig) -> None:
    """Create output directory structure."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    if config.debug.enabled:
        (config.output_dir / "debug").mkdir(exist_ok=True)
        (config.output_dir / "debug" / "matches").mkdir(exist_ok=True)
        (config.output_dir / "debug" / "warped").mkdir(exist_ok=True)


def create_viewer_folder(
    panorama_path: Path,
    output_dir: Path,
    panorama_filename: str = "panorama.jpg"
) -> Path:
    """
    Create viewer folder with HTML and copy of panorama.
    
    Args:
        panorama_path: Path to the generated panorama.
        output_dir: Output directory.
        panorama_filename: Filename for panorama in viewer.
        
    Returns:
        Path to viewer directory.
    """
    viewer_dir = output_dir / "viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy panorama to viewer folder
    dest_pano = viewer_dir / panorama_filename
    shutil.copy(panorama_path, dest_pano)
    
    # Copy HTML template from package
    template_path = Path(__file__).parent / "viewer" / "index.html"
    if template_path.exists():
        shutil.copy(template_path, viewer_dir / "index.html")
    else:
        # Generate HTML inline if template missing
        generate_viewer_html(viewer_dir, panorama_filename)
    
    logger.info(f"Created viewer at {viewer_dir}")
    return viewer_dir


def generate_viewer_html(viewer_dir: Path, panorama_filename: str) -> None:
    """
    Generate viewer HTML file.
    
    Args:
        viewer_dir: Viewer directory.
        panorama_filename: Filename of panorama image.
    """
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>360° Panorama Viewer</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            overflow: hidden;
        }}
        #container {{ 
            width: 100vw; 
            height: 100vh; 
            position: relative;
        }}
        #info {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(10, 10, 15, 0.85);
            padding: 16px 24px;
            border-radius: 12px;
            font-size: 14px;
            z-index: 100;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        #info h1 {{
            font-size: 18px;
            margin-bottom: 8px;
            color: #fff;
        }}
        #info p {{
            opacity: 0.7;
            line-height: 1.5;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 18px;
            z-index: 200;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div id="info">
        <h1>360° Panorama</h1>
        <p>Drag to look around<br>Scroll to zoom</p>
    </div>
    <div id="loading">Loading panorama...</div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        // Three.js 360 panorama viewer
        let camera, scene, renderer;
        let isUserInteracting = false;
        let onPointerDownMouseX = 0, onPointerDownMouseY = 0;
        let lon = 0, onPointerDownLon = 0;
        let lat = 0, onPointerDownLat = 0;
        let phi = 0, theta = 0;

        init();
        animate();

        function init() {{
            const container = document.getElementById('container');
            
            camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 1, 1100);
            scene = new THREE.Scene();
            
            const geometry = new THREE.SphereGeometry(500, 60, 40);
            geometry.scale(-1, 1, 1);
            
            const texture = new THREE.TextureLoader().load(
                '{panorama_filename}',
                function() {{
                    document.getElementById('loading').style.display = 'none';
                }},
                undefined,
                function(err) {{
                    document.getElementById('loading').textContent = 'Error loading panorama';
                    console.error('Error loading panorama:', err);
                }}
            );
            
            const material = new THREE.MeshBasicMaterial({{ map: texture }});
            const mesh = new THREE.Mesh(geometry, material);
            scene.add(mesh);
            
            renderer = new THREE.WebGLRenderer();
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(window.innerWidth, window.innerHeight);
            container.appendChild(renderer.domElement);
            
            // Event listeners
            container.addEventListener('pointerdown', onPointerDown);
            container.addEventListener('pointermove', onPointerMove);
            container.addEventListener('pointerup', onPointerUp);
            container.addEventListener('wheel', onWheel);
            window.addEventListener('resize', onWindowResize);
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function onPointerDown(event) {{
            isUserInteracting = true;
            onPointerDownMouseX = event.clientX;
            onPointerDownMouseY = event.clientY;
            onPointerDownLon = lon;
            onPointerDownLat = lat;
        }}

        function onPointerMove(event) {{
            if (isUserInteracting) {{
                lon = (onPointerDownMouseX - event.clientX) * 0.1 + onPointerDownLon;
                lat = (event.clientY - onPointerDownMouseY) * 0.1 + onPointerDownLat;
            }}
        }}

        function onPointerUp() {{
            isUserInteracting = false;
        }}

        function onWheel(event) {{
            const fov = camera.fov + event.deltaY * 0.05;
            camera.fov = THREE.MathUtils.clamp(fov, 30, 90);
            camera.updateProjectionMatrix();
        }}

        function animate() {{
            requestAnimationFrame(animate);
            
            lat = Math.max(-85, Math.min(85, lat));
            phi = THREE.MathUtils.degToRad(90 - lat);
            theta = THREE.MathUtils.degToRad(lon);
            
            const x = 500 * Math.sin(phi) * Math.cos(theta);
            const y = 500 * Math.cos(phi);
            const z = 500 * Math.sin(phi) * Math.sin(theta);
            
            camera.lookAt(x, y, z);
            renderer.render(scene, camera);
        }}
    </script>
</body>
</html>'''
    
    with open(viewer_dir / "index.html", 'w') as f:
        f.write(html_content)


def run_pipeline(config: PipelineConfig) -> Path:
    """
    Run the complete panorama stitching pipeline.
    
    Args:
        config: Pipeline configuration.
        
    Returns:
        Path to generated panorama.
        
    Raises:
        PanoramaStitchingError: If pipeline fails at any stage.
    """
    logger.info("=" * 60)
    logger.info("360° SPHERICAL PANORAMA STITCHING PIPELINE")
    logger.info("=" * 60)
    
    # Setup output directory
    setup_output_directory(config)
    
    # Save configuration
    config_path = config.output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    # ============================================================
    # STEP 1: Load and sort images
    # ============================================================
    logger.info("\n[STEP 1] Loading and sorting images...")
    
    image_files = get_image_files(config.input_dir)
    image_infos, sort_method = sort_images_robustly(image_files)
    
    logger.info(f"Sorting method: {sort_method}")
    logger.info(f"Image order:")
    for i, info in enumerate(image_infos):
        logger.info(f"  {i+1}. {info.path.name}")
    
    if len(image_infos) < 2:
        raise PanoramaStitchingError("Need at least 2 images to create a panorama")
    
    # Get image dimensions from first image
    image_width, image_height = get_image_dimensions(image_infos[0].path)
    logger.info(f"Image dimensions: {image_width}x{image_height}")
    
    # ============================================================
    # STEP 2: Estimate camera intrinsics
    # ============================================================
    logger.info("\n[STEP 2] Estimating camera intrinsics...")
    
    calib = estimate_intrinsics(
        image_infos,
        image_width,
        image_height,
        config.intrinsics
    )
    
    # Save intrinsics report
    intrinsics_path = config.output_dir / "intrinsics.json"
    save_intrinsics_report(calib, image_infos, image_width, image_height, intrinsics_path)
    
    # ============================================================
    # STEP 3: Load images for matching (downscaled or full resolution)
    # ============================================================
    logger.info("\n[STEP 3] Loading images for feature matching...")
    
    if config.matching.match_width is None:
        match_scale = 1.0
        match_images = load_images(image_infos, None)
        logger.info(f"Using full resolution images for matching (no downscaling)")
    else:
        match_scale = config.matching.match_width / image_width
        match_images = load_images(image_infos, config.matching.match_width)
        logger.info(f"Match scale: {match_scale:.3f} ({config.matching.match_width}px width)")
    
    # ============================================================
    # STEP 4: (Optional) Undistort images
    # ============================================================
    if calib.dist_coeffs and any(d != 0 for d in calib.dist_coeffs):
        logger.info("\n[STEP 3b] Undistorting images...")
        match_images, calib = undistort_images(match_images, calib)
    
    # ============================================================
    # STEP 5: Feature matching
    # ============================================================
    logger.info("\n[STEP 4] Matching features between adjacent images...")
    
    match_results = match_sequential_pairs(
        match_images,
        config.matching,
        scale=match_scale
    )
    
    # Check matching quality
    success, message = check_matching_quality(match_results, config.matching.min_inliers)
    if not success:
        # Count failed pairs
        failed_count = sum(1 for r in match_results if not r.success)
        total_pairs = len(match_results)
        success_rate = (total_pairs - failed_count) / total_pairs * 100
        
        logger.warning(f"Some pairs failed to match ({failed_count}/{total_pairs}, {success_rate:.1f}% success rate)")
        logger.warning("Continuing with identity rotations for failed pairs...")
        logger.warning(message)
        
        # Save debug visualizations for failed matches
        if config.debug.enabled:
            for result in match_results:
                if not result.success:
                    vis_path = config.output_dir / "debug" / "matches" / f"match_{result.src_idx}_{result.dst_idx}.jpg"
                    save_match_visualization(
                        match_images[result.src_idx],
                        match_images[result.dst_idx],
                        result,
                        vis_path
                    )
        
        # Don't fail - continue with identity rotations for failed pairs
        # The rotation computation will handle this automatically
    
    # Save match visualizations if debug enabled
    if config.debug.enabled and config.debug.save_matches:
        for result in match_results:
            vis_path = config.output_dir / "debug" / "matches" / f"match_{result.src_idx}_{result.dst_idx}.jpg"
            save_match_visualization(
                match_images[result.src_idx],
                match_images[result.dst_idx],
                result,
                vis_path
            )
    
    # ============================================================
    # STEP 6: Compute rotations
    # ============================================================
    logger.info("\n[STEP 5] Computing rotations from homographies...")
    
    # Scale K for match resolution
    K_match = calib.K.copy()
    K_match[0, 0] *= match_scale
    K_match[1, 1] *= match_scale
    K_match[0, 2] *= match_scale
    K_match[1, 2] *= match_scale
    
    relative_rotations, rotation_diags = compute_relative_rotations(
        match_results,
        K_match,
        scale_factor=1.0  # Already at match scale
    )
    
    global_rotations = chain_rotations(
        relative_rotations,
        apply_smoothing=config.matching.rotation_smoothing_window > 1,
        smoothing_window=config.matching.rotation_smoothing_window
    )
    log_rotation_summary(global_rotations, rotation_diags)
    
    # ============================================================
    # STEP 5b: Detect and trim circular closure (if >360° captured)
    # ============================================================
    if config.matching.disable_circular_closure:
        logger.info("Circular closure detection is disabled")
        trim_idx = None
    else:
        trim_idx = detect_circular_closure(
            match_images,
            match_results,
            global_rotations,
            config.matching,
            scale=match_scale
        )
        
        if trim_idx is not None and trim_idx < len(image_infos) - 1:
            # Trim excess frames after closure point
            match_images, image_infos, match_results, global_rotations = trim_excess_frames(
                match_images,
                image_infos,
                match_results,
                global_rotations,
                trim_idx
            )
            
            logger.info(f"Circular closure detected: trimmed to {len(image_infos)} frames")

    # ============================================================
    # STEP 5c: Keep the sweep from wrapping onto itself
    # ============================================================
    # A chain that over-estimates can report enough yaw to close the circle even when the
    # camera never came back round. The warp would then drop the last frames onto the same
    # arc as the first ones and paint unrelated content over them. Trimming already handled
    # the case where the frames genuinely do overlap; anything left is drift.
    hfov_deg = 2.0 * np.degrees(np.arctan(calib.K[0, 2] / calib.K[0, 0]))
    collision = head_tail_overlap(match_images, global_rotations, hfov_deg)
    if trim_idx is None and collision is not None:
        first, last = collision
        logger.warning(f"Frames {first}-{last} would land on the first frame's arc")
        global_rotations, wrap_report = enforce_open_sweep(global_rotations, hfov_deg)
    elif collision is not None:
        logger.info("Wrap overlap resolved by trimming at the closure point")

    # ============================================================
    # STEP 6: Load full-resolution images and warp
    # ============================================================
    logger.info("\n[STEP 6] Warping images to equirectangular projection...")
    
    # Re-estimate calibration at full resolution
    calib_full = estimate_intrinsics(
        image_infos,
        image_width,
        image_height,
        config.intrinsics
    )
    
    # Check if we need undistortion
    needs_undistort = calib_full.dist_coeffs and any(d != 0 for d in calib_full.dist_coeffs)
    if needs_undistort:
        logger.info("Will undistort full-resolution images during sequential processing...")
    
    # Use sequential processing for "none" blending (memory-efficient)
    # This avoids loading all full-resolution images and warped images into memory at once
    use_sequential = config.blending.method == "none"
    
    if use_sequential:
        logger.info("Using sequential processing for memory efficiency...")
        
        # Initialize panorama for "none" blending (accumulates result)
        H, W = config.output.pano_height, config.output.pano_width
        panorama = np.zeros((H, W, 3), dtype=np.uint8)
        best_mask_value = np.zeros((H, W), dtype=np.uint8)
        
        # Create undistort function if needed
        undistort_func = None
        if needs_undistort:
            from .intrinsics import undistort_image_single
            import cv2
            
            # Compute undistortion maps once (they're the same for all images)
            dist_coeffs = np.array(calib_full.dist_coeffs, dtype=np.float64)
            K = calib_full.K
            h, w = image_height, image_width
            new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (w, h), alpha=0)
            map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (w, h), cv2.CV_32FC1)
            
            # Update calibration to use new K matrix (no distortion after undistortion)
            from .config import CalibrationData
            calib_full = CalibrationData(
                fx=new_K[0, 0],
                fy=new_K[1, 1],
                cx=new_K[0, 2],
                cy=new_K[1, 2],
                dist_coeffs=None
            )
            
            # Create undistort function using precomputed maps
            def undistort_func(img: np.ndarray) -> np.ndarray:
                return cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        
        # Sequential warp and blend callback
        def blend_callback(warped: np.ndarray, mask: np.ndarray, idx: int):
            nonlocal panorama, best_mask_value
            is_better = mask > best_mask_value
            panorama[is_better] = warped[is_better]
            best_mask_value = np.maximum(best_mask_value, mask)
        
        # Process images sequentially
        masks = warp_and_blend_sequential(
            image_infos,
            global_rotations,
            calib_full,
            config.output,
            blend_callback,
            image_width,
            image_height,
            undistort_func
        )
        
        warped_images = None  # Not stored for "none" blending
        
    else:
        # For multiband/feather blending, we need all images in memory
        # Load full resolution images
        full_images = load_images(image_infos, max_width=None)
        
        # If we have distortion coefficients, undistort full-res images too
        if needs_undistort:
            logger.info("Undistorting full-resolution images...")
            full_images, calib_full = undistort_images(full_images, calib_full)
        
        # Warp images
        warped_images, masks = warp_all_images(
            full_images,
            global_rotations,
            calib_full,
            config.output
        )
        
        # Free full_images from memory
        del full_images
        
        # Coverage statistics
        coverage = estimate_panorama_coverage(masks)
        logger.info(f"Panorama coverage: {coverage['coverage_percent']:.1f}%")
        
        # Save debug warps if enabled
        if config.debug.enabled and config.debug.save_warped_frames > 0:
            save_debug_warps(warped_images, masks, config.output_dir, config.debug.save_warped_frames)
        
        # ============================================================
        # STEP 8: Blend panorama
        # ============================================================
        logger.info("\n[STEP 7] Blending panorama...")
        
        panorama = blend_panorama(warped_images, masks, config.blending)
        
        # Free warped_images from memory
        del warped_images
    
    # Coverage statistics (for "none" blending)
    if config.blending.method == "none":
        coverage = estimate_panorama_coverage(masks)
        logger.info(f"Panorama coverage: {coverage['coverage_percent']:.1f}%")
    
    # Fill any gaps
    panorama = fill_gaps(panorama, masks)
    
    # Save debug seam visualization (only if we have warped_images stored)
    if config.debug.enabled and config.debug.save_seams and warped_images is not None:
        seam_vis = create_seam_visualization(warped_images, masks)
        cv2.imwrite(str(config.output_dir / "debug" / "seams.jpg"), seam_vis)
    
    # ============================================================
    # STEP 9: Save panorama and create viewer
    # ============================================================
    logger.info("\n[STEP 8] Saving panorama and creating viewer...")
    
    # Save panorama
    pano_filename = f"panorama.{config.output.output_format}"
    pano_path = config.output_dir / pano_filename
    save_image(panorama, pano_path, config.output.jpg_quality)
    
    # Create viewer
    viewer_dir = create_viewer_folder(pano_path, config.output_dir, pano_filename)
    
    # ============================================================
    # Summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PANORAMA STITCHING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Input images: {len(image_infos)}")
    logger.info(f"Panorama size: {config.output.pano_width}x{config.output.pano_height}")
    logger.info(f"Coverage: {coverage['coverage_percent']:.1f}%")
    logger.info(f"Output: {pano_path}")
    logger.info(f"Viewer: {viewer_dir / 'index.html'}")
    logger.info("")
    logger.info("To view the panorama, open the following file in a browser:")
    logger.info(f"  file://{viewer_dir.absolute() / 'index.html'}")
    logger.info("=" * 60)
    
    return pano_path

