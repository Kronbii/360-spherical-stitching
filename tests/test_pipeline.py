"""
Integration tests for src/pipeline.py - End-to-end pipeline testing.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import (
    BlendingConfig,
    DebugConfig,
    IntrinsicsConfig,
    MatchingConfig,
    OutputConfig,
    PipelineConfig,
)
from src.pipeline import (
    PanoramaStitchingError,
    create_viewer_folder,
    generate_viewer_html,
    run_pipeline,
    setup_output_directory,
)


class TestSetupOutputDirectory:
    """Tests for setup_output_directory function."""
    
    def test_creates_output_dir(self, temp_dir):
        """Test creation of output directory."""
        config = PipelineConfig(
            input_dir=temp_dir,
            output_dir=temp_dir / "output"
        )
        
        setup_output_directory(config)
        
        assert config.output_dir.exists()
    
    def test_creates_debug_dirs(self, temp_dir):
        """Test creation of debug directories."""
        config = PipelineConfig(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            debug=DebugConfig(enabled=True)
        )
        
        setup_output_directory(config)
        
        assert (config.output_dir / "debug").exists()
        assert (config.output_dir / "debug" / "matches").exists()
        assert (config.output_dir / "debug" / "warped").exists()
    
    def test_no_debug_dirs_when_disabled(self, temp_dir):
        """Test no debug dirs when debug disabled."""
        config = PipelineConfig(
            input_dir=temp_dir,
            output_dir=temp_dir / "output",
            debug=DebugConfig(enabled=False)
        )
        
        setup_output_directory(config)
        
        assert not (config.output_dir / "debug").exists()


class TestGenerateViewerHtml:
    """Tests for generate_viewer_html function."""
    
    def test_creates_html_file(self, temp_dir):
        """Test HTML file creation."""
        viewer_dir = temp_dir / "viewer"
        viewer_dir.mkdir()
        
        generate_viewer_html(viewer_dir, "panorama.jpg")
        
        assert (viewer_dir / "index.html").exists()
    
    def test_html_contains_panorama_filename(self, temp_dir):
        """Test HTML references panorama filename."""
        viewer_dir = temp_dir / "viewer"
        viewer_dir.mkdir()
        
        generate_viewer_html(viewer_dir, "my_pano.jpg")
        
        html_content = (viewer_dir / "index.html").read_text()
        assert "my_pano.jpg" in html_content
    
    def test_html_contains_threejs(self, temp_dir):
        """Test HTML includes Three.js."""
        viewer_dir = temp_dir / "viewer"
        viewer_dir.mkdir()
        
        generate_viewer_html(viewer_dir, "panorama.jpg")
        
        html_content = (viewer_dir / "index.html").read_text()
        assert "three" in html_content.lower()


class TestCreateViewerFolder:
    """Tests for create_viewer_folder function."""
    
    def test_creates_viewer_directory(self, temp_dir, sample_image):
        """Test viewer directory creation."""
        # Create a panorama file
        pano_path = temp_dir / "panorama.jpg"
        cv2.imwrite(str(pano_path), sample_image)
        
        viewer_dir = create_viewer_folder(pano_path, temp_dir)
        
        assert viewer_dir.exists()
        assert (viewer_dir / "index.html").exists()
    
    def test_copies_panorama(self, temp_dir, sample_image):
        """Test panorama is copied to viewer folder."""
        pano_path = temp_dir / "panorama.jpg"
        cv2.imwrite(str(pano_path), sample_image)
        
        viewer_dir = create_viewer_folder(pano_path, temp_dir, "panorama.jpg")
        
        assert (viewer_dir / "panorama.jpg").exists()


class TestRunPipeline:
    """Integration tests for run_pipeline function."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_pipeline(self, test_images_dir, temp_dir):
        """Test complete pipeline execution."""
        config = PipelineConfig(
            input_dir=test_images_dir,
            output_dir=temp_dir / "output",
            matching=MatchingConfig(min_inliers=20),
            output=OutputConfig(pano_width=512),
        )
        
        pano_path = run_pipeline(config)
        
        assert pano_path.exists()
        assert (config.output_dir / "viewer" / "index.html").exists()
        assert (config.output_dir / "intrinsics.json").exists()
        assert (config.output_dir / "config.json").exists()
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_saves_config_json(self, test_images_dir, temp_dir):
        """Test that configuration is saved to JSON."""
        config = PipelineConfig(
            input_dir=test_images_dir,
            output_dir=temp_dir / "output",
            matching=MatchingConfig(min_inliers=20),
            output=OutputConfig(pano_width=512),
        )
        
        run_pipeline(config)
        
        config_path = config.output_dir / "config.json"
        assert config_path.exists()
        
        with open(config_path) as f:
            saved_config = json.load(f)
        
        assert "matching" in saved_config
        assert "output" in saved_config
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_saves_intrinsics_report(self, test_images_dir, temp_dir):
        """Test that intrinsics report is saved."""
        config = PipelineConfig(
            input_dir=test_images_dir,
            output_dir=temp_dir / "output",
            matching=MatchingConfig(min_inliers=20),
            output=OutputConfig(pano_width=512),
        )
        
        run_pipeline(config)
        
        intrinsics_path = config.output_dir / "intrinsics.json"
        assert intrinsics_path.exists()
        
        with open(intrinsics_path) as f:
            report = json.load(f)
        
        assert "intrinsics" in report
        assert "field_of_view" in report
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_debug_mode_saves_artifacts(self, test_images_dir, temp_dir):
        """Test debug mode saves intermediate artifacts."""
        config = PipelineConfig(
            input_dir=test_images_dir,
            output_dir=temp_dir / "output",
            matching=MatchingConfig(min_inliers=20),
            output=OutputConfig(pano_width=512),
            debug=DebugConfig(enabled=True, save_matches=True, save_warped_frames=2),
        )
        
        run_pipeline(config)
        
        # Check debug artifacts exist
        debug_dir = config.output_dir / "debug"
        assert debug_dir.exists()
        
        # Should have match visualizations
        matches_dir = debug_dir / "matches"
        assert matches_dir.exists()
        assert len(list(matches_dir.glob("*.jpg"))) > 0
        
        # Should have warped frame previews
        warped_dir = debug_dir / "warped"
        assert warped_dir.exists()
    
    def test_raises_on_insufficient_images(self, temp_dir):
        """Test error with less than 2 images."""
        # Create directory with single image
        images_dir = temp_dir / "images"
        images_dir.mkdir()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / "single.jpg"), img)
        
        config = PipelineConfig(
            input_dir=images_dir,
            output_dir=temp_dir / "output",
        )
        
        with pytest.raises(PanoramaStitchingError, match="at least 2"):
            run_pipeline(config)
    
    def test_raises_on_empty_directory(self, temp_dir):
        """Test error with empty input directory."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        config = PipelineConfig(
            input_dir=empty_dir,
            output_dir=temp_dir / "output",
        )
        
        with pytest.raises(ValueError, match="No image"):
            run_pipeline(config)
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_feather_blending(self, test_images_dir, temp_dir):
        """Test pipeline with feather blending."""
        config = PipelineConfig(
            input_dir=test_images_dir,
            output_dir=temp_dir / "output",
            matching=MatchingConfig(min_inliers=20),
            blending=BlendingConfig(method="feather"),
            output=OutputConfig(pano_width=512),
        )
        
        pano_path = run_pipeline(config)
        
        assert pano_path.exists()
        # Load and verify it's a valid image
        pano = cv2.imread(str(pano_path))
        assert pano is not None
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_png_output_format(self, test_images_dir, temp_dir):
        """Test PNG output format."""
        config = PipelineConfig(
            input_dir=test_images_dir,
            output_dir=temp_dir / "output",
            matching=MatchingConfig(min_inliers=20),
            output=OutputConfig(pano_width=512, output_format="png"),
        )
        
        pano_path = run_pipeline(config)
        
        assert pano_path.suffix == ".png"
        assert pano_path.exists()


class TestPanoramaStitchingError:
    """Tests for custom exception."""
    
    def test_exception_message(self):
        """Test exception carries message."""
        msg = "Test error message"
        error = PanoramaStitchingError(msg)
        
        assert str(error) == msg
    
    def test_can_be_raised(self):
        """Test exception can be raised and caught."""
        with pytest.raises(PanoramaStitchingError):
            raise PanoramaStitchingError("Test")

