# 360° Spherical Panorama Stitching

Transform your phone photos into stunning 360° immersive panoramas with professional-grade quality. This project automatically stitches together a sequence of images captured while rotating your phone, creating seamless equirectangular panoramas that can be viewed in VR headsets, web browsers, or panoramic viewers.

![Panorama Example](showcase/panorama.jpg)

**🎬 Showcase Video:**

<video width="800" controls autoplay loop muted>
  <source src="showcase/GOAT.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video> 

## ✨ Features

- **🎯 Automatic Processing**: Simply provide a sequence of photos or a video, and the pipeline handles everything
- **📱 Phone Camera Optimized**: Works with any phone camera, automatically extracts camera parameters from EXIF data
- **🎬 Video Support**: Extract frames from videos automatically with intelligent keyframe selection
- **🔍 Robust Matching**: Advanced feature detection and matching handles challenging scenes with low contrast or repetitive patterns
- **🎨 Multiple Blending Modes**: Choose from sharp blending (minimal blur) or smooth blending (seamless transitions)
- **🌐 Interactive Viewer**: Automatically generates a web-based 360° viewer using Three.js
- **⚡ Memory Efficient**: Handles large panoramas (8K+) with optimized processing
- **🛠️ Fully Configurable**: Fine-tune every aspect through a simple YAML configuration file

## 🎨 What Makes This Special?

Unlike simple panorama apps, this tool:

- **Uses computer vision algorithms** (ORB features, RANSAC, homography estimation) for robust alignment
- **Extracts pure rotation** from homographies, ensuring geometrically correct panoramas
- **Handles failed matches intelligently** with neighbor interpolation and temporal smoothing
- **Supports high-resolution output** (up to 8K+ for professional use)
- **Optimized for video input** with frame extraction and motion-based keyframe selection

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Create a config file (see USAGE.md for details)
# Edit config.yaml with your settings

# Run the pipeline
python run.py config.yaml

# View your panorama
open output/viewer/index.html
```

For detailed usage instructions, see [USAGE.md](USAGE.md).

## 📊 Technical Highlights

This implementation features:

- **Pure rotation model** - Assumes camera rotates around its optical center (no parallax)
- **Spherical projection** - Maps images to equirectangular format using inverse mapping
- **Temporal smoothing** - Smooths camera rotation estimates for video sequences
- **Neighbor interpolation** - Handles failed feature matches intelligently
- **Adaptive RANSAC** - Adjusts threshold based on image resolution
- **Memory optimization** - Sequential processing for large inputs

For a deep technical dive, see [TECHNICAL.md](TECHNICAL.md).

## 📸 Sample Output

The generated panoramas can be:
- **Viewed in browsers** using the included HTML viewer
- **Uploaded to platforms** like YouTube (360°), Facebook, or Instagram
- **Imported into VR applications** for immersive experiences
- **Used in professional workflows** for architectural visualization or virtual tours

## 🎯 Perfect For

- **Photographers** creating immersive experiences
- **Real estate** virtual tours
- **Travel content** creators
- **Architectural visualization**
- **VR/AR developers** needing 360° content
- **Educators** teaching computer vision concepts

## 🔧 Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy, Pillow, and other dependencies (see `requirements.txt`)

## 📚 Documentation

- **[USAGE.md](USAGE.md)** - Complete usage guide and configuration reference
- **[TECHNICAL.md](TECHNICAL.md)** - Deep technical dive into algorithms and implementation

## 🤝 Contributing

Contributions are welcome! This is an open-source project built with passion for computer vision and panoramic imaging.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- OpenCV for computer vision algorithms
- Three.js for the 360° viewer framework
- Inspired by classic panorama stitching research and implementations

---

**Made with ❤️ for the panoramic imaging community**
