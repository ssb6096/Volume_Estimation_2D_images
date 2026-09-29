# Volume Estimation of Food from 2D Images

Estimating the volume of food items from ordinary monocular (2D) photos and video. The project compares classical computer vision approaches with deep learning approaches, and was a collaboration with the University of Rochester.

[![Watch the demo on YouTube](https://img.youtube.com/vi/0JI2T99j86U/hqdefault.jpg)](https://www.youtube.com/watch?v=0JI2T99j86U)

▶️ **[Watch the demo on YouTube](https://www.youtube.com/watch?v=0JI2T99j86U)** · 📑 [Project poster (PDF)](docs/Food_Volume_Estimation_Poster.pdf)

---

## Overview

Several approaches were explored and compared:

**Computer vision techniques**
- **Contour detection and bird's-eye view.** A calibrated camera and a perspective warp to a top-down view.
- **Frame transforms.** ArUco markers placed next to the food give camera coordinates and a real-world scale.
- **Structure from Motion (SfM).** Visual SFM uses SIFT features, feature matching and multicore bundle adjustment to reconstruct a 3D point cloud. **MeshLab** then converts the point cloud to a convex hull to compute the volume.

**Deep learning techniques**
- **Fully connected network (Neural Radiance Fields, NeRF).** Works well on synthetic data but takes a long time to render real images. LLFF is used to recover camera poses for NeRF.

**Conclusion.** Under the project's constraints, the computer vision techniques performed better than the deep learning techniques. Future work would use NeRF with good-quality real images to obtain 3D models, and train a GAN for 3D reconstruction of irregular objects.

## Skills and tools

`Python` · `OpenCV` · `Computer vision` · `Camera calibration` · `ArUco markers` · `Structure from Motion` · `MeshLab` · `NeRF` · `Deep learning`

## Repository contents

| Path | What it is |
|---|---|
| `1.camera_calibration-FrameTransform_BirdsEyeView/` | Camera calibration, bird's-eye view and ArUco frame-transform code, with sample data |
| `tiny_nerf.ipynb` | Google Colab NeRF notebook, adapted to run on custom food images (load images into Drive first) |
| `Instructions on video recording_snapping images.txt` | Guidelines for capturing photos and video for each method |
| `Sample video.mp4` | Example of how to record a video for Visual SFM |
| `docs/` | Project poster |

## Third-party tools

The pipeline also relies on these tools. They aren't included in this repo, so download them from the original sources:

| Tool | Used for | Original source |
|---|---|---|
| **VisualSFM** (Changchang Wu) | Structure from Motion: SIFT matching, bundle adjustment and dense reconstruction (with CMVS/PMVS) to get a 3D point cloud | [ccwu.me/vsfm](http://ccwu.me/vsfm/) (Windows, CUDA 64-bit build) |
| **NeRF** (Mildenhall et al., 2020) | Neural Radiance Fields for 3D reconstruction; `tiny_nerf.ipynb` here is adapted from its Colab notebook | [github.com/bmild/nerf](https://github.com/bmild/nerf) |
| **LLFF** (Mildenhall et al., 2019) | `imgs2poses.py` to recover camera poses from images before running NeRF | [github.com/Fyusion/LLFF](https://github.com/Fyusion/LLFF) |
| **MeshLab** | Converting the SfM point cloud to a convex hull and computing its volume | [meshlab.net](https://www.meshlab.net/) |

To reproduce the SfM and NeRF steps, download VisualSFM and clone NeRF and LLFF next to this repo, then follow the capture guidelines in `Instructions on video recording_snapping images.txt`.

## Context

Project from my M.S. in Electrical Engineering at Rochester Institute of Technology, in collaboration with the University of Rochester. Also on [Portfolium](https://portfolium.com/entry/volume-estimation-of-food-from-2-d-images).
