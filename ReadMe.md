
<div align="center">

## LAR IBRICS FORK of: ##

# **CameraHMR: Aligning People with Perspective (3DV 2025)**  

##### Original CameraHMR authors and pages

[**Priyanka Patel**](https://pixelite1201.github.io/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [**Michael J. Black**](https://ps.is.mpg.de/person/black)


🌐 [**Project Page**](https://camerahmr.is.tue.mpg.de) | 📄 [**ArXiv Paper**](https://arxiv.org/abs/2411.08128) | 🎥 [**Video Results**](https://youtu.be/aDmfAxYLV2w) | 🌐 [**GitHub Repo**](https://github.com/pixelite1201/CameraHMR/)

---

![](teaser/teaser.png)  
*Figure: CameraHMR Results*

</div>

---


## 🎬 **Demo**

Best way to use it is to use through https://github.com/upatras-lar/CameraHMR_environment/ that sets up a docker container with everything required to run in different modes


In case one wants to install without the above environment then use:

1. Create a conda environment and install all the requirements.

```bash
conda create -n camerahmr python=3.10
conda activate camerahmr
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

2. Download necessary files using:

Yolov8weights and Verify
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
md5sum yolov8s.pt
# 0a1d5d0619d5a24848a267ec1e3a3118  yolov8s.pt
```
 
3. Download models provided by camerahmr team

```bash
bash fetch_demo_data.sh
```

Alternatively, download files manually from the [CameraHMR website](https://camerahmr.is.tue.mpg.de). Ensure to update paths in [`constants.py`](core/constants.py) if doing so manually.


Run the demo with following command. It will run demo on all images in the specified --image_folder, and save renderings of the reconstructions and the output mesh in --out_folder. 

```
python demo.py --image_folder demo_images --output_folder output_images
```

## IBRICS scripts

#### 1. video_process.py

  

A simple utility that dumps all frames from a given video into a specified folder using ffmpeg for efficienty and speed.

  

##### Arguments

| Option | Description | Default |
|---|---|---|
| `video` | Path to the input video file (e.g. `input.mp4`). | (required) |
| `-o`, `--output-dir` | Directory to save the PNG frames. | `frames/` |

  

##### Example

  

```bash

python3 extract_frames.py my_video.mp4 -o extracted_frames

```

---
  

#### 2. ibrics_main.py

  

Runs the `HumanMeshEstimator` on either a folder of pre‑extracted images or on a video (frames will be extracted first) and outputs a json file with either per-frame or per-joint structure.

  

##### Arguments

  

| Option | Description | Default |
|---|---|---|
| `--images <path>` | Path to a folder of pre‑extracted image frames (mutually exclusive with `--video`). | (optional except if --video is missing) |
| `--video <path>` | Path to an input video file (frames will be extracted first). | (optional except if --images is missing) |
| `-o`, `--output-dir` | Directory to save extracted frames when using `--video`. | `frames/` |
| `--separate` | Flag to save the JSON file **per joint** instead of one **per frame**. | False |
| `--json_name <name>` | Base filename for output JSON (will be prefixed by `per-frame_` or `per-joint_`). | `/env/outputs/per-{frame or joint}_local_trajectories.json` and `/env/outputs/per-{frame or joint}_camera_trajectories.json`|

  

> **Note**: You must supply exactly one of `--images` or `--video`.

  

##### Example: From Video

  

```bash

python3 ibrics_main.py --video my_video.mp4 -o extracted_frames --json_name results.json

# Extracts frames to ./extracted_frames/

# Runs mesh estimation on each frame

# Outputs SMPL trajectories to ./outputs/per-frame_results.json

```

  

##### Example: From Images

  

```bash

python3 ibrics_main.py --images extracted_frames --separate

# Reads frames from ./extracted_frames/

# Outputs the JSON per joint in ./outputs/per-joint_trajectories.json

```

### Json Format

Save joint trajectories + SMPL parameters to JSON, with nested structure and shape info.

- per-frame mode:

```json
{
    "frames": [
    { "trajectory": [[x,y,z],…], "parameters": {…} },
    … (one entry per frame)
    ],
    "average_parameters": {…},
    "shapes": {
    "num_frames": F,
    "joints_per_frame": J,
    "parameter_frames": F
    }
}
```

- per-joint mode:

```json
{
    "trajectories": {
    "joint_0": [[…],[…],…],
    … (one entry per joint)
    },
    "parameters": {
    "frame_0": {…},
    "frame_1": {…},
    … (one entry per frame)
    },
    "average_parameters": {…},
    "shapes": {
    "num_joints": J,
    "frames_per_joint": F,
    "parameter_frames": F
    }
}
```

## 📚 **Citation**

If you find the *original* **CameraHMR** useful in your work, please cite:

```bibtex
@inproceedings{patel2024camerahmr,
title={{CameraHMR}: Aligning People with Perspective},
author={Patel, Priyanka and Black, Michael J.},
booktitle={International Conference on 3D Vision (3DV)},
year={2025} }
```

