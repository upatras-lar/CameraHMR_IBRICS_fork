
<div align="center">

## LAR IBRICS FORK of: ##

# **CameraHMR: Aligning People with Perspective (3DV 2025)**  

##### Original CameraHMR authors

[**Priyanka Patel**](https://pixelite1201.github.io/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; [**Michael J. Black**](https://ps.is.mpg.de/person/black)

---

🌐 [**Project Page**](https://camerahmr.is.tue.mpg.de) | 📄 [**ArXiv Paper**](https://arxiv.org/abs/2411.08128) | 🎥 [**Video Results**](https://youtu.be/aDmfAxYLV2w) | 🌐 [**GitHub Repo**](https://github.com/pixelite1201/CameraHMR/)

---

![](teaser/teaser.png)  
*Figure: CameraHMR Results*

</div>

---


## 🎬 **Demo**

###  **Download Required Files**

Download necessary demo files using:

Yolov8weights and Verify
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
md5sum yolov8s.pt
# 0a1d5d0619d5a24848a267ec1e3a3118  yolov8s.pt
```

 
 
##### Download models provided by camerahmr team

```bash
bash fetch_demo_data.sh
```

Alternatively, download files manually from the [CameraHMR website](https://camerahmr.is.tue.mpg.de). Ensure to update paths in [`constants.py`](core/constants.py) if doing so manually.


Run the demo with following command. It will run demo on all images in the specified --image_folder, and save renderings of the reconstructions and the output mesh in --out_folder. 

```
python demo.py --image_folder demo_images --output_folder output_images
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

