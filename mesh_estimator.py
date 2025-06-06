import cv2
import os
import json
import torch
import smplx
import trimesh
import numpy as np
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from core.utils.utils_detectron2 import DefaultPredictor_Lazy
from core.camerahmr_model import CameraHMR
from core.constants import (
    CHECKPOINT_PATH,
    CAM_MODEL_CKPT,
    SMPL_MODEL_PATH,
    DETECTRON_CKPT,
    DETECTRON_CFG,
)
from ultralytics import YOLO

from core.datasets.dataset import Dataset
from core.utils.renderer_pyrd import Renderer
from core.utils import recursive_to
from core.cam_model.fl_net import FLNet
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, NUM_BETAS
import argparse
import time


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y : start_y + new_height, start_x : start_x + new_width] = (
        resized_img
    )

    return aspect_ratio, final_img


class HumanMeshEstimator:
    def __init__(self, smpl_model_path=SMPL_MODEL_PATH, threshold=0.25):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.init_model().to(self.device).eval()

        self.detection = "YOLO"
        self.detector = YOLO("data/pretrained-models/yolov8s.pt")  # nano version for max FPS
        # self.detection = "Detectron"
        # self.detector = self.init_detector(threshold)

        self.cam_model = self.init_cam_model().eval()
        self.smpl_model = smplx.SMPLLayer(
            model_path=smpl_model_path, num_betas=NUM_BETAS
        ).to(self.device)
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)

    def init_cam_model(self):
        flnet = FLNet()
        ckpt = torch.load(CAM_MODEL_CKPT)["state_dict"]
        flnet.load_state_dict(ckpt)
        return flnet

    def init_model(self):
        return CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False)

    def init_detector(self, threshold):

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = (
                threshold
            )
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    def convert_to_full_img_cam(
        self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length
    ):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2.0 * focal_length / (bbox_height * s)
        cx = 2.0 * (bbox_center[:, 0] - (img_w / 2.0)) / (s * bbox_height)
        cy = 2.0 * (bbox_center[:, 1] - (img_h / 2.0)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):
        smpl_output = self.smpl_model(**{k: v.float() for k, v in params.items()})
        # print(self.smpl_model.J_regressor.shape)
        pred_keypoints_3d = smpl_output.joints
        pred_vertices = smpl_output.vertices
        img_h, img_w = batch["img_size"][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch["box_size"],
            bbox_center=batch["box_center"],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch["cam_int"][:, 0, 0],
        )
        return pred_vertices, pred_keypoints_3d, cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, IMAGE_SIZE)
        img_full_resized = (
            np.transpose(img_full_resized.astype("float32"), (2, 0, 1)) / 255.0
        )
        img_full_resized = self.normalize_img(
            torch.from_numpy(img_full_resized).float()
        )

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        # fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array(
            [[fl_h, 0, img_w / 2], [0, fl_h, img_h / 2], [0, 0, 1]]
        ).astype(np.float32)
        return cam_int

    def remove_pelvis_rotation(self, smpl):
        """We don't trust the body orientation coming out of bedlam_cliff, so we're just going to zero it out."""
        smpl.body_pose[0][0][:] = np.zeros(3)

    def process_image(self, img_path, output_img_folder, i):
        img_cv2 = cv2.imread(str(img_path))

        fname, img_ext = os.path.splitext(os.path.basename(img_path))
        overlay_fname = os.path.join(
            output_img_folder, f"{os.path.basename(fname)}_{i:06d}{img_ext}"
        )
        smpl_fname = os.path.join(
            output_img_folder, f"{os.path.basename(fname)}_{i:06d}.smpl"
        )
        mesh_fname = os.path.join(
            output_img_folder, f"{os.path.basename(fname)}_{i:06d}.obj"
        )


        # Added YOLO model for human detection
        start_time = time.time()
        # Detect humans in the image
        if self.detection == "Detectron":
            ### DETECTRON2

            det_out = self.detector(img_cv2)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
            bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        elif self.detection == "YOLO":
            # else:
            det_out = self.detector(img_cv2)
            # Get the bounding boxes of detected persons
            det_instances = det_out[0].boxes

            # Get the tensors for easier filtering
            boxes = det_instances.xyxy  # (N, 4) x1, y1, x2, y2
            scores = det_instances.conf  # (N,)
            classes = det_instances.cls  # (N,)

            # Filter: class == 0 (person) and THRESHOLD (CONFIDENCE) score > 0.5
            valid_idx = (classes == 0) & (scores > 0.5)
            boxes = boxes[valid_idx]

            # Now calculate bbox_center and bbox_scale like before
            boxes_np = boxes.cpu().numpy()
            bbox_scale = (boxes_np[:, 2:4] - boxes_np[:, 0:2]) / 200.0
            bbox_center = (boxes_np[:, 2:4] + boxes_np[:, 0:2]) / 2.0

        else:
            raise ValueError(f"Unknown detector: {self.detection}")

        total_time = time.time() - start_time
        print(f"Person detection time: {total_time: .2f}s")

        # Get Camera intrinsics using HumanFoV Model
        cam_int = self.get_cam_intrinsics(img_cv2)
        
        # Create the Torch dataset
        dataset = Dataset(img_cv2, bbox_center, bbox_scale, cam_int, False, img_path)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=10
        )


        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch["img_size"][0]
            
            # inference of the model
            with torch.no_grad():
                out_smpl_params, out_cam, focal_length_ = self.model(batch)

            # get the vertices and the joints of the smpl
            output_vertices, output_joints, output_cam_trans = self.get_output_mesh(
                out_smpl_params, out_cam, batch
            )

            # create the smpl mesh
            mesh = trimesh.Trimesh(
                output_vertices[0].cpu().numpy(), self.smpl_model.faces, process=False
            )
            mesh.export(mesh_fname)

            # Render overlay
            focal_length = (focal_length_[0], focal_length_[0])
            pred_vertices_array = (
                (output_vertices + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            )
            renderer = Renderer(
                focal_length=focal_length[0],
                img_w=img_w,
                img_h=img_h,
                faces=self.smpl_model.faces,
                same_mesh_color=True,
            )
            front_view = renderer.render_front_view(
                pred_vertices_array, bg_img_rgb=img_cv2.copy()
            )
            final_img = front_view
            # Write overlay
            cv2.imwrite(overlay_fname, final_img)
            renderer.delete()

    def run_on_images(self, image_folder, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        image_extensions = [
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.tiff",
            "*.webp",
        ]
        images_list = [
            image
            for ext in image_extensions
            for image in glob(os.path.join(image_folder, ext))
        ]
        for ind, img_path in enumerate(images_list):
            self.process_image(img_path, out_folder, ind)


    # ---------------------------------- IBRICS ---------------------------------- #
    def run_on_video_frames(self, image_folder):
        """
        Process all images in a folder (sorted), returning:
          - trajectories: (F, J, 3) numpy array of joint world positions
          - params_list:  list of dicts of SMPL & camera parameters per frame
        """
        
        # --------------------------- Get the video frames --------------------------- #
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]
        paths = []
        for e in exts:
            paths.extend(glob(os.path.join(image_folder, e)))
        paths = sorted(paths)
        traj = []
        traj_camera = []
        params_list = []
        
        # -------------------------------- Frames Loop ------------------------------- #
        for p in paths:
            img = cv2.imread(p)

            start_time = time.time()
            # Detect humans in the image
            if self.detection == "Detectron":
                ### DETECTRON2

                det_instances = self.detector(img)["instances"]
                valid_idx = (det_instances.pred_classes == 0) & (
                    det_instances.scores > 0.5
                )
                boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

                # if zero append 0
                if len(boxes) == 0:
                    traj.append(np.zeros((self.smpl_model.num_joints(), 3)))
                    traj_camera.append(np.zeros((self.smpl_model.num_joints(), 3)))
                    # append empty params
                    params_list.append(
                        {
                            k: np.zeros_like(v[0].cpu().numpy()).tolist()
                            for k, v in {}.items()
                        }
                    )
                    continue
                scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
                center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

            elif self.detection == "YOLO":
                # else:
                det_out = self.detector(img)
                # Get the bounding boxes of detected persons
                det_instances = det_out[0].boxes

                # Get the tensors for easier filtering
                boxes = det_instances.xyxy  # (N, 4) x1, y1, x2, y2
                scores = det_instances.conf  # (N,)
                classes = det_instances.cls  # (N,)

                # Filter: class == 0 (person) and THRESHOLD (CONFIDENCE) score > 0.5
                valid_idx = (classes == 0) & (scores > 0.5)
                boxes = boxes[valid_idx]


                # if zero append 0
                if len(boxes) == 0:
                    traj.append(np.zeros((self.smpl_model.num_joints(), 3)))
                    traj_camera.append(np.zeros((self.smpl_model.num_joints(), 3)))
                    # append empty params
                    params_list.append(
                        {
                            k: np.zeros_like(v[0].cpu().numpy()).tolist()
                            for k, v in {}.items()
                        }
                    )
                    continue

                
                boxes_np = boxes.cpu().numpy()
                scale = (boxes_np[:, 2:4] - boxes_np[:, 0:2]) / 200.0
                center = (boxes_np[:, 2:4] + boxes_np[:, 0:2]) / 2.0

            else:
                raise ValueError(f"Unknown detector: {self.detection}")

            total_time = time.time() - start_time
            print(f"Person detection time: {total_time: .2f}s")

            # Get camera intrinsics
            cam_int = self.get_cam_intrinsics(img)  # e.g. a (3×3) matrix or a vector

            # build the torch Dataloader
            ds = Dataset(img, center, scale, cam_int, False, None)
            dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
            
            # -------------------------------- Batch loop -------------------------------- #
            for b in dl:
                b = recursive_to(b, self.device)
                
                #inference of the model
                with torch.no_grad():
                    params, cam_pred, _ = self.model(b)
                    
                # collect parameters for json
                frame_params = {}
                for k, v in params.items():
                    frame_params[k] = v[0].cpu().numpy().tolist()
                frame_params["cam_pred"] = cam_pred[0].cpu().numpy().tolist()
                frame_params["focal_length"] = float(cam_int[0, 0])
                params_list.append(frame_params)
                
                # compute joints
                verts, joints_local, cam_t = self.get_output_mesh(params, cam_pred, b)

                # Determine how many body joints the model actually has:
                n_body = self.smpl_model.J_regressor.shape[0]  # → 24 for SMPL-H
                # joints_local = joints_local[:, :n_body, :]  # shape (1, 24, 3)
                
                # Save camera-frame and local-frame as trajectories
                joints_body_camera = (
                    joints_local[0].cpu().numpy() + cam_t[0].cpu().numpy()[None, :]
                )
                joints_body_local = joints_local[0].cpu().numpy()
                traj_camera.append(joints_body_camera)
                traj.append(joints_body_local)

        # return the smpl and camera parameters list and the trajectories in local and camera frames
        return np.stack(traj, 0), params_list, np.stack(traj_camera, 0)

    def save_trajectories_json(
        self,
        trajectories: np.ndarray,
        params_list: list,
        filename: str,
        separate_per_joint: bool = False,
    ):
        """
        Save joint trajectories + SMPL parameters to JSON, with nested structure and shape info.

        - per-frame mode:
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

        - per-joint mode:
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
        """
        data = {}
        F, J, _ = trajectories.shape

        if not separate_per_joint:
            # Per‐frame: a list of objects, each containing that frame's trajectory+params
            frames = []
            for i in range(F):
                frames.append(
                    {
                        "trajectory": trajectories[i].tolist(),
                        "parameters": params_list[i],
                    }
                )
            data["frames"] = frames
            shapes = {"num_frames": F, "joints_per_frame": J, "parameter_frames": F}
        else:
            # Per‐joint: group trajectories by joint, but keep params per frame
            trajs = {f"joint_{j}": trajectories[:, j, :].tolist() for j in range(J)}
            data["trajectories"] = trajs
            params_by_frame = {f"frame_{i}": params_list[i] for i in range(F)}
            data["parameters"] = params_by_frame
            shapes = {"num_joints": J, "frames_per_joint": F, "parameter_frames": F}

        # Compute average parameters across all frames
        avg_params = {}
        std_params = {}
        if params_list:
            for key in params_list[0].keys():
                # stack into shape (F, D…), where D… is parameter dimensionality
                arr = np.stack([np.array(p[key]) for p in params_list], axis=0)
                avg_params[key] = arr.mean(axis=0).tolist()
                std_params[key] = arr.std(axis=0).tolist()

        data["average_parameters"] = avg_params
        data["std_parameters"] = std_params

        # Add our shape info
        data["shapes"] = shapes

        # Write out
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
