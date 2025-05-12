#!/usr/bin/env python3
import os
import argparse
from mesh_estimator import HumanMeshEstimator
from video_process import extract_frames


def make_parser():
    parser = argparse.ArgumentParser(
        description='Extract 3D joint trajectories and SMPL parameters'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--images',
        type=str,
        help='Path to folder of pre-extracted image frames'
    )
    group.add_argument(
        '--video',
        type=str,
        help='Path to input video (frames will be extracted first)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='frames',
        help='Directory for extracted frames (used with --video)'
    )
    parser.add_argument(
        '--separate',
        action='store_true',
        help='Save per-joint instead of per-frame'
    )
    parser.add_argument(
        '--json_name',
        type=str,
        default='trajectories.json',
        help='Output JSON filename'
    )
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    # list of valid image extensions (lowercase)
    exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

    # ensure we have frames to process
    if args.video:
        images_dir = extract_frames(args.video, args.output_dir)
        if not os.path.isdir(images_dir):
            parser.error(f"Failed to create frames folder: {images_dir}")
        # check for at least one frame with a valid extension
        files = os.listdir(images_dir)
        if not any(f.lower().endswith(ext) for f in files for ext in exts):
            parser.error(f"No frames with extensions {exts} found in {images_dir} after extraction.")
    else:
        images_dir = args.images
        if not os.path.isdir(images_dir):
            parser.error(f"Images folder not found: {images_dir}")
        files = os.listdir(images_dir)
        if not any(f.lower().endswith(ext) for f in files for ext in exts):
            parser.error(f"No images with extensions {exts} found in {images_dir}.")

    estimator = HumanMeshEstimator()
    traj, params, traj_camera = estimator.run_on_video_frames(images_dir)

    # Make sure our outputs directory exists
    out_dir = "/env/outputs"
    os.makedirs(out_dir, exist_ok=True)
    # Build full output path
    mode = 'per-joint' if args.separate else 'per-frame'
    output_path_local = os.path.join(out_dir, mode +"_local_"+ args.json_name)
    output_path_camera = os.path.join(out_dir, mode +"_camera_"+ args.json_name)

    estimator.save_trajectories_json(
        trajectories=traj,
        params_list=params,
        filename=output_path_local,
        separate_per_joint=args.separate
    )
    
    estimator.save_trajectories_json(
        trajectories=traj_camera,
        params_list=params,
        filename=(output_path_camera),
        separate_per_joint=args.separate
    )
    
    print(f"Saved {mode} trajectories & parameters in local space for {traj.shape[0]} frames to {output_path_local}")
    print(f"Saved {mode} trajectories & parameters in camera space for {traj.shape[0]} frames to {output_path_camera}")


if __name__ == '__main__':
    main()
