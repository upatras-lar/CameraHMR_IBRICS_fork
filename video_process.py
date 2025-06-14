#!/usr/bin/env python3

# ---------------------------------- IBRICS ---------------------------------- #

import ffmpeg
import os
import argparse


# ------------------ ffmpeg frames extraction to output_dir ------------------ #
def extract_frames(video_path: str, output_dir: str = "frames") -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # ------------------- Run without warning and error logging ------------------ #
    
    # (
    #     ffmpeg
    #     .input(video_path)
    #     .output(os.path.join(output_dir, "frame_%05d.png"), start_number=0)
    #     .run(capture_stdout=True, capture_stderr=True)
    # )
    
    # -------------------------- Run with error logging -------------------------- #
    print(f"Running ffmpeg to extract frames:\n  ffmpeg -i {video_path} -start_number 0 {output_dir}/frame_%05d.png")
    ffmpeg.input(video_path) \
          .output(os.path.join(output_dir, "frame_%05d.png"), start_number=0) \
          .run()   # ← drop capture_stdout / capture_stderr
    
    print(f"Extracted frames to '{output_dir}/'")
    return output_dir

def run():
    # args: input video, output directory
    parser = argparse.ArgumentParser(
        description="Dump all frames from a video into an output directory."
    )
    parser.add_argument("video", help="Path to input video file (e.g. input.mp4)")
    parser.add_argument(
        "-o", "--output-dir",
        default="frames",
        help="Directory to save extracted frames (default: ./frames)"
    )
    args = parser.parse_args()

    extract_frames(args.video, args.output_dir)

if __name__ == "__main__":
    output_dir = run()