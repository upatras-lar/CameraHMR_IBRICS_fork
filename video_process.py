#!/usr/bin/env python3
import ffmpeg
import os
import argparse

def extract_frames(video_path: str, output_dir: str = "frames") -> None:
    os.makedirs(output_dir, exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(os.path.join(output_dir, "frame_%05d.png"), start_number=0)
        .run(capture_stdout=True, capture_stderr=True)
    )
    print(f"Extracted frames to '{output_dir}/'")
    return output_dir

def run():
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