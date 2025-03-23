#!/usr/bin/env python

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

from src.image import align_bbox, crop_image, expand_bbox, resize_image
from src.path import find
from src.video import VIDEO_FORMATS, Video


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Script to crop annotated faces from video files."
    )
    parser.add_argument(
        "filenames",
        type=str,
        nargs="+",
        help="Path(s) to a video file or a directory. If it is a directory, "
        "all videos inside the directory are proccessed. If the "
        "--recursive flag is provided, all subdirectories are recursively "
        "traversed and proccessed too. JSON annotations must have the "
        "same filename as the video (e.g., mydir/video.mp4 and "
        "mydir/video.json) unless --ann-path is set.",
    )
    parser.add_argument(
        "--prefix",
        "-p",
        type=str,
        help="Root directory to save the cropped faces. By default, they are "
        "saved in the same location as the input file.",
    )
    parser.add_argument(
        "--ann-path",
        "-a",
        type=str,
        help="Path to JSON annotations file or root directory. By default, "
        "the script searches for annotation files with the same name as the "
        "video files.",
    )
    parser.add_argument(
        "--crop-size",
        "-c",
        type=int,
        help="Size of the saved image crops, in pixels. Ignore to skip the "
        "resize step and save each crop with its original size.",
    )
    parser.add_argument(
        "--bbox-scale",
        "-b",
        type=float,
        default=1.3,
        help="Factor to use to increase the bounding box size. Default: 1.3 "
        "(increases the size by 30 percent).",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Align faces to match the center of the bounding box to the "
        "position of the nose landmark.",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="When the input filename is a directory, also process recursively "
        "all subdirectories inside.",
    )
    parser.add_argument(
        "--quiet",
        "--silent",
        "-q",
        action="store_true",
        help="Hide progress bars.",
    )
    args = parser.parse_args(argv)
    return args


def process_file(
    video_path: Path,
    ann_path: Path | None,
    out_dir: Path | None,
    crop_size: int | None,
    bbox_scale: float,
    align: bool,
) -> None:
    if video_path.suffix not in VIDEO_FORMATS:
        raise ValueError(
            "Input file must be a valid video file: "
            f"{video_path} ({VIDEO_FORMATS})"
        )

    if ann_path is None:
        ann_path = video_path.with_suffix(".json")
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation path not found {ann_path}")

    if out_dir is None:
        out_dir = video_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path, "r") as ann_file:
        anns = json.load(ann_file)

    with Video(str(video_path)) as video_file:
        for frame_idx in range(video_file.num_frames):
            frame = video_file.read()
            frame_idx_str = str(frame_idx)
            for face_idx, face_anns in anns.items():
                if frame_idx_str in face_anns:
                    frame_anns = face_anns[frame_idx_str]
                    bbox = np.array(frame_anns["bbox"])
                    if align:
                        new_center = (
                            frame_anns["landmarks"][4],
                            frame_anns["landmarks"][5],
                        )
                        bbox = align_bbox(bbox, new_center)
                    bbox = expand_bbox(bbox, bbox_scale)
                    crop = crop_image(frame, bbox)
                    if crop_size is not None:
                        crop = resize_image(crop, crop_size)

                    face_dir = out_dir / face_idx
                    face_dir.mkdir(exist_ok=True)
                    crop_path = face_dir / f"{frame_idx:06d}.png"
                    cv2.imwrite(str(crop_path), crop)

    tqdm.write(f"Saved cropped images to {out_dir}", file=sys.stdout)


def process_dir(
    input_path: Path,
    ann_path: Path | None,
    out_dir: Path | None,
    crop_size: int | None,
    bbox_scale: float,
    align: bool,
    recursive: bool,
    quiet: bool,
) -> None:
    video_files = find(input_path, VIDEO_FORMATS, recursive)
    for video_path in tqdm(
        video_files,
        desc="Processing directory",
        leave=False,
        disable=quiet,
        dynamic_ncols=True,
    ):
        crop_ann_path = video_path.with_suffix(".json")
        if ann_path is not None:
            rel_path = crop_ann_path.relative_to(input_path)
            crop_ann_path = ann_path / rel_path
        if not crop_ann_path.exists():
            continue

        crop_out_dir = None
        if out_dir is not None:
            rel_path = video_path.with_suffix("").relative_to(input_path)
            crop_out_dir = out_dir / rel_path

        process_file(
            video_path=video_path,
            ann_path=crop_ann_path,
            out_dir=crop_out_dir,
            crop_size=crop_size,
            bbox_scale=bbox_scale,
            align=align,
        )


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filenames = args.filenames
    prefix = None if args.prefix is None else Path(args.prefix)
    ann_path = None if args.ann_path is None else Path(args.ann_path)
    crop_size = args.crop_size
    bbox_scale = args.bbox_scale
    align = args.align
    recursive = args.recursive
    quiet = args.quiet

    disable = quiet or len(filenames) == 1
    for filename in tqdm(
        filenames,
        desc="Processing input files",
        leave=False,
        disable=disable,
        dynamic_ncols=True,
    ):
        filename = Path(filename)
        if filename.is_file():
            process_file(
                video_path=filename,
                ann_path=ann_path,
                out_dir=prefix,
                crop_size=crop_size,
                bbox_scale=bbox_scale,
                align=align,
            )
        elif filename.is_dir():
            process_dir(
                input_path=filename,
                ann_path=ann_path,
                out_dir=prefix,
                crop_size=crop_size,
                bbox_scale=bbox_scale,
                align=align,
                recursive=recursive,
                quiet=quiet,
            )
        else:
            tqdm.write(
                f"crop_faces.py: WARNING: file {filename} does not exist."
            )


if __name__ == "__main__":
    main(sys.argv[1:])
