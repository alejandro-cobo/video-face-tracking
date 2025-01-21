import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from tqdm import tqdm

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.image import align_bbox, crop_image, expand_bbox, resize_image
from src.video import VIDEO_FORMATS, Video
sys.path.remove(ROOT_PATH)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Script to crop annotated faces from video files.")
    parser.add_argument(
        'filename',
        type=str,
        help='Path to a video file or a directory. If it is a directory, all videos inside the directory all '
             'proccessed. If the --recursive flag is provided, all subdirectories are recursively traversed and '
             'proccessed too. JSON annotations must have the same filename as the video (e.g., '
             'mydir/video.mp4 and mydir/video.json).'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        help='Root directory to save the cropped faces. By default, they are saved in the same location as the input '
             'file.'
    )
    parser.add_argument(
        '--crop-size', '-c',
        type=int,
        help='Size of the saved image crops, in pixels. Ignore to skip the resize step and save each crop with its '
             'original size.'
    )
    parser.add_argument(
        '--bbox-scale', '-b',
        type=float,
        default=1.3,
        help='Factor to use to increase the bounding box size. Default: 1.3 (increases the size by 30 percent).'
    )
    parser.add_argument(
        '--align', '-a',
        action='store_true',
        help='Align faces to match the center of the bounding box to the position of the nose landmark.'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='When the input filename is a directory, also process recursively all subdirectories inside.'
    )
    parser.add_argument('--quiet', '--silent', '-q', action='store_true', help='Hide progress bars.')
    args = parser.parse_args(argv)
    return args



def process_file(
    video_path: Path,
    out_dir: Path | None,
    crop_size: int | None,
    bbox_scale: float,
    align: bool
) -> None:
    if video_path.suffix not in VIDEO_FORMATS:
        raise ValueError(f'Input file must be a valid video file: {video_path} ({VIDEO_FORMATS})')

    ann_path = video_path.with_suffix('.json')
    if not ann_path.exists():
        raise FileNotFoundError(f'Annotation path not found {ann_path}')

    if out_dir is None:
        out_dir = video_path.with_suffix('')
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(ann_path, 'r') as ann_file:
        anns = json.load(ann_file)

    with Video(str(video_path)) as video_file:
        for frame_idx in range(video_file.num_frames):
            frame = video_file.read()
            frame_idx_str = str(frame_idx)
            for face_idx, face_anns in anns.items():
                if frame_idx_str in face_anns:
                    frame_anns = face_anns[frame_idx_str]
                    bbox = np.array(frame_anns['bbox'])
                    if align:
                        new_center = (frame_anns['landmarks'][4], frame_anns['landmarks'][5])
                        bbox = align_bbox(bbox, new_center)
                    bbox = expand_bbox(bbox, bbox_scale)
                    crop = crop_image(frame, bbox)
                    if crop_size is not None:
                        crop = resize_image(crop, crop_size)
                    crop_path = out_dir / f'{frame_idx:05d}_{int(face_idx):03d}.png'
                    cv2.imwrite(str(crop_path), crop)

    tqdm.write(f'Saved cropped images to {out_dir}', file=sys.stdout)


def process_dir(
    input_path: Path,
    out_dir: Path | None,
    crop_size: int | None,
    bbox_scale: float,
    align: bool,
    recursive: bool,
    quiet: bool
) -> None:
    def test_valid_file(file_path: Path) -> bool:
        ann_path = file_path.with_suffix('.json')
        return file_path.suffix in VIDEO_FORMATS and ann_path.exists()

    video_files = []
    if recursive:
        for root, _, files in input_path.walk():
            for file in files:
                file = root / file
                if test_valid_file(file):
                    video_files.append(file)
    else:
        for file in input_path.iterdir():
            if test_valid_file(file):
                video_files.append(file)

    for video_path in tqdm(video_files, desc='Processing files', leave=False, disable=quiet):
        ann_out_dir = None
        if out_dir is not None:
            rel_path = video_path.with_suffix('').relative_to(input_path)
            ann_out_dir = out_dir / rel_path
        process_file(video_path, ann_out_dir, crop_size, bbox_scale, align)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filename = Path(args.filename)
    prefix = None if args.prefix is None else Path(args.prefix)
    crop_size = args.crop_size
    bbox_scale = args.bbox_scale
    align = args.align
    recursive = args.recursive
    quiet = args.quiet

    if filename.is_file() and filename.suffix in VIDEO_FORMATS:
        process_file(
            video_path=filename,
            out_dir=prefix,
            crop_size=crop_size,
            bbox_scale=bbox_scale,
            align=align
        )
    elif filename.is_dir():
        process_dir(
            input_path=filename,
            out_dir=prefix,
            crop_size=crop_size,
            bbox_scale=bbox_scale,
            align=align,
            recursive=recursive,
            quiet=quiet
        )
    else:
        raise FileNotFoundError(f'Filename {filename} is not a valid file or directory.')


if __name__ == '__main__':
    main(sys.argv[1:])

