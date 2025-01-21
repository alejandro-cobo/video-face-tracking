import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.face_tracker import FaceTracker
from src.path import find
from src.video import VIDEO_FORMATS
sys.path.remove(ROOT_PATH)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Video face tracking tool to annotate video datasets.")
    parser.add_argument(
        'filename',
        type=str,
        help='Path to a video file or a directory. If it is a directory, all videos inside the directory all '
             'proccessed. If the --recursive flag is provided, all subdirectories are recursively traversed and '
             'proccessed too.'
    )
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        help='Root directory to save the annotation files. By default, they are saved in the same location as the '
             'input file.'
    )
    parser.add_argument(
        '--max-frames', '-f',
        type=int,
        help='Max number of frames to process for each video. By default, all frames are proccessed.'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='When the input filename is a directory, also process recursively all subdirectories inside.'
    )
    parser.add_argument('--quiet', '--silent', '-q', action='store_true', help='Hide progress bars.')
    args = parser.parse_args(argv)
    return args


def process_file(video_path: Path, out_dir: Path | None, face_tracker: FaceTracker) -> None:
    if video_path.suffix not in VIDEO_FORMATS:
        raise ValueError(f'video file must be a valid video file: {video_path} ({VIDEO_FORMATS})')

    if out_dir is None:
        out_dir = video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    faces = face_tracker(str(video_path))

    out_path = out_dir / f'{video_path.stem}.json'
    with open(out_path, 'w') as out_file:
        json.dump(faces, out_file)
    tqdm.write(f'Saved annotations file to {out_path}', file=sys.stdout)


def process_dir(
    input_path: Path,
    out_dir: Path | None,
    face_tracker: FaceTracker,
    recursive: bool,
    quiet: bool
) -> None:
    video_files = find(input_path, VIDEO_FORMATS, recursive)
    for video_path in tqdm(video_files, desc='Processing files', leave=False, disable=quiet):
        ann_out_dir = None
        if out_dir is not None:
            rel_path = video_path.parent.relative_to(input_path)
            ann_out_dir = out_dir / rel_path
        process_file(video_path, ann_out_dir, face_tracker)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filename = Path(args.filename)
    prefix = None if args.prefix is None else Path(args.prefix)
    max_frames = args.max_frames
    recursive = args.recursive
    quiet = args.quiet

    face_tracker = FaceTracker(max_frames=max_frames, quiet=quiet)
    if filename.is_file() and filename.suffix in VIDEO_FORMATS:
        process_file(video_path=filename, out_dir=prefix, face_tracker=face_tracker)
    elif filename.is_dir():
        process_dir(
            input_path=filename,
            out_dir=prefix,
            face_tracker=face_tracker,
            recursive=recursive,
            quiet=quiet
        )
    else:
        raise FileNotFoundError(f'Filename {filename} is not a valid file or directory.')


if __name__ == '__main__':
    main(sys.argv[1:])

