import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.face_tracker import FaceTracker
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


def process_file(input_path: Path, face_tracker: FaceTracker) -> None:
    if input_path.suffix not in VIDEO_FORMATS:
        raise ValueError(f'Input file must be a valid video file: {input_path} ({VIDEO_FORMATS})')
    faces = face_tracker(str(input_path))
    out_path = input_path.with_suffix('.json')
    with open(out_path, 'w') as out_file:
        json.dump(faces, out_file)
    tqdm.write(f'Saved annotations file: {out_path}', file=sys.stdout)


def process_dir(input_path: Path, face_tracker: FaceTracker, recursive: bool, quiet: bool) -> None:
    video_files = []
    if recursive:
        for root, _, files in input_path.walk():
            for file in files:
                if file.suffix in VIDEO_FORMATS:
                    video_files.append(root / file)
    else:
        for file in input_path.iterdir():
            if file.suffix in VIDEO_FORMATS:
                video_files.append(file)

    for file in tqdm(video_files, desc='Processing files', leave=False, disable=quiet):
        process_file(file, face_tracker)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filename = Path(args.filename)
    max_frames = args.max_frames
    recursive = args.recursive
    quiet = args.quiet

    face_tracker = FaceTracker(max_frames=max_frames, quiet=quiet)
    if filename.is_file() and filename.suffix in VIDEO_FORMATS:
        process_file(input_path=filename, face_tracker=face_tracker)
    elif filename.is_dir():
        process_dir(
            input_path=filename,
            face_tracker=face_tracker,
            recursive=recursive,
            quiet=quiet
        )
    else:
        raise FileNotFoundError(f'Filename {filename} is not a valid file or directory.')


if __name__ == '__main__':
    main(sys.argv[1:])

