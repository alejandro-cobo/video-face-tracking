import argparse
import json
from pathlib import Path
import sys

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.path import find
sys.path.remove(ROOT_PATH)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Tool to remove face annotations with low number of frames.")
    parser.add_argument('filename', type=str, help='Path to a JSON file or directory.')
    parser.add_argument(
        '--min-frames', '-f',
        type=int,
        default=2,
        help='Minimum number of annotated frames to keep a face.'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='When the input filename is a directory, also process recursively all subdirectories inside.'
    )
    args = parser.parse_args(argv)
    return args


def process_file(input_path: Path, min_frames: int) -> None:
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)

    new_data = {}
    for face_id, face_anns in data.items():
        if len(face_anns) < min_frames:
            print(f'Removed face {face_id} from {input_path} with {len(face_anns)} frames')
            continue
        new_data[face_id] = face_anns

    with open(input_path, 'w') as json_file:
        json.dump(new_data, json_file)


def process_dir(input_path: Path, min_frames: int, recursive: bool) -> None:
    for file in find(input_path, '.json', recursive):
        process_file(file, min_frames)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filename = Path(args.filename)
    min_frames = args.min_frames
    recursive = args.recursive

    if filename.is_file():
        process_file(filename, min_frames)
    elif filename.is_dir():
        process_dir(filename, min_frames, recursive)
    else:
        raise FileNotFoundError(f'Filename {filename} is not a file or directory.')


if __name__ == '__main__':
    main(sys.argv[1:])

