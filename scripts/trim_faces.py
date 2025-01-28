import argparse
import json
from pathlib import Path
import sys

from tqdm import tqdm

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.path import find
sys.path.remove(ROOT_PATH)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Tool to remove face annotations with low number of frames.")
    parser.add_argument('filenames', type=str, nargs='+', help='Path(s) to a JSON file or directory.')
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
    parser.add_argument('--quiet', '--silent', '-q', action='store_true', help='Hide progress bars.')
    args = parser.parse_args(argv)
    return args


def process_file(input_path: Path, min_frames: int) -> None:
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)

    new_data = {}
    for face_id, face_anns in data.items():
        if len(face_anns) < min_frames:
            tqdm.write(f'Removed face {face_id} from {input_path} with {len(face_anns)} frames', file=sys.stdout)
            continue
        new_data[face_id] = face_anns

    with open(input_path, 'w') as json_file:
        json.dump(new_data, json_file)


def process_dir(input_path: Path, min_frames: int, recursive: bool, quiet: bool) -> None:
    for file in tqdm(
        find(input_path, '.json', recursive),
        desc='Processing directory',
        leave=False,
        disable=quiet,
        dynamic_ncols=True
    ):
        process_file(file, min_frames)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filenames = args.filenames
    min_frames = args.min_frames
    recursive = args.recursive
    quiet = args.quiet

    disable = quiet or len(filenames) == 1
    for filename in tqdm(filenames, desc='Processing input files', leave=False, disable=disable, dynamic_ncols=True):
        filename = Path(filename)
        if filename.is_file():
            process_file(input_path=filename, min_frames=min_frames)
        elif filename.is_dir():
            process_dir(input_path=filename, min_frames=min_frames, recursive=recursive, quiet=quiet)
        else:
            tqdm.write(f'trim_faces.py: WARNING: file {filename} does not exist.')


if __name__ == '__main__':
    main(sys.argv[1:])

