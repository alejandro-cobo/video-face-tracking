import argparse
import json
from pathlib import Path
import sys


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Tool to reduce the size of JSON annotations by rounding floating point numbers.")
    parser.add_argument('filename', type=str, help='Path to a JSON file or directory.')
    parser.add_argument(
        '--precision', '-p',
        type=int,
        default=2,
        help='Number of decimal positions to keep. Default: 2.'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='When the input filename is a directory, also process recursively all subdirectories inside.'
    )
    parser.add_argument(
        '--ignore', '-i',
        type=str,
        choices=['bbox', 'prob', 'landmarks'],
        nargs='+',
        help='Ignore one or more annotation keys.'
    )
    args = parser.parse_args(argv)
    return args


def num_to_str(num: int, fmt: str = '%d') -> str:
    for metric in ('B', 'KB', 'MB', 'GB'):
        if num < 1000:
            return fmt % num + metric
        num //= 1000
    return fmt % (num * 1000) + metric


def process_file(input_path: Path, precision: int, ignore: list[str] | None = None) -> int:
    ori_size = input_path.stat().st_size
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)

    for face_anns in data.values():
        for frame_anns in face_anns.values():
            if ignore is None or 'bbox' not in ignore:
                frame_anns['bbox'] = [round(x, precision) for x in frame_anns['bbox']]
            if ignore is None or 'prob' not in ignore:
                frame_anns['prob'] = round(frame_anns['prob'], precision)
            if ignore is None or 'landmarks' not in ignore:
                frame_anns['landmarks'] = [round(x, precision) for x in frame_anns['landmarks']]

    with open(input_path, 'w') as json_file:
        json.dump(data, json_file)
    diff = ori_size - input_path.stat().st_size
    print(f'{input_path}: {num_to_str(diff)} deleted')
    return diff


def process_dir(input_path: Path, precision: int, recursive: bool, ignore: list[str] | None = None) -> None:
    total_size = 0
    if recursive:
        for root, _, files in input_path.walk():
            for file in files:
                if file.suffix == '.json':
                    total_size += process_file(root / file, precision, ignore)
    else:
        for file in input_path.iterdir():
            if file.suffix == '.json':
                total_size += process_file(file, precision, ignore)
    print(f'Total: {num_to_str(total_size)} deleted')


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filename = Path(args.filename)
    precision = args.precision
    ignore = args.ignore
    recursive = args.recursive

    if filename.is_file():
        process_file(filename, precision, ignore)
    elif filename.is_dir():
        process_dir(filename, precision, recursive, ignore)
    else:
        raise FileNotFoundError(f'Filename {filename} is not a file or directory.')


if __name__ == '__main__':
    main(sys.argv[1:])

