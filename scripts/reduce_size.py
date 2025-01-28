import argparse
from functools import partial
import json
from pathlib import Path
import sys

from tqdm import tqdm

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.path import find
sys.path.remove(ROOT_PATH)

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Tool to reduce the size of JSON annotations by rounding floating point numbers.")
    parser.add_argument('filenames', type=str, nargs='+', help='Path(s) to a JSON file or directory.')
    parser.add_argument(
        '--precision', '-p',
        type=int,
        default=2,
        help='Number of decimal positions to keep. Default: 2.'
    )
    parser.add_argument(
        '--ignore', '-i',
        type=str,
        choices=['bbox', 'prob', 'landmarks'],
        nargs='+',
        help='Ignore one or more annotation keys.'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='When the input filename is a directory, also process recursively all subdirectories inside.'
    )
    parser.add_argument('--quiet', '--silent', '-q', action='store_true', help='Hide progress bars.')
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

    round_func = int if precision == 0 else partial(round, ndigits=precision)
    for face_anns in data.values():
        for frame_anns in face_anns.values():
            if ignore is None or 'bbox' not in ignore:
                frame_anns['bbox'] = [round_func(x) for x in frame_anns['bbox']]
            if ignore is None or 'prob' not in ignore:
                frame_anns['prob'] = round_func(frame_anns['prob'])
            if ignore is None or 'landmarks' not in ignore:
                frame_anns['landmarks'] = [round_func(x) for x in frame_anns['landmarks']]

    with open(input_path, 'w') as json_file:
        json.dump(data, json_file)
    diff = ori_size - input_path.stat().st_size
    tqdm.write(f'{input_path}: {num_to_str(diff)} deleted', file=sys.stdout)
    return diff


def process_dir(input_path: Path, precision: int, ignore: list[str] | None, recursive: bool, quiet: bool) -> None:
    total_size = 0
    for file in tqdm(
        find(input_path, '.json', recursive),
        desc='Processing directory',
        leave=False,
        disable=quiet,
        dynamic_ncols=True
    ):
        total_size += process_file(file, precision, ignore)
    tqdm.write(f'Total: {num_to_str(total_size)} deleted', file=sys.stdout)


def main(argv: list[str]) -> None:
    args = parse_args(argv)

    filenames = args.filenames
    precision = args.precision
    ignore = args.ignore
    recursive = args.recursive
    quiet = args.quiet

    disable = quiet or len(filenames) == 1
    for filename in tqdm(filenames, desc='Processing input files', leave=False, disable=disable, dynamic_ncols=True):
        filename = Path(filename)
        if filename.is_file():
            process_file(input_path=filename, precision=precision, ignore=ignore)
        elif filename.is_dir():
            process_dir(input_path=filename, precision=precision, ignore=ignore, recursive=recursive, quiet=quiet)
        else:
            tqdm.write(f'reduce_size.py: WARNING: file {filename} does not exist.')


if __name__ == '__main__':
    main(sys.argv[1:])

