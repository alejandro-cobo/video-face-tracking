from pathlib import Path
from typing import Sequence


def find(path: Path, pattern: str | Sequence[str] | None = None, recursive: bool = False) -> list[Path]:
    if isinstance(pattern, str):
        pattern = [pattern]

    file_list = []
    dir_list = [path]
    for dir in dir_list:
        for file in dir.iterdir():
            if file.is_dir() and recursive:
                dir_list.append(file)
            if pattern is not None and file.suffix not in pattern:
                continue
            file_list.append(file)

    return file_list

