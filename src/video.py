from typing import Callable, Sequence

import cv2
from imutils.video import FileVideoStream
import numpy as np


__all__ = ['VIDEO_FORMATS', 'Video', 'play_video']

VIDEO_FORMATS = ('.mp4', '.mov', '.avi', '.wmv', '.webm', '.flv')


class Video(FileVideoStream):
    """ Simple context manager that wraps imutils.video.FileVideoStream and automatically calls stop() on exit """
    def __init__(
        self,
        path: str,
        transform: Callable | None = None,
        queue_size: int = 128,
        max_frames: int | None = None
    ) -> None:
        super().__init__(path, transform, queue_size)

        frame_count = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_frames = frame_count if max_frames is None else min(frame_count, max_frames)
        self.heigh =int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width =int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps =int(self.stream.get(cv2.CAP_PROP_FPS))

    def __enter__(self) -> "Video":
        self.start()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.stop()


def play_video(
    frames: Sequence[np.ndarray],
    fps: float = 30,
    win_name: str = 'Video',
    quit_btn: str | list[str] = 'q',
    next_btn: str | list[str] = 'n',
    pause_btn: str | list[str] = ' ',
    back_btn: str | list[str] = 'r',
    quiet: bool = False
) -> int:
    def process_keys(keys: str | list[str]) -> list[int]:
        keys = [keys] if isinstance(keys, str) else keys
        return [ord(key) for key in keys]

    if not quiet:
        print('Video playback controls:')
        print('To quit the video: ', quit_btn)
        print('To skip to the next video: ', next_btn)
        print('To pause/play the video: ', pause_btn)
        print('To go back one frame: ', back_btn)
        print('Any other key advances one frame')

    quit_btn = process_keys(quit_btn)
    next_btn = process_keys(next_btn)
    pause_btn = process_keys(pause_btn)
    back_btn = process_keys(back_btn)

    index = 0
    paused = False
    delay = int(1000 / fps)
    exit_code = 0

    while index < len(frames):
        cv2.imshow(win_name, frames[index])
        key = cv2.waitKey(0 if paused else delay)

        if key in quit_btn:
            exit_code = 1
            break
        elif key in next_btn:
            break
        elif key in pause_btn:
            paused = not paused
        elif key in back_btn:
            index = max(0, index - 1)
            continue

        index += 1

    cv2.destroyWindow(win_name)
    return exit_code

