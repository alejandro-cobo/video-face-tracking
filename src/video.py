from typing import Callable

import cv2
from imutils.video import FileVideoStream


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

