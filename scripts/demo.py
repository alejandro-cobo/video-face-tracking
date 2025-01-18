import argparse
from pathlib import Path
import sys

import cv2

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.face_tracker import FaceTracker
from src.utils import draw_face_anns
from src.video import Video
sys.path.remove(ROOT_PATH)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Video face tracking demo to visualize detections.")
    parser.add_argument('filename', type=str, help='Path to a video file.')
    args = parser.parse_args(argv)
    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    tracker = FaceTracker()
    faces = tracker(args.filename)

    print("Showing video. Press 'q' on the window to stop video playback.")
    with Video(args.filename) as video:
        delay = int(1000 / video.fps)
        for frame_idx in range(video.num_frames):
            frame_idx_str = str(frame_idx)
            frame = video.read()
            for face_idx, face in faces.items():
                if frame_idx_str in face:
                    frame = draw_face_anns(frame, face[frame_idx_str], face_idx)

            cv2.imshow('Video', frame)
            key = cv2.waitKey(delay)
            if key == ord('q'):
                break


if __name__ == '__main__':
    main(sys.argv[1:])

