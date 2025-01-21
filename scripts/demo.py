import argparse
import json
from pathlib import Path
import sys

ROOT_PATH: str = str(Path(__file__).parent.parent)
sys.path.insert(0, ROOT_PATH)
from src.draw import draw_face_anns
from src.face_tracker import FaceTracker
from src.video import Video, play_video
sys.path.remove(ROOT_PATH)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser("Video face tracking demo to visualize detections.")
    parser.add_argument('filename', type=str, help='Path to a video file.')
    parser.add_argument(
        '--ann-path',
        '-a',
        type=str,
        help='Path to a JSON file containing pre-computed annotations. This skips the face detection step.'
    )
    args = parser.parse_args(argv)
    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    filename = args.filename
    ann_path = args.ann_path

    if ann_path is None:
        tracker = FaceTracker()
        faces = tracker(filename)
    else:
        with open(ann_path, 'r') as ann_file:
            faces = json.load(ann_file)

    frames = []
    with Video(filename) as video:
        fps = video.fps
        for frame_idx in range(video.num_frames):
            frame_idx_str = str(frame_idx)
            frame = video.read()
            for face_idx, face in faces.items():
                if frame_idx_str in face:
                    frame = draw_face_anns(frame, face[frame_idx_str], face_idx)

            frames.append(frame)

    print("Showing video. Press 'q' on the window to stop video playback.")
    play_video(frames, fps)


if __name__ == '__main__':
    main(sys.argv[1:])

