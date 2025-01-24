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
    parser.add_argument(
        '--det-thresh',
        type=float,
        default=0.7,
        help='Minimum detector confidence score to consider a detection as valid. Default: 0.7.'
    )
    parser.add_argument(
        '--box-disp-thresh',
        type=float,
        default=0.3,
        help='Maximum displacement of a bounding box to consider it the same as the previous frame. Default: 0.3 (30 '
             'percent of the max side of the box).'
    )
    parser.add_argument(
        '--cos-sim-thresh',
        type=float,
        default=0.5,
        help='Maximum cosine similarity score to match two faces. Default: 0.5.'
    )
    args = parser.parse_args(argv)
    return args


def main(argv: list[str]) -> None:
    args = parse_args(argv)
    filename = args.filename
    ann_path = args.ann_path
    det_thresh = args.det_thresh
    box_disp_thresh = args.box_disp_thresh
    cos_sim_thresh = args.cos_sim_thresh

    if ann_path is None:
        face_tracker = FaceTracker(
            det_thresh=det_thresh,
            box_disp_thresh=box_disp_thresh,
            cos_sim_thresh=cos_sim_thresh
        )
        faces = face_tracker(filename)
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

    play_video(frames, fps)


if __name__ == '__main__':
    main(sys.argv[1:])

