from typing import Any

from insightface.app import FaceAnalysis
import numpy as np
from tqdm import tqdm

from .video import Video

__all__ = ['FaceTracker']

FaceAnnotation = dict[str, dict[str, Any]]


class FaceTracker:
    def __init__(self, det_thresh: float = 0.0, box_disp_thresh: float = 0.1, cos_sim_thresh: float = 0.7) -> None:
        self.det_thresh = det_thresh
        self.box_disp_thresh = box_disp_thresh
        self.cos_sim_thresh = cos_sim_thresh
        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, filename: str, max_frames: int | None = None, quiet: bool = False) -> dict[str, FaceAnnotation]:
        video_anns = {}
        with Video(filename, max_frames=max_frames) as video:
            for frame_idx in tqdm(range(video.num_frames), desc='Processing video', leave=False, disable=quiet):
                frame = video.read()
                faces = self.app.get(frame)
                video_anns[str(frame_idx)] = faces

        if len(video_anns) == 0:
            return {}

        last_boxes = None
        last_classes = None
        face_anns = {}
        face_emb = {}
        for frame_idx, faces in tqdm(video_anns.items(), desc='Generating annotations', leave=False, disable=quiet):
            curr_boxes = []
            curr_classes = []

            for face_idx, face in enumerate(faces):
                if face.det_score < self.det_thresh:
                    continue
                face_dict = {
                    'bbox': face.bbox.tolist(),
                    'prob': float(face.det_score),
                    'landmarks': face.kps.flatten().tolist()
                }

                if last_boxes is None:
                    final_class = str(face_idx)
                    face_anns[final_class] = {str(frame_idx): face_dict}
                    face_emb[final_class] = face.embedding
                else:
                    box_idx_min, box_dist = self.boxes_get_min_dist(face.bbox, last_boxes)
                    box_dist = box_dist / max(face.bbox[[2, 3]] - face.bbox[[0, 1]])
                    face_class, emb_dist = self.emb_get_min_cos_sim(face.embedding, face_emb)

                    if box_dist < self.box_disp_thresh:
                        final_class = str(last_classes[box_idx_min])
                    elif emb_dist < self.cos_sim_thresh:
                        final_class = face_class
                    else:
                        final_class = str(len(face_anns))
                        face_emb[final_class] = face.embedding
                    face_anns.setdefault(final_class, {})[frame_idx] = face_dict

                curr_boxes.append(face.bbox)
                curr_classes.append(final_class)

            if len(curr_boxes) > 0:
                last_boxes = np.array(curr_boxes)
                last_classes = curr_classes

        return face_anns

    def boxes_get_min_dist(self, bbox: np.ndarray, ref_boxes: np.ndarray) -> tuple[int, float]:
        center = (bbox[:2] + bbox[2:]) / 2
        ref_centers = (ref_boxes[:, :2] + ref_boxes[:, 2:]) / 2
        distances = np.linalg.norm(center[None, :] - ref_centers, axis=1)
        idx_min = int(np.argmin(distances))
        return idx_min, distances[idx_min]

    def emb_get_min_cos_sim(self, emb: np.ndarray, face_emb: dict[str, np.ndarray]) -> tuple[str, float]:
        ref_emb = np.stack([v for v in face_emb.values()])
        dot_product = np.matmul(ref_emb, emb)
        magnitude_emb = np.linalg.norm(emb)
        magnitude_ref_emb = np.linalg.norm(ref_emb, axis=1)
        cos_sim = 1 - dot_product / (magnitude_emb * magnitude_ref_emb)
        idx_min = int(np.argmin(cos_sim))
        face_idx = list(face_emb.keys())[idx_min]
        return face_idx, cos_sim[idx_min]

