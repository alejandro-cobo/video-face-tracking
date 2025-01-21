from typing import Any

from insightface.app import FaceAnalysis
import numpy as np
from tqdm import tqdm

from .video import Video

__all__ = ['FaceTracker']

FaceAnnotation = dict[str, dict[str, Any]]


class FaceEmbeddings:
    """ Utility class to store face embeddings """
    def __init__(self) -> None:
        self.face_embeddings = dict()
        self.last_bbox = dict()

    def __len__(self) -> int:
        return len(self.face_embeddings)

    def add(self, face_id: str, bbox: np.ndarray, emb: np.ndarray) -> None:
        self.face_embeddings.setdefault(face_id, []).append(emb)
        self.last_bbox[face_id] = bbox

    def get_embedding(self, face_id: str) -> np.ndarray:
        if face_id not in self.face_embeddings:
            raise RuntimeError(f'Face ID not found: {face_id}')
        return np.mean(np.stack(self.face_embeddings[face_id]), axis=0)

    def get_cos_sim(self, face_id: str, emb: np.ndarray) -> float:
        ref_emb = self.get_embedding(face_id)
        dot_product = np.matmul(ref_emb, emb)
        magnitude_emb = np.linalg.norm(emb)
        magnitude_ref_emb = np.linalg.norm(ref_emb)
        return 1 - dot_product / (magnitude_emb * magnitude_ref_emb)

    def get_closest_face(self, emb: np.ndarray) -> tuple[str, float]:
        ref_emb = np.stack([self.get_embedding(face_id) for face_id in self.face_embeddings])
        dot_product = np.matmul(ref_emb, emb)
        magnitude_emb = np.linalg.norm(emb)
        magnitude_ref_emb = np.linalg.norm(ref_emb, axis=1)
        cos_sim = 1 - dot_product / (magnitude_emb * magnitude_ref_emb)
        idx_min = int(np.argmin(cos_sim))
        face_id = list(self.face_embeddings.keys())[idx_min]
        return face_id, cos_sim[idx_min]

    def get_closest_box(self, bbox: np.ndarray) -> tuple[str, float]:
        center = (bbox[:2] + bbox[2:]) / 2
        ref_boxes = np.stack([box for box in self.last_bbox.values()])
        ref_centers = (ref_boxes[:, :2] + ref_boxes[:, 2:]) / 2
        distances = np.linalg.norm(center[None, :] - ref_centers, axis=1)
        idx_min = int(np.argmin(distances))
        face_id = list(self.last_bbox.keys())[idx_min]
        min_dist = distances[idx_min] / max(bbox[[2, 3]] - bbox[[0, 1]])
        return face_id, min_dist


class FaceTracker:
    def __init__(
        self,
        det_thresh: float = 0.0,
        box_disp_thresh: float = 0.3,
        cos_sim_thresh: float = 0.5,
        max_frames: int | None = None,
        quiet: bool = False
    ) -> None:
        self.det_thresh = det_thresh
        self.box_disp_thresh = box_disp_thresh
        self.cos_sim_thresh = cos_sim_thresh
        self.max_frames = max_frames
        self.quiet = quiet

        self.app = FaceAnalysis(allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def __call__(self, filename: str) -> dict[str, FaceAnnotation]:
        face_anns = {}
        face_emb = FaceEmbeddings()
        with Video(filename, max_frames=self.max_frames) as video:
            for frame_idx in tqdm(range(video.num_frames), desc='Processing video', leave=False, disable=self.quiet):
                frame = video.read()
                faces = self.app.get(frame)
                for face_idx, face in enumerate(faces):
                    if face.det_score < self.det_thresh:
                        continue
                    face_dict = {
                        'bbox': face.bbox.tolist(),
                        'prob': float(face.det_score),
                        'landmarks': face.kps.flatten().tolist()
                    }

                    if len(face_emb) == 0:
                        final_class = str(face_idx)
                        face_anns[final_class] = {str(frame_idx): face_dict}
                    else:
                        final_class = None
                        box_class, box_dist = face_emb.get_closest_box(face.bbox)
                        if (
                            box_dist < self.box_disp_thresh and
                            face_emb.get_cos_sim(box_class, face.embedding) < self.cos_sim_thresh * 3
                        ):
                            final_class = box_class

                        if final_class is None:
                            face_class, emb_dist = face_emb.get_closest_face(face.embedding)
                            if emb_dist < self.cos_sim_thresh:
                                final_class = face_class

                        if final_class is None:
                            final_class = str(len(face_anns))
                        face_anns.setdefault(final_class, {})[str(frame_idx)] = face_dict

                    face_emb.add(final_class, face.bbox, face.embedding)

        return face_anns


