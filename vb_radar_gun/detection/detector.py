import supervision as sv
import numpy as np
import os
from tqdm import tqdm
from typing import Any
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from vb_radar_gun.helpers.sahi import sahi_to_sv
from vb_radar_gun.detection.detections import Detections
from vb_radar_gun.detection.extensions import DetectionExtension, VideoAnnotator


class DetectionModel:
    def __init__(
        self,
        model_path: str = "models/map0782.pt",
        model_type: str = "yolov8",
        confidence_threshold: float = 0.5,
        device: str = "mps",
        half: bool = True,
    ):
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.dirname(__file__))), model_path
        )
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.half = half
        self.det_model = self.init_det_model()

    def init_det_model(self) -> AutoDetectionModel:
        det_model = AutoDetectionModel.from_pretrained(
            model_type=self.model_type,
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device,
        )
        if self.half:
            det_model.model.half()
        return det_model

    def process_frame(
        self, frame: np.ndarray, sahi_parameters: dict[str, Any]
    ) -> sv.Detections:
        result = get_sliced_prediction(
            frame,
            self.det_model,
            **sahi_parameters,
            verbose=0,
        )
        detections = sahi_to_sv(result)
        return detections


class VolleyballDetector:
    def __init__(
        self,
        detection_model: DetectionModel = DetectionModel(),
        sahi_parameters: dict[str, Any] = {
            "slice_height": 640,
            "slice_width": 640,
            "overlap_height_ratio": 0.2,
            "overlap_width_ratio": 0.2,
        },
        track_parameters: dict[str, Any] = {
            "track_activation_threshold": 0.25,
            "lost_track_buffer": 15,
            "minimum_consecutive_frames": 1,
        },
        extensions: list[DetectionExtension] = [VideoAnnotator()],
    ):
        self.detection_model = detection_model
        self.sahi_parameters = sahi_parameters
        self.track_parameters = track_parameters
        self.extensions = extensions

    def detect(
        self, video_path: str, st_time: float = 0.0, end_time: float = None
    ) -> Detections:
        for extension in self.extensions:
            extension.before_processing(video_path)

        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        tracker = sv.ByteTrack(frame_rate=video_info.fps,
                               **self.track_parameters)
        frames_gen = sv.get_video_frames_generator(
            video_path,
            start=int(st_time * video_info.fps),
            end=int(end_time * video_info.fps) if end_time else None,
        )

        times: list[float] = []
        detections_list: list[sv.Detections] = []
        for frame_id, frame in tqdm(enumerate(frames_gen),
                                    total=video_info.total_frames if end_time is None else int(video_info.fps*(end_time-st_time))):
            detections = self.detection_model.process_frame(
                frame, self.sahi_parameters)
            if detections:
                detections = tracker.update_with_detections(detections)
                times.append(frame_id / video_info.fps)
                detections_list.append(detections)

            for extension in self.extensions:
                extension.process_frame(frame, detections)
                
        detections = Detections(times, detections_list)

        for extension in self.extensions:
            extension.after_processing()

        return detections
