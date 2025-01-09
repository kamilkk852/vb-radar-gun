import numpy as np
import supervision as sv
import os
from abc import ABC, abstractmethod


class DetectionExtension(ABC):
    @abstractmethod
    def before_processing(self, video_path: str) -> None:
        pass

    @abstractmethod
    def process_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        pass

    @abstractmethod
    def after_processing(self) -> None:
        pass


class VideoAnnotator(DetectionExtension):
    def __init__(self, output_path: str = "predictions", output_fps: int = 25):
        self.output_path = os.path.join(
            os.path.dirname(os.path.dirname(
                os.path.dirname(__file__))), output_path
        )
        self.output_fps = output_fps
        self.bbox_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def _output_video_path(self, video_path: str) -> str:
        return os.path.join(
            self.output_path,
            os.path.basename(video_path).lower(),  # .replace(".mp4", ".avi"),
        )

    def _video_info(self, video_path: str) -> sv.VideoInfo:
        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        video_info.fps = self.output_fps

        return video_info

    def _output_video_sink(self, video_path: str) -> sv.VideoSink:
        video_info = self._video_info(video_path)
        output_video_path = self._output_video_path(video_path)

        return sv.VideoSink(output_video_path, video_info=video_info, codec="avc1")

    def before_processing(self, video_path: str) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        self.video_sink = self._output_video_sink(video_path).__enter__()

    def process_frame(self, frame: np.ndarray, detections: sv.Detections) -> None:
        if not detections:
            self.video_sink.write_frame(frame)
            return

        labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
        annotated_frame = self.bbox_annotator.annotate(
            frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )
        self.video_sink.write_frame(annotated_frame)

    def after_processing(self) -> None:
        self.video_sink.__exit__(None, None, None)
