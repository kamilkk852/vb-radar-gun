import supervision as sv
import numpy as np
from typing import Any
from itertools import chain


class Detections:
    def __init__(
        self, times: list[float], detections_list: list[sv.Detections]
    ) -> None:
        self.times = times
        self.detections_list = detections_list

        not_none = [
            detections is not None for detections in self.detections_list]
        self.times = [time for time, nn in zip(self.times, not_none) if nn]
        self.detections_list = [
            detections for detections in self.detections_list if detections
        ]

        self.bboxes = [
            np.array(detections[0].xyxy)[0] for detections in self.detections_list
        ]
        self.widths = [bbox[2] - bbox[0] for bbox in self.bboxes]
        self.heights = [bbox[3] - bbox[1] for bbox in self.bboxes]
        self.sizes = list(zip(self.widths, self.heights))

    def filter_by(self, key: str, value: Any) -> "Detections":
        detections = [
            detections[getattr(detections, key) == value]
            for detections in self.detections_list
        ]

        filtered_times = [
            time for time, detections in zip(self.times, detections) if detections
        ]
        filtered_detections = [
            detections for detections in detections if detections]

        return Detections(filtered_times, filtered_detections)

    def unique_attr_vals(self, key: str) -> set:
        return set(
            chain(
                *[list(getattr(detections, key)) for detections in self.detections_list]
            )
        )
