import numpy as np
import supervision as sv
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score
from vb_radar_gun.velocity.camera import CameraProperties, DistanceCalculator
from vb_radar_gun.detection.detector import Detections


class VelocityEstimator:
    def __init__(
        self,
        smooth_window: int = 5,
        ball_diameter: float = 0.21,
        regression_points: int = 30,
        model: RANSACRegressor = RANSACRegressor(stop_probability=0.7, random_state=0),
        camera_properties: CameraProperties = CameraProperties(
            focal_length=15 / 1000, sensor_width=44.2 / 1000, image_width=1920
        ),
    ) -> None:
        self.smooth_window = smooth_window
        self.regression_points = regression_points
        self.camera_properties = camera_properties
        self.distance_calculator = DistanceCalculator(ball_diameter, camera_properties)
        self.model = model

    def _smooth_detections(self, detections: Detections) -> Detections:
        smoother = sv.DetectionsSmoother(length=self.smooth_window)
        smoothed_detections_list = [
            smoother.update_with_detections(sv_detections)
            for sv_detections in detections.detections_list
        ]

        return Detections(detections.times, smoothed_detections_list)

    def _get_data(self, detections: Detections) -> tuple[np.ndarray, np.ndarray]:
        distances = [
            self.distance_calculator.get_distance(size) for size in detections.sizes
        ]

        return np.array(detections.times), np.array(distances)

    def _slope_to_velocity(self, slope: float) -> float:
        return 1.1 * 3.6 * slope

    def _criterion(self, slope: float, r2: float) -> bool:
        velocity = self._slope_to_velocity(slope)

        return velocity >= 20 and velocity < 150 and r2 > 0.9

    def _find_best_slope(
        self, times: np.ndarray, distances: np.ndarray
    ) -> tuple[float, float]:
        best_performace = 0.0
        best_slope = 0.0

        for i in range(max(1, len(times) - self.regression_points)):
            X = times[i : i + self.regression_points].reshape(-1, 1)
            y = distances[i : i + self.regression_points]

            self.model.fit(X, y)
            slope = abs(self.model.estimator_.coef_[0])
            r2 = r2_score(y, self.model.predict(X))
            performance = slope / np.power(1 - r2, 1 / 4)

            if performance > best_performace and self._criterion(slope, r2):
                best_performace = performance
                best_slope = slope

        return best_slope

    def estimate_velocity(self, detections: Detections) -> float:
        detections = self._smooth_detections(detections)

        velocities = []
        for tracker_id in detections.unique_attr_vals("tracker_id"):
            tracker_detections = detections.filter_by("tracker_id", tracker_id)
            times, distances = self._get_data(tracker_detections)
            best_slope = self._find_best_slope(times, distances)
            velocity = self._slope_to_velocity(best_slope)
            velocities.append(velocity)

        return max(velocities, default=0.0)
