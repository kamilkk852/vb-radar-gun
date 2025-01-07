class CameraProperties:
    def __init__(self, focal_length: float, sensor_width: float, image_width: int):
        self.focal_length = focal_length
        self.sensor_width = sensor_width
        self.image_width = image_width


class DistanceCalculator:
    def __init__(self, ball_diameter: float, camera_properties: CameraProperties):
        self.ball_diameter = ball_diameter
        self.camera_properties = camera_properties

    def size_to_diameter(self, size: tuple[float, float]) -> float:
        return min(size)

    def pixel_diameter_to_distance(self, pixel_d: float) -> float:
        image_w = self.camera_properties.image_width
        focal_l = self.camera_properties.focal_length
        sensor_w = self.camera_properties.sensor_width
        true_ball_d = self.ball_diameter

        return image_w * true_ball_d * focal_l / (pixel_d * sensor_w)

    def get_distance(self, size: tuple[float, float]) -> float:
        pixel_d = self.size_to_diameter(size)
        return self.pixel_diameter_to_distance(pixel_d)
