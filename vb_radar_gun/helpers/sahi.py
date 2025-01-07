import numpy as np
import supervision as sv


def sahi_to_sv(sahi_predictions) -> sv.Detections:
    bboxes = []
    scores = []
    class_ids = []

    for prediction in sahi_predictions.to_coco_annotations():
        x_min, y_min, width, height = prediction["bbox"]
        x_max = x_min + width
        y_max = y_min + height
        bboxes.append([x_min, y_min, x_max, y_max])
        scores.append(prediction["score"])
        class_ids.append(prediction["category_id"])

    if not bboxes:
        return None
    # Convert lists to NumPy arrays
    bboxes = np.array(bboxes)
    scores = np.array(scores)
    class_ids = np.array(class_ids)

    # Create sv.Detections object
    detections = sv.Detections(xyxy=bboxes, confidence=scores, class_id=class_ids)

    return detections
