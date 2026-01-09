import cv2
import numpy as np

def draw_boxes(image, boxes, labels):
    img = np.array(image)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box.tolist())

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return img
