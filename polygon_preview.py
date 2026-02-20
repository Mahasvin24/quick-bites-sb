import argparse
import json

import cv2
import numpy as np

from carrillo_counter import DEFAULT_CONFIG


DEFAULT_IMAGE_PATH = "images/carrillo_current.jpg"
DEFAULT_OUTPUT_PATH = "images/carrillo_polygon_preview.jpg"


def _parse_polygon_norm(polygon_text):
    points = json.loads(polygon_text)
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError("Polygon must be a list of at least 3 [x_norm, y_norm] points.")

    parsed = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("Each polygon point must be [x_norm, y_norm].")
        x_norm = float(point[0])
        y_norm = float(point[1])
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            raise ValueError("Normalized points must be within [0, 1].")
        parsed.append((x_norm, y_norm))
    return parsed


def _norm_to_pixels(points_norm, width, height):
    return np.array(
        [(int(x_norm * width), int(y_norm * height)) for x_norm, y_norm in points_norm],
        dtype=np.int32,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Draw an entrance polygon (normalized) on carrillo current image."
    )
    parser.add_argument(
        "--polygon",
        default=json.dumps(DEFAULT_CONFIG["entrance_polygon_norm"]),
        help='Normalized polygon JSON, e.g. \'[[0.38,0.42],[0.62,0.42],[0.62,0.86],[0.38,0.86]]\'',
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE_PATH, help="Input image path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output image path.")
    args = parser.parse_args()

    polygon_norm = _parse_polygon_norm(args.polygon)
    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read image at: {args.image}")

    height, width = image.shape[:2]
    polygon_px = _norm_to_pixels(polygon_norm, width, height)

    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon_px], (0, 255, 0))
    preview = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    cv2.polylines(preview, [polygon_px], isClosed=True, color=(0, 255, 0), thickness=2)

    for idx, (x_px, y_px) in enumerate(polygon_px):
        cv2.circle(preview, (x_px, y_px), 4, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(idx),
            (x_px + 6, y_px - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(args.output, preview)
    print(f"Saved polygon preview to: {args.output}")
    print(f"Polygon norm: {polygon_norm}")
    print(f"Polygon px: {polygon_px.tolist()}")


if __name__ == "__main__":
    main()
