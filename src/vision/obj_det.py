import cv2
import numpy as np
from picamera2 import Picamera2
from dataclasses import dataclass


@dataclass
class DetectionResult:
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float
    contour: np.ndarray
    box_points: np.ndarray
    confidence: float = 0.0
    part_id: str | None = None


def score_to_confidence(best_score: float, threshold: float = 0.2) -> float:
    if threshold <= 0:
        conf = 1.0 / (1.0 + best_score)
    else:
        conf = 1.0 - (best_score / threshold)
    return float(np.clip(conf, 0.0, 1.0))

# ---------------------------
# Load reference images (edges or binary)
# ---------------------------
def get_contour(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Threshold to binary
    _, thresh = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Return largest contour
    return max(contours, key=cv2.contourArea), thresh
    

contour1, thresh1 = get_contour("Part_1.jpg")
contour2, thresh2 = get_contour("Part_2.jpg")
contour3, thresh3 = get_contour("Part_3.jpg")
contour4, thresh4 = get_contour("Part_4.jpg")

# ---------------------------
# Camera setup
# ---------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

# ---------------------------
# Main loop
# ---------------------------
while True:
    frame = picam2.capture_array()

    cropped = frame[70:330, 0:640]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Binary image (VERY important)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get biggest object
        cnt = max(contours, key=cv2.contourArea)

        # Ignore tiny noise
        if cv2.contourArea(cnt) > 500:
            # Centroid from image moments (in cropped ROI coordinates)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            # Orientation from minimum-area rectangle
            rect = cv2.minAreaRect(cnt)  # ((cx,cy), (w,h), angle)
            (_, _), (w, h), angle = rect
            if w < h:
                angle = angle + 90

            # Convert centroid to full-frame coordinates
            cx_full = cx
            cy_full = cy + 150  # ROI starts at y=150

            # Compare shapes
            score1 = cv2.matchShapes(contour1, cnt, 1, 0.0)
            score2 = cv2.matchShapes(contour2, cnt, 1, 0.0)
            score3 = cv2.matchShapes(contour3, cnt, 1, 0.0)
            score4 = cv2.matchShapes(contour4, cnt, 1, 0.0)

            scores = [score1, score2, score3, score4]

            best_score = min(scores)  # LOWER is better
            best_index = scores.index(best_score)
            part_id = f"Part_{best_index + 1}" if best_score < 0.2 else None
            confidence = score_to_confidence(best_score, threshold=0.2)

            print(f"Scores: {scores}")
            print(f"Centroid (cropped): ({cx}, {cy})")
            print(f"Centroid (full):    ({cx_full}, {cy_full})")
            print(f"Angle: {angle:.1f} deg")

            if best_score < 0.2:  # tune this
                print(f"✅ Detected Part {best_index + 1}")

            # Draw contour
            cv2.drawContours(cropped, [cnt], -1, (0, 255, 0), 2)

            # Draw oriented bounding box and centroid
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            detection_result = DetectionResult(
                center_x=float(cx),
                center_y=float(cy),
                width=float(w),
                height=float(h),
                angle=float(angle),
                contour=cnt,
                box_points=box,
                confidence=confidence,
                part_id=part_id,
            )

            print(f"DetectionResult: {detection_result}")

            cv2.drawContours(cropped, [box], 0, (255, 0, 0), 2)
            cv2.circle(cropped, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                cropped,
                f"{part_id or 'Unknown'} C=({cx_full},{cy_full}) A={angle:.1f}",
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Threshold", thresh)
    cv2.imshow("Frame", frame)
    thresh1_small = cv2.resize(thresh1, (320, 240))
    thresh2_small = cv2.resize(thresh2, (320, 240))
    thresh3_small = cv2.resize(thresh3, (320, 240))
    thresh4_small = cv2.resize(thresh4, (320, 240))
    cv2.imshow("thresh1", thresh1_small)
    cv2.imshow("thresh2", thresh2_small)
    cv2.imshow("thresh3", thresh3_small)
    cv2.imshow("thresh4", thresh4_small)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()