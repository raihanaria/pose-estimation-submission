import cv2


# COCO 17 keypoints
SKELETON = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def draw_pose(
    image,
    keypoints,
    keypoint_conf,
    bbox,
    activity_result,
    person_id,
    keypoint_threshold=0.35
):

    output = image.copy()

    # -----------------------------------------------------
    # Draw skeleton
    # -----------------------------------------------------

    for a, b in SKELETON:

        if (
            keypoint_conf[a] is None
            or keypoint_conf[b] is None
        ):
            continue

        if (
            keypoint_conf[a] < keypoint_threshold
            or keypoint_conf[b] < keypoint_threshold
        ):
            continue

        xa, ya = keypoints[a]
        xb, yb = keypoints[b]

        cv2.line(
            output,
            (int(xa), int(ya)),
            (int(xb), int(yb)),
            (255, 200, 0),
            2
        )

    # -----------------------------------------------------
    # Draw keypoints
    # -----------------------------------------------------

    for i, point in enumerate(keypoints):

        conf = keypoint_conf[i]

        if conf is None:
            continue

        if conf < keypoint_threshold:
            continue

        x, y = point

        cv2.circle(
            output,
            (int(x), int(y)),
            4,
            (0, 255, 0),
            -1
        )

    # -----------------------------------------------------
    # Bounding box
    # -----------------------------------------------------

    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        output,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (255, 255, 0),
        2
    )

    # -----------------------------------------------------
    # Activity label
    # -----------------------------------------------------

    activity = activity_result[
        "activity"
    ]

    confidence = activity_result[
        "confidence"
    ]

    text = (
        f"Person {person_id}: "
        f"{activity}"
    )

    if activity != "Unknown":

        text += (
            f" ({confidence:.2f})"
        )

    text_y = max(
        int(y1) - 10,
        20
    )

    cv2.putText(
        output,
        text,
        (int(x1), text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output