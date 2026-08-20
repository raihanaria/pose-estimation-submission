# main.py

import os
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from feature_extraction import extract_features
from action_classifier import classify_person
from visualization import draw_pose


# =========================================================
# CONFIGURATION
# =========================================================

TEST_DIR = "test_set"

MODEL_PATH = "yolo11n-pose.pt"

OUTPUT_DIR = "outputs"
ANNOTATED_DIR = os.path.join(
    OUTPUT_DIR,
    "annotated"
)

JSON_DIR = os.path.join(
    OUTPUT_DIR,
    "results"
)

SUBMISSION_DIR = os.path.join(
    OUTPUT_DIR,
    "submission"
)

PERSON_CONF = 0.25
KEYPOINT_CONF = 0.35


# Assessment says 6 outputs although
# objective mentions 7 input images.
#
# We process all 7 internally but
# exclude images3.jpeg from submission
# because it represents an unsupported
# squat pose according to our assumption.

EXCLUDED_FROM_SUBMISSION = {
    "images3.jpeg"
}


def make_directories():

    os.makedirs(
        ANNOTATED_DIR,
        exist_ok=True
    )

    os.makedirs(
        JSON_DIR,
        exist_ok=True
    )

    os.makedirs(
        SUBMISSION_DIR,
        exist_ok=True
    )


def convert_value(value):

    if value is None:
        return None

    if isinstance(
        value,
        (np.float32, np.float64)
    ):
        return float(value)

    if isinstance(
        value,
        (np.int32, np.int64)
    ):
        return int(value)

    return value


def main():

    make_directories()

    # -----------------------------------------------------
    # Load pose model
    # -----------------------------------------------------

    print(
        f"Loading model: {MODEL_PATH}"
    )

    model = YOLO(
        MODEL_PATH
    )

    image_files = sorted(
        [
            p for p in Path(
                TEST_DIR
            ).iterdir()
            if p.suffix.lower()
            in [
                ".jpg",
                ".jpeg",
                ".png"
            ]
        ]
    )

    print(
        f"Found {len(image_files)} images."
    )

    all_results = {}

    # -----------------------------------------------------
    # Process images
    # -----------------------------------------------------

    for image_path in image_files:

        print()
        print(
            "=" * 70
        )

        print(
            f"Processing: "
            f"{image_path.name}"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "ERROR: Cannot read image."
            )

            continue

        annotated = image.copy()

        # -------------------------------------------------
        # Pose inference
        # -------------------------------------------------

        results = model.predict(
            source=image,
            conf=PERSON_CONF,
            verbose=False
        )

        result = results[0]

        image_result = {
            "image": image_path.name,
            "persons": []
        }

        # -------------------------------------------------
        # No person
        # -------------------------------------------------

        if (
            result.keypoints is None
            or result.keypoints.xy is None
        ):

            print(
                "No person detected."
            )

            output_name = (
                image_path.stem
                + "_result.jpg"
            )

            cv2.imwrite(
                os.path.join(
                    ANNOTATED_DIR,
                    output_name
                ),
                annotated
            )

            all_results[
                image_path.name
            ] = image_result

            continue

        keypoints_xy = (
            result.keypoints
            .xy
            .cpu()
            .numpy()
        )

        if result.keypoints.conf is not None:

            keypoints_conf = (
                result.keypoints
                .conf
                .cpu()
                .numpy()
            )

        else:

            keypoints_conf = np.ones(
                (
                    len(keypoints_xy),
                    17
                ),
                dtype=float
            )

        # -------------------------------------------------
        # Bounding boxes
        # -------------------------------------------------

        boxes = None

        if result.boxes is not None:

            boxes = (
                result.boxes
                .xyxy
                .cpu()
                .numpy()
            )

        # -------------------------------------------------
        # Person loop
        # -------------------------------------------------

        for person_idx in range(
            len(keypoints_xy)
        ):

            person_number = (
                person_idx + 1
            )

            raw_points = (
                keypoints_xy[
                    person_idx
                ]
            )

            raw_conf = (
                keypoints_conf[
                    person_idx
                ]
            )

            # -------------------------------------------------
            # Reliability filtering
            # -------------------------------------------------

            reliable_points = []

            reliable_conf = []

            for kp_idx in range(17):

                x, y = raw_points[
                    kp_idx
                ]

                conf = float(
                    raw_conf[
                        kp_idx
                    ]
                )

                if conf >= KEYPOINT_CONF:

                    reliable_points.append(
                        [
                            float(x),
                            float(y)
                        ]
                    )

                    reliable_conf.append(
                        conf
                    )

                else:

                    reliable_points.append(
                        None
                    )

                    reliable_conf.append(
                        None
                    )

            # -------------------------------------------------
            # Feature extraction
            # -------------------------------------------------

            features = extract_features(
                reliable_points
            )

            # -------------------------------------------------
            # Classification
            # -------------------------------------------------

            classification = (
                classify_person(
                    features
                )
            )

            print(
                f"Person #{person_number}: "
                f"{classification['activity']} "
                f"("
                f"{classification['confidence']:.2f}"
                f")"
            )

            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            if boxes is not None:

                bbox = boxes[
                    person_idx
                ]

            else:

                xs = [
                    p[0]
                    for p in reliable_points
                    if p is not None
                ]

                ys = [
                    p[1]
                    for p in reliable_points
                    if p is not None
                ]

                if xs and ys:

                    bbox = [
                        min(xs),
                        min(ys),
                        max(xs),
                        max(ys)
                    ]

                else:

                    bbox = [
                        0,
                        0,
                        0,
                        0
                    ]

            # -------------------------------------------------
            # Visualization
            # -------------------------------------------------

            annotated = draw_pose(
                annotated,
                reliable_points,
                reliable_conf,
                bbox,
                classification,
                person_number,
                KEYPOINT_CONF
            )

            # -------------------------------------------------
            # JSON keypoint output
            # -------------------------------------------------

            keypoint_output = []

            for kp_idx in range(17):

                point = (
                    reliable_points[
                        kp_idx
                    ]
                )

                conf = (
                    reliable_conf[
                        kp_idx
                    ]
                )

                keypoint_output.append({

                    "id": kp_idx,

                    "name": [
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
                        "right_ankle"
                    ][kp_idx],

                    "x":
                        None
                        if point is None
                        else point[0],

                    "y":
                        None
                        if point is None
                        else point[1],

                    "confidence":
                        conf,

                    "reliable":
                        conf is not None
                })

            person_result = {

                "person_id":
                    person_number,

                "bbox": [
                    float(x)
                    for x in bbox
                ],

                "keypoints":
                    keypoint_output,

                "features":
                    {
                        key:
                            convert_value(
                                value
                            )
                        for key, value
                        in features.items()
                    },

                "classification":
                    classification
            }

            image_result[
                "persons"
            ].append(
                person_result
            )

        # -------------------------------------------------
        # Save annotated image
        # -------------------------------------------------

        output_name = (
            image_path.stem
            + "_result.jpg"
        )

        annotated_path = os.path.join(
            ANNOTATED_DIR,
            output_name
        )

        cv2.imwrite(
            annotated_path,
            annotated
        )

        # -------------------------------------------------
        # Save JSON
        # -------------------------------------------------

        json_name = (
            image_path.stem
            + "_result.json"
        )

        json_path = os.path.join(
            JSON_DIR,
            json_name
        )

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                image_result,
                f,
                indent=2,
                ensure_ascii=False
            )

        # -------------------------------------------------
        # Submission output
        # -------------------------------------------------

        if (
            image_path.name
            not in EXCLUDED_FROM_SUBMISSION
        ):

            submission_path = os.path.join(
                SUBMISSION_DIR,
                output_name
            )

            cv2.imwrite(
                submission_path,
                annotated
            )

        all_results[
            image_path.name
        ] = image_result

    # -----------------------------------------------------
    # Save combined JSON
    # -----------------------------------------------------

    combined_path = os.path.join(
        OUTPUT_DIR,
        "all_results.json"
    )

    with open(
        combined_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "=" * 70
    )

    print(
        "PROCESSING COMPLETE"
    )

    print(
        f"Annotated images: "
        f"{ANNOTATED_DIR}"
    )

    print(
        f"Submission images: "
        f"{SUBMISSION_DIR}"
    )

    print(
        f"JSON results: "
        f"{JSON_DIR}"
    )

    print(
        f"Combined JSON: "
        f"{combined_path}"
    )


if __name__ == "__main__":
    main()