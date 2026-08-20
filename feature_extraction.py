import math
import numpy as np


# =========================================================
# COCO 17 KEYPOINT INDICES
# =========================================================

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4

LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6

LEFT_ELBOW = 7
RIGHT_ELBOW = 8

LEFT_WRIST = 9
RIGHT_WRIST = 10

LEFT_HIP = 11
RIGHT_HIP = 12

LEFT_KNEE = 13
RIGHT_KNEE = 14

LEFT_ANKLE = 15
RIGHT_ANKLE = 16


# =========================================================
# KEYPOINT HANDLING
# =========================================================

def get_point(keypoints, index):
    """
    Get one keypoint as numpy array [x, y].

    Supported input formats:

    FORMAT 1:
        [x, y]

    FORMAT 2:
        {
            "x": x,
            "y": y,
            "reliable": True
        }

    FORMAT 3:
        None

    Returns:
        numpy array [x, y]
        or None if keypoint is unavailable/unreliable.
    """

    # -----------------------------------------------------
    # Check index
    # -----------------------------------------------------

    if keypoints is None:
        return None

    if index < 0 or index >= len(keypoints):
        return None

    kp = keypoints[index]

    # -----------------------------------------------------
    # Missing keypoint
    # -----------------------------------------------------

    if kp is None:
        return None

    # -----------------------------------------------------
    # Dictionary format
    # -----------------------------------------------------

    if isinstance(kp, dict):

        reliable = kp.get(
            "reliable",
            True
        )

        if not reliable:
            return None

        x = kp.get("x")
        y = kp.get("y")

        if x is None or y is None:
            return None

        try:

            x = float(x)
            y = float(y)

        except (
            TypeError,
            ValueError
        ):

            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return None

        return np.array(
            [x, y],
            dtype=np.float32
        )

    # -----------------------------------------------------
    # List / tuple / numpy array format
    # -----------------------------------------------------

    if isinstance(
        kp,
        (list, tuple, np.ndarray)
    ):

        if len(kp) < 2:
            return None

        try:

            x = float(kp[0])
            y = float(kp[1])

        except (
            TypeError,
            ValueError
        ):

            return None

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            return None

        return np.array(
            [x, y],
            dtype=np.float32
        )

    # -----------------------------------------------------
    # Unknown format
    # -----------------------------------------------------

    return None


# =========================================================
# BASIC GEOMETRY
# =========================================================

def midpoint(point_a, point_b):
    """
    Calculate midpoint between two points.
    """

    if (
        point_a is None
        or point_b is None
    ):
        return None

    return (
        point_a + point_b
    ) / 2.0


def distance(point_a, point_b):
    """
    Calculate Euclidean distance between two points.
    """

    if (
        point_a is None
        or point_b is None
    ):
        return None

    value = np.linalg.norm(
        point_a - point_b
    )

    if not math.isfinite(float(value)):
        return None

    return float(value)


def calculate_angle(
    point_a,
    point_b,
    point_c
):
    """
    Calculate angle ABC in degrees.

    B is the joint being measured.

             A
              \
               B
              /
             C
    """

    if (
        point_a is None
        or point_b is None
        or point_c is None
    ):
        return None

    vector_ba = (
        point_a - point_b
    )

    vector_bc = (
        point_c - point_b
    )

    norm_ba = np.linalg.norm(
        vector_ba
    )

    norm_bc = np.linalg.norm(
        vector_bc
    )

    if (
        norm_ba < 1e-6
        or norm_bc < 1e-6
    ):
        return None

    cosine_angle = (
        np.dot(
            vector_ba,
            vector_bc
        )
        /
        (
            norm_ba
            * norm_bc
        )
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(
            cosine_angle
        )
    )

    if not math.isfinite(
        float(angle)
    ):
        return None

    return float(angle)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize_keypoints(keypoints):
    """
    Normalize keypoints relative to hip center
    and torso length.

    This function is kept available for
    future feature engineering.

    Returns:
        list of dictionaries
        or None if normalization is impossible.
    """

    left_hip = get_point(
        keypoints,
        LEFT_HIP
    )

    right_hip = get_point(
        keypoints,
        RIGHT_HIP
    )

    left_shoulder = get_point(
        keypoints,
        LEFT_SHOULDER
    )

    right_shoulder = get_point(
        keypoints,
        RIGHT_SHOULDER
    )

    hip_center = midpoint(
        left_hip,
        right_hip
    )

    shoulder_center = midpoint(
        left_shoulder,
        right_shoulder
    )

    if (
        hip_center is None
        or shoulder_center is None
    ):
        return None

    torso_scale = distance(
        hip_center,
        shoulder_center
    )

    if (
        torso_scale is None
        or torso_scale < 1e-6
    ):
        return None

    normalized = []

    for index in range(
        len(keypoints)
    ):

        point = get_point(
            keypoints,
            index
        )

        if point is None:

            normalized.append({
                "x": None,
                "y": None,
                "reliable": False
            })

            continue

        x = (
            point[0]
            - hip_center[0]
        ) / torso_scale

        y = (
            point[1]
            - hip_center[1]
        ) / torso_scale

        normalized.append({
            "x": float(x),
            "y": float(y),
            "reliable": True
        })

    return normalized


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_features(keypoints):
    """
    Extract geometric pose features from
    the 17 human pose keypoints.

    Input:
        keypoints:
            list containing either:

                [x, y]

            or:

                {
                    "x": x,
                    "y": y,
                    "reliable": True
                }

            or:

                None

    Output:
        dictionary containing pose features.
    """

    # =====================================================
    # RAW KEYPOINTS
    # =====================================================

    nose = get_point(
        keypoints,
        NOSE
    )

    left_shoulder = get_point(
        keypoints,
        LEFT_SHOULDER
    )

    right_shoulder = get_point(
        keypoints,
        RIGHT_SHOULDER
    )

    left_elbow = get_point(
        keypoints,
        LEFT_ELBOW
    )

    right_elbow = get_point(
        keypoints,
        RIGHT_ELBOW
    )

    left_wrist = get_point(
        keypoints,
        LEFT_WRIST
    )

    right_wrist = get_point(
        keypoints,
        RIGHT_WRIST
    )

    left_hip = get_point(
        keypoints,
        LEFT_HIP
    )

    right_hip = get_point(
        keypoints,
        RIGHT_HIP
    )

    left_knee = get_point(
        keypoints,
        LEFT_KNEE
    )

    right_knee = get_point(
        keypoints,
        RIGHT_KNEE
    )

    left_ankle = get_point(
        keypoints,
        LEFT_ANKLE
    )

    right_ankle = get_point(
        keypoints,
        RIGHT_ANKLE
    )

    # =====================================================
    # BODY CENTERS
    # =====================================================

    shoulder_center = midpoint(
        left_shoulder,
        right_shoulder
    )

    hip_center = midpoint(
        left_hip,
        right_hip
    )

    # =====================================================
    # JOINT ANGLES
    # =====================================================

    left_elbow_angle = calculate_angle(
        left_shoulder,
        left_elbow,
        left_wrist
    )

    right_elbow_angle = calculate_angle(
        right_shoulder,
        right_elbow,
        right_wrist
    )

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    # =====================================================
    # BODY DIMENSIONS
    # =====================================================

    shoulder_width = distance(
        left_shoulder,
        right_shoulder
    )

    hip_width = distance(
        left_hip,
        right_hip
    )

    shoulder_to_hip = distance(
        shoulder_center,
        hip_center
    )

    # =====================================================
    # LIMB / BODY DISTANCES
    # =====================================================

    wrist_distance = distance(
        left_wrist,
        right_wrist
    )

    ankle_distance = distance(
        left_ankle,
        right_ankle
    )

    # =====================================================
    # ARM LENGTH
    # =====================================================

    left_upper_arm = distance(
        left_shoulder,
        left_elbow
    )

    left_forearm = distance(
        left_elbow,
        left_wrist
    )

    if (
        left_upper_arm is not None
        and left_forearm is not None
    ):

        left_arm_length = (
            left_upper_arm
            + left_forearm
        )

    else:

        left_arm_length = None

    right_upper_arm = distance(
        right_shoulder,
        right_elbow
    )

    right_forearm = distance(
        right_elbow,
        right_wrist
    )

    if (
        right_upper_arm is not None
        and right_forearm is not None
    ):

        right_arm_length = (
            right_upper_arm
            + right_forearm
        )

    else:

        right_arm_length = None

    # =====================================================
    # LEG LENGTH
    # =====================================================

    left_thigh = distance(
        left_hip,
        left_knee
    )

    left_shin = distance(
        left_knee,
        left_ankle
    )

    if (
        left_thigh is not None
        and left_shin is not None
    ):

        left_leg_length = (
            left_thigh
            + left_shin
        )

    else:

        left_leg_length = None

    right_thigh = distance(
        right_hip,
        right_knee
    )

    right_shin = distance(
        right_knee,
        right_ankle
    )

    if (
        right_thigh is not None
        and right_shin is not None
    ):

        right_leg_length = (
            right_thigh
            + right_shin
        )

    else:

        right_leg_length = None

    # =====================================================
    # RELATIVE WRIST POSITIONS
    # =====================================================

    left_wrist_relative_y = None

    right_wrist_relative_y = None

    if (
        left_wrist is not None
        and shoulder_center is not None
    ):

        left_wrist_relative_y = float(
            left_wrist[1]
            - shoulder_center[1]
        )

    if (
        right_wrist is not None
        and shoulder_center is not None
    ):

        right_wrist_relative_y = float(
            right_wrist[1]
            - shoulder_center[1]
        )

    # =====================================================
    # RELATIVE KNEE POSITIONS
    # =====================================================

    left_knee_relative_y = None

    right_knee_relative_y = None

    if (
        left_knee is not None
        and hip_center is not None
    ):

        left_knee_relative_y = float(
            left_knee[1]
            - hip_center[1]
        )

    if (
        right_knee is not None
        and hip_center is not None
    ):

        right_knee_relative_y = float(
            right_knee[1]
            - hip_center[1]
        )

    # =====================================================
    # ADDITIONAL USEFUL FEATURES
    # =====================================================

    # Average knee angle
    knee_angles = [
        value
        for value in [
            left_knee_angle,
            right_knee_angle
        ]
        if value is not None
    ]

    if knee_angles:

        mean_knee_angle = float(
            sum(knee_angles)
            / len(knee_angles)
        )

    else:

        mean_knee_angle = None

    # Average elbow angle
    elbow_angles = [
        value
        for value in [
            left_elbow_angle,
            right_elbow_angle
        ]
        if value is not None
    ]

    if elbow_angles:

        mean_elbow_angle = float(
            sum(elbow_angles)
            / len(elbow_angles)
        )

    else:

        mean_elbow_angle = None

    # Elbow asymmetry
    if (
        left_elbow_angle is not None
        and right_elbow_angle is not None
    ):

        elbow_angle_difference = float(
            abs(
                left_elbow_angle
                - right_elbow_angle
            )
        )

    else:

        elbow_angle_difference = None

    # Knee asymmetry
    if (
        left_knee_angle is not None
        and right_knee_angle is not None
    ):

        knee_angle_difference = float(
            abs(
                left_knee_angle
                - right_knee_angle
            )
        )

    else:

        knee_angle_difference = None

    # =====================================================
    # FINAL FEATURE DICTIONARY
    # =====================================================

    features = {

        # -------------------------------------------------
        # Joint angles
        # -------------------------------------------------

        "left_elbow_angle":
            left_elbow_angle,

        "right_elbow_angle":
            right_elbow_angle,

        "left_knee_angle":
            left_knee_angle,

        "right_knee_angle":
            right_knee_angle,

        "mean_elbow_angle":
            mean_elbow_angle,

        "mean_knee_angle":
            mean_knee_angle,

        "elbow_angle_difference":
            elbow_angle_difference,

        "knee_angle_difference":
            knee_angle_difference,

        # -------------------------------------------------
        # Body dimensions
        # -------------------------------------------------

        "shoulder_width":
            shoulder_width,

        "hip_width":
            hip_width,

        "shoulder_to_hip":
            shoulder_to_hip,

        # -------------------------------------------------
        # Limb distances
        # -------------------------------------------------

        "wrist_distance":
            wrist_distance,

        "ankle_distance":
            ankle_distance,

        "left_arm_length":
            left_arm_length,

        "right_arm_length":
            right_arm_length,

        "left_leg_length":
            left_leg_length,

        "right_leg_length":
            right_leg_length,

        # -------------------------------------------------
        # Relative positions
        # -------------------------------------------------

        "left_wrist_relative_y":
            left_wrist_relative_y,

        "right_wrist_relative_y":
            right_wrist_relative_y,

        "left_knee_relative_y":
            left_knee_relative_y,

        "right_knee_relative_y":
            right_knee_relative_y,
    }

    return features