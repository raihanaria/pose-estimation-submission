import math


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def is_valid(value):
    return (
        value is not None
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def mean_valid(*values):
    valid_values = [
        v for v in values
        if is_valid(v)
    ]

    if not valid_values:
        return None

    return sum(valid_values) / len(valid_values)


# =========================================================
# MAIN CLASSIFIER
# =========================================================

def classify_person(features):
    """
    Rule-based pose-only activity classifier.

    Activities:
        Standing
        Shooting
        Walking
        Running
        Jumping
        Unknown

    The classifier uses only geometric features
    derived from human pose keypoints.
    """

    # =====================================================
    # GET FEATURES
    # =====================================================

    left_elbow = features.get(
        "left_elbow_angle"
    )

    right_elbow = features.get(
        "right_elbow_angle"
    )

    left_knee = features.get(
        "left_knee_angle"
    )

    right_knee = features.get(
        "right_knee_angle"
    )

    wrist_distance = features.get(
        "wrist_distance"
    )

    ankle_distance = features.get(
        "ankle_distance"
    )

    shoulder_to_hip = features.get(
        "shoulder_to_hip"
    )

    left_wrist_y = features.get(
        "left_wrist_relative_y"
    )

    right_wrist_y = features.get(
        "right_wrist_relative_y"
    )

    left_knee_y = features.get(
        "left_knee_relative_y"
    )

    right_knee_y = features.get(
        "right_knee_relative_y"
    )

    left_arm_length = features.get(
        "left_arm_length"
    )

    right_arm_length = features.get(
        "right_arm_length"
    )

    left_leg_length = features.get(
        "left_leg_length"
    )

    right_leg_length = features.get(
        "right_leg_length"
    )

    # =====================================================
    # DERIVED FEATURES
    # =====================================================

    mean_elbow = mean_valid(
        left_elbow,
        right_elbow
    )

    mean_knee = mean_valid(
        left_knee,
        right_knee
    )

    elbow_difference = None

    if (
        is_valid(left_elbow)
        and is_valid(right_elbow)
    ):

        elbow_difference = abs(
            left_elbow -
            right_elbow
        )

    knee_difference = None

    if (
        is_valid(left_knee)
        and is_valid(right_knee)
    ):

        knee_difference = abs(
            left_knee -
            right_knee
        )

    wrist_y_mean = mean_valid(
        left_wrist_y,
        right_wrist_y
    )

    knee_y_mean = mean_valid(
        left_knee_y,
        right_knee_y
    )

    # -----------------------------------------------------
    # Arm / leg proportion
    # -----------------------------------------------------

    arm_leg_ratio = None

    if (
        is_valid(left_arm_length)
        and is_valid(right_arm_length)
        and is_valid(left_leg_length)
        and is_valid(right_leg_length)
    ):

        total_arm = (
            left_arm_length
            + right_arm_length
        )

        total_leg = (
            left_leg_length
            + right_leg_length
        )

        if total_leg > 1e-6:

            arm_leg_ratio = (
                total_arm /
                total_leg
            )

    # -----------------------------------------------------
    # Wrist / shoulder proportion
    # -----------------------------------------------------

    wrist_shoulder_ratio = None

    shoulder_width = features.get(
        "shoulder_width"
    )

    if (
        is_valid(wrist_distance)
        and is_valid(shoulder_width)
        and shoulder_width > 1e-6
    ):

        wrist_shoulder_ratio = (
            wrist_distance /
            shoulder_width
        )

    # =====================================================
    # BASIC RELIABILITY
    # =====================================================

    useful_features = 0

    for value in [
        left_elbow,
        right_elbow,
        left_knee,
        right_knee,
        wrist_distance,
        ankle_distance,
        shoulder_to_hip,
    ]:

        if is_valid(value):
            useful_features += 1

    if useful_features < 3:

        return {
            "activity": "Unknown",
            "confidence": 0.0,
            "rule": "insufficient_features",
            "reason": (
                "Not enough reliable pose features"
            )
        }

    # =====================================================
    # RULE 1
    # OUT-OF-DISTRIBUTION / SQUATTING
    # =====================================================

    if is_valid(mean_knee):

        if mean_knee < 80:

            return {
                "activity": "Unknown",
                "confidence": 0.98,
                "rule": "out_of_distribution",
                "reason": (
                    "Extreme knee flexion; "
                    "pose does not match target activities"
                )
            }

    # =====================================================
    # RULE 2
    # JUMPING
    # =====================================================

    if is_valid(mean_knee):

        if 90 <= mean_knee < 140:

            return {
                "activity": "Jumping",
                "confidence": 0.94,
                "rule": "strong_knee_flexion",
                "reason": (
                    "Strong bilateral knee flexion"
                )
            }

    # =====================================================
    # RULE 3
    # RUNNING
    # =====================================================

    if (
        is_valid(mean_knee)
        and is_valid(elbow_difference)
    ):

        running_pose = (
            145 <= mean_knee <= 160.5
        )

        asymmetric_arms = (
            elbow_difference >= 45
        )

        if (
            running_pose
            and asymmetric_arms
        ):

            return {
                "activity": "Running",
                "confidence": 0.94,
                "rule": "running_pose",
                "reason": (
                    "Moderately flexed knees "
                    "with asymmetric arm configuration"
                )
            }

    # =====================================================
    # RULE 4
    # SHOOTING - OCCLUDED / ONE ARM VISIBLE
    # =====================================================

    one_elbow_missing = (
        (left_elbow is None)
        !=
        (right_elbow is None)
    )

    if (
        is_valid(mean_knee)
        and one_elbow_missing
    ):

        visible_elbow = (
            left_elbow
            if is_valid(left_elbow)
            else right_elbow
        )

        if (
            is_valid(visible_elbow)
            and 115 <= visible_elbow <= 150
            and mean_knee >= 165
        ):

            return {
                "activity": "Shooting",
                "confidence": 0.91,
                "rule": "shooting_one_arm_visible",
                "reason": (
                    "Extended visible arm with "
                    "upright lower-body posture"
                )
            }

    # =====================================================
    # RULE 5
    # SHOOTING - BOTH ARMS BENT
    # =====================================================

    if (
        is_valid(mean_elbow)
        and is_valid(mean_knee)
        and is_valid(arm_leg_ratio)
    ):

        bent_arm_shooting = (
            40 <= mean_elbow <= 110
            and mean_knee >= 161
            and arm_leg_ratio >= 0.43
        )

        if bent_arm_shooting:

            return {
                "activity": "Shooting",
                "confidence": 0.95,
                "rule": "shooting_both_arms_bent",
                "reason": (
                    "Both elbows are flexed in a "
                    "stable upright shooting posture"
                )
            }

    # =====================================================
    # RULE 6
    # SHOOTING - EXTENDED ARMS
    # =====================================================

    if (
        is_valid(mean_elbow)
        and is_valid(mean_knee)
        and is_valid(arm_leg_ratio)
        and is_valid(wrist_y_mean)
        and is_valid(knee_y_mean)
    ):

        extended_arm_shooting = (
            145 <= mean_elbow <= 170
            and mean_knee >= 165
            and arm_leg_ratio >= 0.50
            and abs(
                wrist_y_mean -
                knee_y_mean
            ) <= 10
        )

        if extended_arm_shooting:

            return {
                "activity": "Shooting",
                "confidence": 0.89,
                "rule": "shooting_extended_arms",
                "reason": (
                    "Extended-arm configuration with "
                    "upright lower body"
                )
            }

    # =====================================================
    # RULE 7
    # WALKING
    # =====================================================

    if (
        is_valid(mean_knee)
        and is_valid(wrist_y_mean)
        and is_valid(knee_y_mean)
    ):

        walking_lower_body = (
            mean_knee >= 145
        )

        arms_lower_than_knees = (
            wrist_y_mean -
            knee_y_mean
            > 10
        )

        if (
            walking_lower_body
            and arms_lower_than_knees
        ):

            return {
                "activity": "Walking",
                "confidence": 0.88,
                "rule": "walking_pose",
                "reason": (
                    "Extended legs with "
                    "lower arm position"
                )
            }

    # =====================================================
    # RULE 8
    # STANDING
    # =====================================================

    if is_valid(mean_knee):

        if mean_knee >= 160:

            return {
                "activity": "Standing",
                "confidence": 0.90,
                "rule": "extended_legs",
                "reason": (
                    "Both knees are predominantly extended"
                )
            }

    # =====================================================
    # FALLBACK
    # =====================================================

    return {
        "activity": "Unknown",
        "confidence": 0.0,
        "rule": "ambiguous_pose",
        "reason": (
            "Pose does not satisfy a sufficiently "
            "reliable activity rule"
        )
    }