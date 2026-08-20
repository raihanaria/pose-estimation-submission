# Human Pose & Action Classification

**Computer Vision AI Engineer Assessment**

A pose-based human activity classification system using **YOLO11n-Pose**
for human pose estimation and a **rule-based classifier** based on
geometric features derived from human pose keypoints.

The system is designed to classify the following target activities:

-   Standing
-   Shooting
-   Walking
-   Running
-   Jumping

The activity classification is performed using human pose keypoints and
their derived geometric features. No image-based action recognition
model is used for the activity classification stage.

------------------------------------------------------------------------

## 1. Approach

### 1.1 System Overview

The system is divided into two main stages:

1.  **Human Pose Estimation**
2.  **Pose-Based Action Classification**

Overall pipeline:

``` text
Input Image
    |
    v
YOLO11n-Pose
    |
    v
Person Detection + 17 COCO Keypoints
    |
    v
Keypoint Confidence Filtering
    |
    +-----------------------------+
    |                             |
Reliable Keypoints        Unreliable Keypoints
    |                             |
    |                       Treated as Missing
    |                             |
    +-------------+---------------+
                  |
                  v
          Feature Extraction
                  |
                  v
         Geometric Pose Features
                  |
                  v
          Rule-Based Classifier
                  |
                  v
          Activity + Confidence
                  |
                  v
        Annotated Image + JSON
```

### 1.2 Pose Estimation Model

The selected pose estimation model is:

-   **Model:** YOLO11n-Pose
-   **Model weight:** `yolo11n-pose.pt`
-   **Framework:** Ultralytics YOLO

YOLO11n-Pose was selected because it provides human pose estimation
together with person detection and directly outputs the standard **17
COCO human body keypoints** required by the assessment.

The 17 keypoints are:

| ID | Keypoint Name | ID | Keypoint Name |
|:--:|:--------------|:--:|:--------------|
| 0  | Nose          | 9  | Left Wrist    |
| 1  | Left Eye      | 10 | Right Wrist   |
| 2  | Right Eye     | 11 | Left Hip      |
| 3  | Left Ear      | 12 | Right Hip     |
| 4  | Right Ear     | 13 | Left Knee     |
| 5  | Left Shoulder | 14 | Right Knee    |
| 6  | Right Shoulder| 15 | Left Ankle    |
| 7  | Left Elbow    | 16 | Right Ankle   |
| 8  | Right Elbow   | -- | --            |

For every detected person, the pose model provides:

-   Keypoint coordinates `(x, y)`
-   Keypoint confidence
-   Person bounding box
-   Pose information for the 17 keypoints

### 1.3 Keypoint Confidence and Reliability

The system does not blindly use every keypoint returned by the pose
estimator.

A keypoint confidence threshold is applied:

``` text
KEYPOINT_CONF = 0.35
```

A keypoint is considered reliable only when its confidence is equal to
or above the threshold.

Conceptually:

``` python
if confidence >= KEYPOINT_CONF:
    keypoint = valid
else:
    keypoint = missing
```

Therefore, low-confidence keypoints are treated as unreliable/missing
rather than being included as valid pose measurements.

For example:

``` text
left_elbow_angle  : MISSING
right_elbow_angle : 132.209
```

This means that the left elbow could not be reliably estimated, while
the right elbow was available.

This approach prevents unreliable keypoints from producing incorrect
geometric measurements.

### 1.4 Handling Occlusion and Imperfect Pose Estimation

The system is designed with robustness to partial occlusion and
imperfect pose estimation in mind.

When one or more keypoints cannot be reliably detected:

1.  The keypoint is marked as missing/unreliable.
2.  Features requiring the missing keypoint are not calculated.
3.  Other reliable keypoints are still used.
4.  Classification rules are only applied when sufficient pose
    information is available.
5.  If the pose does not provide enough reliable information, the
    classifier can return `Unknown`.

This behaviour is intentional. The system therefore does not attempt to
force an activity prediction when the available pose evidence is
insufficient.

For example, a person may have reliable lower-body keypoints while the
elbows and wrists are unavailable because of occlusion. The available
features can still be used for classification, while features that
depend on missing keypoints remain unavailable.

------------------------------------------------------------------------

## 2. Action Classification

### 2.1 Classification Method

The action classifier is a:

> Rule-based classifier using engineered geometric features derived from
> human pose keypoints.

The classifier does not directly analyze RGB image appearance.

Instead, the classification process is:

``` text
17 Pose Keypoints
        |
        v
Geometric Feature Extraction
        |
        v
Rule-Based Classification
        |
        v
Activity Label + Confidence
```

A rule-based approach was selected because the assessment allows
geometric/rule-based classification and the provided test set does not
constitute a sufficiently large labelled training dataset for training a
dedicated action recognition model.

The main advantage of this approach is interpretability. Each prediction
can be traced back to a specific geometric configuration of the human
pose.

### 2.2 Keypoint-Derived Features

Several geometric features are extracted from the 17 pose keypoints.

#### Joint Angles

**Elbow angles**

-   `left_elbow_angle`
-   `right_elbow_angle`

The elbow angle is calculated from:

``` text
Shoulder -> Elbow -> Wrist
```

Elbow angles are useful for distinguishing upper-body configurations,
particularly for shooting and running.

**Knee angles**

-   `left_knee_angle`
-   `right_knee_angle`

The knee angle is calculated from:

``` text
Hip -> Knee -> Ankle
```

Knee angles are useful for distinguishing:

-   Standing
-   Walking
-   Running
-   Jumping
-   Strongly crouched or unsupported poses

#### Aggregate Angular Features

The classifier also uses aggregated and asymmetric measurements such as:

-   `mean_elbow_angle`
-   `mean_knee_angle`
-   `elbow_angle_difference`
-   `knee_angle_difference`

The mean angle provides a summary of the overall pose, while the
left/right difference captures asymmetry between both sides of the body.

Upper-body asymmetry is particularly useful for differentiating running
from some static upright poses.

#### Body Dimensions

The system also calculates body geometry such as:

-   `shoulder_width`
-   `hip_width`
-   `shoulder_to_hip`

These features provide relative body structure and scale information.

#### Limb Lengths and Distances

The following features are derived from the pose:

-   `wrist_distance`
-   `ankle_distance`
-   `left_arm_length`
-   `right_arm_length`
-   `left_leg_length`
-   `right_leg_length`

These features describe the relative configuration of the arms and legs.

They are also used to construct relative proportions such as an
arm-to-leg length ratio.

#### Relative Vertical Positions

The system also uses relative vertical positions:

-   `left_wrist_relative_y`
-   `right_wrist_relative_y`
-   `left_knee_relative_y`
-   `right_knee_relative_y`

These features help describe the relative position of the upper and
lower limbs and are useful for distinguishing different body
configurations.

### 2.3 Activity Classification Rules

The classifier evaluates several geometric rules in an ordered sequence.

Target activities:

-   Standing
-   Shooting
-   Walking
-   Running
-   Jumping

An `Unknown` class is also available when the pose does not provide
sufficient or sufficiently distinctive evidence.

#### Standing

Standing is primarily identified from an upright lower-body
configuration.

A representative condition is:

``` text
mean_knee_angle >= 160 degrees
```

when sufficient pose information is available.

The rule is intentionally evaluated after more distinctive activities so
that poses with extended knees but stronger activity-specific
characteristics can be classified first.

#### Shooting

Shooting requires multiple rules because the same action can produce
different pose configurations depending on body orientation, camera
perspective, and occlusion.

**Shooting Pattern 1: One Visible Elbow**

This rule handles cases where one elbow is missing/unreliable.

Conceptually:

``` text
one elbow missing
AND
visible elbow angle approximately 115–150 degrees
AND
mean knee angle >= 165 degrees
```

This allows the classifier to identify a shooting pose even when one
upper-limb keypoint is unavailable.

**Shooting Pattern 2: Both Arms Bent**

This rule handles shooting poses where both elbows are detected.

``` text
40 <= mean_elbow_angle <= 110
AND
mean_knee_angle >= 161
AND
arm_to_leg_ratio >= 0.43
```

This rule captures upright shooting configurations with flexed elbows.

The arm-to-leg ratio provides additional geometric information so that
the classifier does not rely only on elbow angle.

**Shooting Pattern 3: Extended Arms**

Some shooting poses can have relatively extended elbows.

For these configurations:

``` text
145 <= mean_elbow_angle <= 170
AND
mean_knee_angle >= 165
AND
arm_to_leg_ratio >= 0.50
```

Additional relative vertical-position information is also considered.

The multiple shooting rules were introduced because a single condition
such as "one elbow must be missing" was too restrictive for poses where
both elbows were successfully detected.

#### Walking

Walking is identified from an upright/moderately extended lower-body
configuration together with the relative position of the arms.

The classifier considers:

``` text
mean_knee_angle >= 145 degrees
```

together with the relative wrist/knee configuration.

Walking is evaluated separately from running because a single static
pose can provide only limited information about temporal motion.

#### Running

Running is identified using a combination of knee flexion and upper-body
asymmetry.

A representative rule is:

``` text
145 <= mean_knee_angle <= 160.5
AND
elbow_angle_difference >= 45 degrees
```

The combination represents a moderately flexed lower body and an
asymmetric arm configuration.

This combination helps distinguish running from upright standing poses.

#### Jumping

Jumping is primarily identified through stronger knee flexion.

A representative condition is:

``` text
90 <= mean_knee_angle < 140 degrees
```

This is intended to distinguish the jumping pose from predominantly
extended-leg activities.

#### Unknown

The classifier returns:

``` text
Unknown
```

when:

-   Too few keypoints are reliable
-   Required geometric features cannot be calculated
-   The pose is ambiguous
-   The pose does not sufficiently match the target activity classes

This is preferable to forcing a prediction based on unreliable evidence.

`Unknown` can also occur for poses outside the target activity
categories.

------------------------------------------------------------------------

## 3. Assumptions and Limitations

### 3.1 Pose-Only Classification

The action classifier uses only human pose keypoints and features
derived from them.

It does not intentionally use:

-   RGB appearance
-   Clothing
-   Background
-   Object recognition
-   Scene context
-   Image texture
-   Other image-level semantic information

This keeps the action classification within the pose-based constraint of
the assessment.

### 3.2 Single-Image Classification

The current implementation classifies activities independently for each
image.

Temporal information is not currently used.

Therefore, activities such as:

-   Walking
-   Running
-   Jumping

are classified from a single pose rather than from motion across
multiple frames.

This is an important limitation because temporal information could
improve the distinction between activities that have similar static
poses.

### 3.3 Rule-Based Thresholds

The classification thresholds were selected empirically based on the
pose geometry observed in the provided assessment images.

Therefore, these thresholds should not be interpreted as universal
thresholds for human activity recognition.

For a production system, the thresholds should be evaluated and tuned
using a larger labelled validation dataset.

### 3.4 Occlusion and Missing Keypoints

Partial occlusion can cause pose keypoints to become unavailable.

The system handles this by treating low-confidence keypoints as missing.

However, if too many keypoints are unavailable, reliable classification
may not be possible.

In such situations, the system prefers:

``` text
Unknown
```

rather than forcing an unreliable prediction.

### 3.5 Similar Pose Configurations

Some activities can produce similar static configurations.

For example:

-   Standing and some shooting poses can both have extended knees.
-   Walking and running can overlap in individual static frames.
-   Some poses can contain only partial information due to occlusion.

For this reason, the classifier uses combinations of multiple geometric
features instead of relying on a single keypoint or threshold whenever
possible.

### 3.6 Unsupported Activities

The target activity classes are:

-   Standing
-   Shooting
-   Walking
-   Running
-   Jumping

The system does not attempt to recognize every possible human activity.

A pose outside these target classes may therefore be returned as:

``` text
Unknown
```

For example, a strongly crouched or squatting pose may not correspond to
any of the specified target activities.

------------------------------------------------------------------------

## 4. Reproducibility

### 4.1 Environment

The project is implemented in Python.

Recommended environment:

-   Python 3.10+
-   Windows / Linux

### 4.2 Dependencies

The main libraries used are:

-   Ultralytics
-   OpenCV
-   NumPy

Install the dependencies using:

``` bash
pip install ultralytics opencv-python numpy
```

If a `requirements.txt` file is provided:

``` bash
pip install -r requirements.txt
```

### 4.3 Installation

Clone or copy the project repository:

``` bash
git clone https://github.com/raihanaria/pose-estimation-submission.git
cd pose-estimation-submission
```

Create a virtual environment:

``` bash
python -m venv venv
```

Activate the environment on Windows:

``` bash
venv\Scripts\activate
```

Install the dependencies:

``` bash
pip install -r requirements.txt
```

### 4.4 Model Weights

The following model weight is required:

``` text
yolo11n-pose.pt
```

Place it in the project root or the location configured in `main.py`.

Example:

``` text
project/
├── yolo11n-pose.pt
├── main.py
├── feature_extraction.py
├── action_classifier.py
├── visualization.py
├── requirements.txt
└── README.md
```

------------------------------------------------------------------------

## 5. How to Run

### 5.1 Prepare Input Images

Place the input images inside:

``` text
test_set/
```

Example:

``` text
test_set/
├── image6.jpeg
├── image7.jpeg
├── images.jpeg
├── images2.jpeg
├── images3.jpeg
├── images4.jpeg
└── images5.jpeg
```

The application supports common image formats such as:

-   `.jpg`
-   `.jpeg`
-   `.png`

### 5.2 Run the Application

From the project root:

``` bash
python main.py
```

The program will:

1.  Load the YOLO11n-Pose model.
2.  Read the input images.
3.  Detect people.
4.  Extract the 17 human pose keypoints for each detected person.
5.  Apply keypoint confidence filtering.
6.  Treat unreliable keypoints as missing.
7.  Calculate pose-derived geometric features.
8.  Apply the rule-based activity classifier.
9.  Generate annotated images.
10. Save structured JSON results.

------------------------------------------------------------------------

## 6. Expected Output

The application produces annotated images and structured results.

A typical output structure is:

``` text
outputs/
├── annotated/
├── results/
└── submission/
```

### 6.1 Annotated Images

Annotated images contain information such as:

-   Person bounding box
-   Pose skeleton
-   Reliable keypoints
-   Person ID
-   Predicted activity
-   Activity confidence

Example:

``` text
Person #1: Shooting (0.95)
```

The annotations allow the pose estimation and activity classification
results to be visually inspected.

### 6.2 JSON Results

The system also stores structured results for each processed image.

A simplified example is:

``` json
{
  "image": "image7.jpeg",
  "persons": [
    {
      "person_id": 1,
      "bbox": [100, 50, 250, 400],
      "keypoints": [
        {
          "id": 0,
          "name": "nose",
          "x": 150.2,
          "y": 80.4,
          "confidence": 0.91,
          "reliable": true
        }
      ],
      "features": {
        "left_elbow_angle": 44.63,
        "right_elbow_angle": 44.63,
        "mean_knee_angle": 162.42
      },
      "classification": {
        "activity": "Shooting",
        "confidence": 0.95
      }
    }
  ]
}
```

The exact JSON structure depends on the implementation in `main.py`.

Unreliable keypoints are represented as missing/unreliable rather than
being reported as valid detections.

------------------------------------------------------------------------

## 7. Example Classification Output

An example terminal output is:

``` text
======================================================================
Processing: image6.jpeg
Person #1: Standing (0.90)
Person #2: Standing (0.90)
======================================================================
Processing: image7.jpeg
Person #1: Shooting (0.95)
Person #2: Shooting (0.95)
Person #3: Shooting (0.95)
Person #4: Shooting (0.95)
Person #5: Shooting (0.95)
Person #6: Shooting (0.95)
Person #7: Shooting (0.95)
Person #8: Shooting (0.89)
Person #9: Standing (0.90)
======================================================================
Processing: images.jpeg
Person #1: Shooting (0.91)
Person #2: Unknown (0.00)
======================================================================
Processing: images2.jpeg
Person #1: Running (0.94)
Person #2: Running (0.94)
======================================================================
Processing: images3.jpeg
Person #1: Unknown (0.98)
======================================================================
Processing: images4.jpeg
Person #1: Walking (0.88)
Person #2: Walking (0.88)
Person #3: Unknown (0.00)
Person #4: Standing (0.90)
======================================================================
Processing: images5.jpeg
Person #1: Jumping (0.94)
```

The exact predictions can vary depending on the model version,
environment, and implementation thresholds.

------------------------------------------------------------------------

## 8. Project Structure

A recommended project structure is:

``` text
project/
├── main.py
├── feature_extraction.py
├── action_classifier.py
├── visualization.py
├── requirements.txt
├── README.md
├── yolo11n-pose.pt
│
├── test_set/
│   ├── image6.jpeg
│   ├── image7.jpeg
│   ├── images.jpeg
│   ├── images2.jpeg
│   ├── images3.jpeg
│   ├── images4.jpeg
│   └── images5.jpeg
│
└── outputs/
    ├── annotated/
    ├── results/
    └── submission/
```

------------------------------------------------------------------------

## 9. Assessment Compliance

The implementation is designed to address the main requirements of the
assessment.

### Pose Estimation

-   Uses a human pose estimation model.
-   Detects the 17 COCO human pose keypoints.
-   Provides keypoint coordinates.
-   Uses keypoint confidence values.
-   Applies confidence filtering.
-   Treats insufficient-confidence keypoints as missing/unreliable.
-   Avoids intentionally reporting unreliable keypoints as valid
    detections.

### Action Classification

-   Uses human pose keypoints as the basis of activity classification.
-   Uses engineered geometric features.
-   Uses a transparent rule-based classifier.
-   Does not use a separate image-based action recognition model.

### Robustness

The pipeline is designed to handle partial occlusion and imperfect pose
estimation.

When a keypoint is unreliable:

``` text
Low-confidence keypoint
        |
        v
   Missing/None
        |
        v
Feature unavailable
        |
        v
Use remaining reliable pose information
        |
        v
If insufficient
        |
        v
     Unknown
```

This prevents the system from forcing an activity classification from
unreliable pose information.

------------------------------------------------------------------------

## 10. Design Rationale

The system intentionally separates pose estimation from activity
classification.

Instead of directly predicting an action from the RGB image:

``` text
RGB Image
    |
    v
Action Recognition Model
    |
    v
Activity
```

the proposed approach uses:

``` text
RGB Image
    |
    v
Pose Estimation
    |
    v
17 Keypoints
    |
    v
Geometric Features
    |
    v
Interpretable Rules
    |
    v
Activity
```

This makes the classification process easier to inspect and debug.

For example, a shooting prediction can be investigated through its
underlying pose features such as:

-   `mean_elbow_angle`
-   `mean_knee_angle`
-   `arm_to_leg_ratio`
-   Wrist position

rather than treating the prediction as a black-box image classification
result.

------------------------------------------------------------------------

## 11. Limitations and Future Improvements

The current implementation is intended for the provided assessment and
is not a complete production-level human activity recognition system.

Potential improvements include:

1.  Collecting a larger labelled dataset.
2.  Creating a dedicated training and validation split.
3.  Learning the classification thresholds automatically.
4.  Training a classifier using pose-derived features.
5.  Adding temporal pose information for video-based recognition.
6.  Adding person tracking across frames.
7.  Performing quantitative robustness evaluation under different
    occlusion levels.
8.  Calibrating activity confidence scores.
9.  Evaluating precision, recall, F1-score, and confusion matrix.
10. Testing the system across different camera angles, body sizes,
    lighting conditions, and occlusion patterns.

A temporal model could be particularly useful for distinguishing
activities such as walking, running, and jumping because those
activities contain motion information that cannot always be inferred
reliably from a single image.

------------------------------------------------------------------------

## 12. Conclusion

This project implements a pose-first human activity classification
pipeline using **YOLO11n-Pose** and a **rule-based geometric
classifier**.

The main design principles are:

``` text
Pose Estimation
      +
Confidence Filtering
      +
Missing Keypoint Handling
      +
Geometric Feature Engineering
      +
Rule-Based Classification
```

The system focuses on five target activities:

-   Standing
-   Shooting
-   Walking
-   Running
-   Jumping

while allowing:

``` text
Unknown
```

when the pose does not provide sufficient or sufficiently reliable
evidence.

The approach prioritizes interpretable pose-based reasoning and
robustness to imperfect pose estimation rather than forcing a prediction
from unreliable keypoints.
