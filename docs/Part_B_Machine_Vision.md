# II. METHODS

## B. Machine Vision Program Development

The machine vision system serves as the primary sensor for the collaborative robot, enabling it to detect incoming sheet metal workpieces on the conveyor belt and extract their precise position, orientation, and identity in real time. The software pipeline — implemented entirely in Python — is structured as a sequence of modular stages that process raw camera frames and output reliable coordinates directly to the robot controller over a real-time network protocol. All tuning parameters are externalised in a single YAML configuration file (`config/vision_config.yaml`), allowing every numerical threshold to be adjusted without modifying source code.

### 1) Edge Detection

The first objective of the vision pipeline is to isolate the workpiece from the background conveyor belt. A robust preprocessing sequence (`src/vision/preprocess.py`) is applied to each acquired frame to produce a clean binary mask:

- **Grayscale Conversion:** The 3-channel BGR image is converted to a single-channel grayscale image, reducing the data from three colour planes to one intensity channel and thereby lowering computational overhead.
- **Gaussian Blurring:** A Gaussian filter with a configurable kernel size (default 3×3) is applied to smooth out high-frequency noise, surface reflections, and minor imperfections on the sheet metal, ensuring stable edge extraction in subsequent steps.
- **Binary Thresholding:** A fixed-intensity binary threshold (default 140/250) segments the lighter sheet metal from the darker conveyor belt surface. Pixels whose intensity exceeds the threshold are set to the maximum value (white, representing the object), while the remaining background pixels are set to zero (black).
- **Morphological Closing:** A morphological *close* operation — dilation followed by erosion — is applied using a rectangular structuring element (default 7×7). This fills any small gaps, holes, or lighting artefacts within the object's mask, guaranteeing the workpiece is represented as a single contiguous region.

Following preprocessing, edge extraction is finalised by applying OpenCV's contour-finding algorithm (`cv2.findContours`) to the binary mask (`src/vision/detection.py`). Configured with `RETR_EXTERNAL` mode, the algorithm traces only the outermost boundaries of white regions, effectively extracting the continuous edges of the workpiece while ignoring any internal features.

### 2) Centroid Identification

Once the contours are extracted, the system identifies the workpiece by intelligently filtering out noise through a series of geometric gates. The detected contours are first evaluated by enclosed area; the largest contour is selected as the primary candidate for further analysis. This candidate then passes through the following validation stages:

- **Area Bounds:** The contour area must fall within a configurable range (default 10 000–200 000 px²). Contours outside this range — whether too small (dust, scratches) or too large (belt edges, lighting gradients) — are rejected.
- **Edge Proximity:** If a contour's axis-aligned bounding box lies within a configurable margin of the frame border (default 10 px), it is rejected. This prevents partially visible workpieces entering or exiting the field of view from producing inaccurate centroid readings.
- **Solidity:** The ratio of contour area to convex hull area must exceed a minimum threshold (default 0.60). This rejects irregular, non-solid shapes that are unlikely to be genuine workpieces.

To pinpoint the exact centre of the workpiece, the system computes the **image moments** of the selected contour. The centroid coordinates $(c_x, c_y)$ are derived from the zeroth- and first-order spatial moments as follows:

$$c_x = \frac{M_{10}}{M_{00}}, \qquad c_y = \frac{M_{01}}{M_{00}}$$

Using image moments rather than the centre of the bounding rectangle yields the true centre of mass of the contour, which is more accurate for asymmetric or irregular workpiece shapes. These centroid coordinates represent the exact centre of the sheet metal in the camera's field of view, serving as the foundational reference point for the robot's inverse kinematics and path planning.

### 3) Workpiece Orientation Calibration

In addition to the centroid, the system must determine the rotational orientation of the workpiece so the robot can rotate its end-effector to match. Rather than relying on the angle returned by the minimum-area bounding rectangle — which can be unstable for non-rectangular or irregular shapes — the pipeline employs **Principal Component Analysis (PCA)** on the contour boundary points.

PCA computes the eigenvectors of the covariance matrix of the contour point cloud. The first eigenvector points along the direction of maximum geometric spread — the principal axis of the workpiece. The orientation angle $\theta$ is then calculated as:

$$\theta = \arctan2(e_{1y},\; e_{1x})$$

where $e_{1x}$ and $e_{1y}$ are the components of the first eigenvector. This approach is mathematically robust for all convex and irregular shapes, unlike the rectangle-based angle which is sensitive to small perturbations in corner positions. By determining this angular deviation in real time, the vision system informs the collaborative robot of the exact orientation of the workpiece, allowing the cobot to dynamically rotate its end-effector before engaging the magnetic gripper.

### 4) Camera Calibration and Coordinate Mapping

While edge detection and centroid identification operate in the pixel domain of the camera's image sensor, the robot controller requires coordinates in real-world dimensions (millimetres) to execute path planning. To bridge this gap, two critical transformations are integrated into the vision pipeline:

**Intrinsic Camera Calibration.** Lens distortion correction is performed to eliminate barrel or pincushion distortion introduced by the wide-angle lens. By capturing multiple images of a known checkerboard pattern (default 9×6 inner corners, 25 mm square size), the system computes the 3×3 camera matrix and the five distortion coefficients using `cv2.calibrateCamera()`. For efficient per-frame operation, the undistortion mapping tables are precomputed once via `cv2.initUndistortRectifyMap()` and then applied to every incoming frame using `cv2.remap()`, avoiding the overhead of re-computing the correction at each iteration. The calibration result, including the RMS reprojection error, is serialised to `config/camera_calibration.json` and auto-loaded by the pipeline at startup.

**Homography Transformation.** A 3×3 perspective homography matrix maps undistorted 2D pixel coordinates to real-world 2D coordinates on the surface of the conveyor belt. By recording the pixel positions of at least four reference markers whose physical positions (in millimetres) are known, the system computes the homography using `cv2.findHomography()` with RANSAC outlier rejection. During real-time operation, the pipeline multiplies the detected workpiece centroid $(x, y)$ in pixels by this matrix via `cv2.perspectiveTransform()`, outputting the precise $(X, Y)$ position in millimetres relative to a defined origin point on the conveyor. The homography is stored in `config/homography.json` and auto-loaded at startup. Additionally, the pipeline can apply a configurable camera-to-robot origin offset $(\Delta X, \Delta Y)$ to translate from the camera coordinate frame to the robot's base frame.

**Region of Interest (ROI).** Before preprocessing, the frame is cropped to a configurable region of interest defined as fractional bounds (e.g., top 26 % – bottom 85 % of the frame height). This removes noisy edges, mounting hardware, or areas outside the belt from the detection pipeline, reducing false positives and lowering computational load. The detected centroid coordinates are offset back to the full-frame coordinate space after detection, so the ROI is transparent to all downstream transforms.

### 5) Confidence-Based Detection Filtering

The area and edge-proximity gates alone cannot prevent all false positives on an empty belt — reflections, lighting gradients, and belt texture can still produce contours that pass the initial checks. To address this, each candidate contour is subjected to additional geometric validation:

- **Aspect Ratio Range:** The width-to-height ratio of the rotated bounding box must fall between configurable bounds (default 0.3–10.0). Extremely elongated or excessively square contours are rejected.
- **Confidence Score:** A composite score in the range [0, 1] is computed as the average of two geometric metrics:
  - *Solidity* — contour area divided by convex hull area (1.0 indicates a perfectly convex shape).
  - *Rectangularity* — contour area divided by bounding box area (1.0 indicates a perfect rectangle).

Only detections whose confidence score exceeds a configurable threshold (default 0.60) are accepted; everything below is discarded, preventing the system from sending spurious coordinates to the robot. All thresholds are tuneable in `config/vision_config.yaml`.

### 6) Shape Classification

With multiple workpiece variants travelling the conveyor, the system must not only locate a part but also identify *which* part it is. A shape classification stage (`src/vision/shape_classifier.py`) is inserted immediately after contour detection to solve this.

At startup, the classifier loads a set of reference images (`Part_1.jpg`, `Part_2.jpg`, …) from a configurable directory (`data/reference_images/`), converts each to a binary mask, and extracts its largest contour. During real-time operation, whenever a valid contour passes all detection gates, the classifier compares it against every reference using OpenCV's `cv2.matchShapes()` function. This function computes the distance between **Hu Moment** signatures — seven shape descriptors that are mathematically invariant to translation, rotation, and scale. The reference with the lowest distance score is selected as the match.

A configurable match threshold (default 0.20) acts as a reject gate: if the best score exceeds this value, the part is labelled as *unknown*, preventing misclassification when an unexpected object appears. The resulting part identity (e.g., `Part_1`) is attached to the detection result and propagated through the rest of the pipeline — appearing in the live overlay, the terminal dashboard, and the data sent to the robot controller.

Because the classifier auto-discovers all `Part_*.jpg` / `Part_*.png` files in the reference directory, adding support for a new workpiece variant requires only placing a single reference photograph — no code changes are needed. The entire feature can be toggled via the `classification.enabled` flag in `config/vision_config.yaml`.

### 7) Detection Tracking and Debounce

On a real conveyor belt, per-frame detection results are inherently noisy: a workpiece may be detected on one frame, lost on the next due to a lighting fluctuation, and re-detected on the frame after. Without filtering, each of these transient events would trigger a separate coordinate send to the robot, resulting in duplicate or contradictory commands.

To solve this, a **detection tracker** (`src/vision/detection_tracker.py`) implements a three-state finite-state machine that governs when coordinates are actually transmitted:

| State | Transition | Description |
|-------|-----------|-------------|
| **IDLE** | $N$ consecutive detections → CONFIRMING | No part present; waiting for stable detections. |
| **CONFIRMING** | Confirmed → SENT | Accumulating consecutive detections. The centroid coordinates are averaged over the confirmation window to reduce jitter. |
| **SENT** | $M$ consecutive non-detections → IDLE | Coordinates have been sent exactly once. Further detections of the same part are suppressed until the part exits the field of view. |

The configurable parameters are:

- **Confirm frames** (default 10): consecutive detection frames required before sending.
- **Exit frames** (default 10): consecutive non-detection frames required before resetting to IDLE.
- **Distance threshold** (default 5 mm): if a new detection jumps farther than this distance from the locked position, it is treated as a brand-new part and the confirmation process restarts.

This mechanism guarantees that each physical part triggers exactly one coordinate transmission to the robot, regardless of per-frame detection jitter.

### 8) Robot Communication via RTDE

The final stage of the pipeline transmits the validated workpiece coordinates to the Universal Robots collaborative robot using the **Real-Time Data Exchange (RTDE)** protocol (`src/vision/RTDEsender.py`). RTDE provides a deterministic, low-latency communication channel that writes directly into the robot controller's input registers at 500 Hz, bypassing the overhead of script-based interfaces.

The vision system writes five double-precision registers per transmission:

| Register | Content | Unit |
|----------|---------|------|
| `input_double_register_0` | X position | metres (mm ÷ 1000) |
| `input_double_register_1` | Y position | metres (mm ÷ 1000) |
| `input_double_register_2` | Orientation angle | radians |
| `input_double_register_3` | Valid flag (1.0 = object present, 0.0 = no object) | — |
| `input_double_register_4` | Part ID number (e.g., 1 for `Part_1`) | — |

The RTDE connection is established at pipeline startup with automatic retry logic (default 5 attempts). If the connection drops during operation, the pipeline continues processing frames and attempts to reconnect periodically (every 10 frames). When no object is present on the belt and the tracker returns to IDLE, a configurable "no-object signal" is sent to inform the robot that the belt is clear.

### 9) Libraries and Dependencies

| Library | Purpose |
|---------|---------|
| **OpenCV** (`opencv-python`) | Core vision: image I/O, preprocessing, contour detection, PCA orientation, shape matching, camera calibration, homography, and live display overlay |
| **NumPy** (`numpy`) | Array operations for frames and coordinate transforms; required by OpenCV internally |
| **PyYAML** (`pyyaml`) | Parses the YAML configuration file so all parameters are editable without code changes |
| **Picamera2** (`picamera2`) | Captures frames from the Raspberry Pi Global Shutter Camera (Sony IMX296) via `libcamera`; imported conditionally and only required on the Pi |
| **RTDE** (`rtde`) | Universal Robots Real-Time Data Exchange client library; writes detection coordinates into the robot controller's input registers |
| **Rich** (`rich`) | Renders a live, multi-panel terminal dashboard displaying system status, detection metrics, performance graphs, and the coordinates sent to the robot |


### Vision Pipeline Flow

To visually summarise the execution flow described above, the following diagram illustrates the sequential operations performed on every incoming camera frame.

```text
┌──────────────────────────────────────────────────────────────────┐
│                        run_vision.py                             │
│                      (entry point)                               │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     pipeline.py                                  │
│              (main loop — orchestrator)                          │
│                                                                  │
│   Loads config + calibration files (if they exist)               │
│   Opens FrameSource ──→ Loop:                                    │
│                           │                                      │
│        ┌──────────────────┘                                      │
│        │                                                         │
│        ▼                                                         │
│   ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│   │ FrameSource │──▶│ undistort()  │──▶│    ROI crop      │     │
│   │             │   │ (if calib    │   │ (configurable    │     │
│   │ Video file  │   │  exists)     │   │  edge removal)   │     │
│   │ or Picam2   │   └──────────────┘   └───────┬──────────┘     │
│   │ or USB cam  │                              │                │
│   └─────────────┘                              ▼                │
│                     ┌──────────────┐   ┌──────────────────┐     │
│                     │ preprocess() │──▶│ detect_object()  │     │
│                     │              │   │                  │     │
│                     │ BGR→Gray     │   │ Find contours   │     │
│                     │ Gaussian blur│   │ Largest contour │     │
│                     │ Binary thresh│   │ Area bounds     │     │
│                     │ Morph close  │   │ Edge proximity  │     │
│                     └──────────────┘   │ Solidity gate   │     │
│                                        │ Moments centroid│     │
│                                        │ PCA orientation │     │
│                                        │ Aspect ratio    │     │
│                                        │ Confidence score│     │
│                                        │ → (x, y, θ) px  │     │
│                                        └───────┬──────────┘     │
│                                                │                │
│                                                ▼                │
│                                        ┌──────────────────┐     │
│                                        │ ShapeClassifier  │     │
│                                        │ (if enabled)     │     │
│                                        │                  │     │
│                                        │ Hu Moment match  │     │
│                                        │ vs Part_*.jpg    │     │
│                                        │ → part_id        │     │
│                                        └───────┬──────────┘     │
│                                                │                │
│                                                ▼                │
│                     ┌──────────────┐   ┌──────────────────┐     │
│                     │  compensate  │◀──│ pixel_to_world() │     │
│                     │  _belt       │   │ (if homography   │     │
│                     │  _motion()   │   │  exists)         │     │
│                     │ (if belt     │   │ + camera offset  │     │
│                     │  enabled)    │   │ → (X, Y, θ) mm  │     │
│                     │ → pick pos   │   └──────────────────┘     │
│                     └──────┬───────┘                            │
│                            │                                    │
│                            ▼                                    │
│                   ┌────────────────┐                            │
│                   │ Detection      │                            │
│                   │ Tracker        │                            │
│                   │                │                            │
│                   │ IDLE →         │                            │
│                   │ CONFIRMING →   │                            │
│                   │ SENT           │                            │
│                   │ (single-send   │                            │
│                   │  debounce)     │                            │
│                   └───────┬────────┘                            │
│                           │                                     │
│                           ▼                                     │
│            ┌──────────────────────────────┐                     │
│            │ RTDE Sender                  │                     │
│            │ send_pose(X, Y, θ, valid,    │                     │
│            │           part_id)           │                     │
│            │ → Robot input registers      │                     │
│            └──────────────────────────────┘                     │
│                           │                                     │
│                           ▼                                     │
│                   ┌────────────────┐                            │
│                   │ Draw overlay   │                            │
│                   │ • Contour      │                            │
│                   │ • Red centroid │                            │
│                   │ • Part label   │                            │
│                   │ • mm or px text│                            │
│                   │ cv2.imshow()   │                            │
│                   └────────────────┘                            │
└──────────────────────────────────────────────────────────────────┘

       ▲                           ▲                       ▲
       │ config                    │ auto-loaded           │ auto-loaded
       │                           │ if file exists        │ if file exists
┌──────┴───────┐   ┌──────────────┴────────┐   ┌─────────┴──────────┐
│ vision_      │   │ camera_calibration    │   │ homography.json    │
│ config.yaml  │   │ .json                 │   │                    │
│              │   │                       │   │ Pixel → mm         │
│ thresholds   │   │ Camera matrix +       │   │ transform matrix   │
│ ROI, belt,   │   │ distortion coeffs     │   │                    │
│ tracking,    │   │ (from checkerboard)   │   │ (from ref points)  │
│ classific.,  │   │                       │   │                    │
│ RTDE, display│   │                       │   │                    │
└──────────────┘   └───────────────────────┘   └────────────────────┘
```

### Vision Pipeline Output Visualisation

The side-by-side image below illustrates the result of the vision pipeline operating on a camera frame. On the right is the binary mask generated after grayscale conversion, Gaussian blur, thresholding, and morphological closing. On the left is the final output layer demonstrating the continuous edge detection and bounding box fitting algorithm. 

The coloured contour outline traces the actual detected boundary of the workpiece. At its centre lies the calculated red centroid — derived from image moments — alongside the normalised rotation angle computed via PCA, actively predicting exactly how the collaborative robot must approach and rotate to retrieve the targeted metallic sheet.

![Vision Pipeline Detection and Mask](/Users/zain/.gemini/antigravity/brain/cf73f009-3a05-4132-a74b-c8de63cdc12f/vision_result_demo.jpg)

