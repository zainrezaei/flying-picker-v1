# The Math Behind the Flying Picker Vision Pipeline

This document breaks down the mathematical concepts and algorithms powering the Flying Picker's vision system. It details how the software translates raw camera pixels into physical, millimeter-accurate coordinates for the robot to use.

---

## 1. Object Detection & Confidence Metrics
Before the system considers picking an object, it must verify that the blob of pixels it sees is actually a clean, solid geometric part and not just overlapping shadows or debris. It does this by scoring the geometry.

### Convex Hull & Solidity
Imagine stretching a tight rubber band around the outer boundary of the part. The smooth shape this rubber band makes is called the **Convex Hull**.
* **What it measures:** How "solid" or "bumpy" the perimeter is. If a contour has deep craters or gaps, its area will be much smaller than the rubber band wrapped around it.
* **Math:** `Solidity = Area(Contour) / Area(ConvexHull)`
* **The Reality:** A perfect rectangle has a solidity of exactly `1.0`. A wrench or a star-shape will have a low solidity (`~0.5`) because the hull bridges over the empty gaps.

### Bounding Rectangularity
We use OpenCV's `minAreaRect` to calculate the absolute smallest rectangle that can securely enclose the object.
* **What it measures:** How efficiently the object packs into a box. 
* **Math:** `Rectangularity = Area(Contour) / Area(BoundingBox)`
* **The Reality:** An ellipse will have a rectangularity of `0.78` ($\pi/4$). A rectangle will be `1.0`. A diagonal line would be extremely low.

### The Confidence Score
Instead of passing everything blindly to the robot arm, the pipeline combines these values into a strict 0.0 to 1.0 confidence score.
* **Math:** `Confidence = (Solidity + Rectangularity) / 2.0`
* If the score drops below the configured `min_confidence` threshold, the pipeline ignores it as noise, preventing the robot from attacking non-parts.

---

## 2. Center of Gravity & Rotation (Image Moments)
Originally, we used the rotated bounding box (`minAreaRect`) to figure out which way the part was pointing. However, for symmetrical shapes like squares, the bounding box algorithm can "snap" to the corners instead of the flat edges. This creates a severe "diamond effect" where the box wildly toggles its rotation between $0^\circ$ and $45^\circ$, causing chaotic robot behavior.

To solve this, we rely on **Spatial Image Moments** to calculate the physical mechanics of the shape.

### The Centroid (Center of Mass)
Moments (denoted as $m_{pq}$) act as a weighted mathematical average of every single pixel inside the shape. 
* **Area ($m_{00}$):** The total sum of all white pixels.
* **Centroid Math:** 
  $$ C_x = \frac{m_{10}}{m_{00}}, \quad C_y = \frac{m_{01}}{m_{00}} $$
* **Why it matters:** Unlike finding the midpoint of a bounding box (which gets skewed if a speck of dirt extends the box), the centroid guarantees the absolute physical balance point of the object. 

### Principal Axis of Inertia (The True Angle)
We extract the second-order central moments ($\mu_{20}, \mu_{02}, \mu_{11}$). These describe the shape's "pixel variance", or how effectively the mass is spread outwards from the centroid.
* $\mu_{20}$: How much the shape is spread horizontally.
* $\mu_{02}$: How much the shape is spread vertically.
* $\mu_{11}$: The covariance, or diagonal spread.

Using these, we can algebraically determine the **Principal Axis of Inertia**—the exact axis along which the object is longest:
* **Math:** 
  $$ \theta = \frac{1}{2} \arctan \left( \frac{2\mu_{11}}{\mu_{20} - \mu_{02}} \right) $$
* **Why it matters:** Because this relies on the internal density of the pixels rather than an arbitrary exterior box, the angle never suffers from the "corner snapping" bug. As the part smoothly rotates on the conveyor, the formula guarantees a perfectly stable rotation angle for the robot grip.

---

## 3. Pixel-to-World Translation (Homography)
Cameras suffer from **Perspective Distortion**. An object situated on the far back-right of the camera view is physically further away than an object directly underneath the lens. Because of this, 100 pixels in the center of the image represents a different physical millimeter distance than 100 pixels at the extreme edge of the image.

To fix this naturally occurring geometry problem, we apply a **Homography Transformation**.

### The Transformation Matrix
Homography mathematically projects our flat 2D camera pixel grid onto the flat 2D physical conveyor belt surface using a pre-calibrated $3 \times 3$ Transformation Matrix ($H$).

$$ \begin{bmatrix} X' \\ Y' \\ W \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} $$

### The Millimeter Result
Once multiplied, we divide by the $W$ scaling factor to get the real-world metrics.
* **Result:** $X_{mm} = X' / W$, and $Y_{mm} = Y' / W$.
* **Why it matters:** This algebraic flattening completely neutralizes camera angle and perspective. The pipeline can extract a coordinate at the extreme fuzzy edge of the camera frame, and the homography guarantees that coordinate maps safely to millimeter truth in the robot arm's coordinate space.

---

## 4. Belt Motion Compensation (Kinematics)
The vision system is not instantaneous. 
1. The camera captures the frame.
2. The Raspberry Pi runs the homography and detection math.
3. The coordinate output transmits across the network via RTDE.
4. The UR robot physically travels to the destination.

Because the conveyor belt is rolling continuously, the part travels further down the belt while all of this computation and travel is happening.

### Predictive Kinematics
We use linear kinematics to project the coordinate forward in time, telling the robot where the part *will* be, not where the camera saw it *was*.
* **Formula:** 
  $$ Y_{pick} = Y_{detect} + (V_{belt} \times t_{delay}) $$
* $V_{belt}$: Speed of the conveyor belt in mm/second.
* $t_{delay}$: The fixed sum representing (Camera Latency + Processing Time + Robot Travel Time).
* **Why it matters:** It generates a "future state coordinate" decoupled entirely from the raw camera detection. The system fires coordinates ahead of the part, allowing the robot to flawlessly rendezvous with the part mid-flight.

---

## 5. Shape Classification (Hu Moments)
To distinguish between Part 1, Part 2, etc., we calculate **Hu Moments**.

Hu Moments are seven mathematical constants derived from the central moments mentioned earlier. Their key feature is that they are mathematically **invariant**. 

### What is Invariance?
"Invariant" simply means "does not change". Just like your body weight is *rotation invariant* (you weigh the same whether you face North or South), Hu Moments do not change based on physical alterations in the camera view:
1. **Translation Invariant:** The numbers are exactly the same if the item is on the left side of the belt vs. the right side of the belt. Position doesn't matter.
2. **Scale Invariant:** The numbers are exactly the same if the camera is extremely close up or very far away. Size doesn't matter.
3. **Rotation Invariant:** The numbers are exactly the same if the part lands perfectly straight or rotated entirely diagonally. Angle doesn't matter.

### What are the 7 Numbers?
The 7 numbers ($h_1$ through $h_7$) are complex statistical values that represent the pixel mass distribution of the shape.
1. **Hu Moment 1 (Spread):** The general moment of inertia. It measures how far, on average, the pixels are spread out from the center of gravity.
2. **Hu Moment 2 (Elongation):** Measures how "stretched out" or elongated the shape is overall.
3. **Hu Moments 3, 4, 5, & 6 (Asymmetry/Skew):** These measure the balance of the shape. A teardrop, with its heavy round bottom and light pointy top, has an asymmetrical weight distribution that these middle moments capture.
4. **Hu Moment 7 (Mirror Invariant):** This moment specifically measures mirror symmetry. If you take an asymmetrical part and flip it completely upside-down on the belt, the first six moments will remain identical, but **Hu Moment 7 will flip its sign** (e.g., `-0.005` to `+0.005`). This tells the computer if the part is laying face down!

### The Classification Pipeline
When a live contour is detected, the pipeline instantly calculates its 7 Hu Moments. It then compares these numbers to the saved reference moments of your known parts using an OpenCV logarithmic absolute-difference metric. 
* Because of the invariance properties, you only ever need to save **one** reference snapshot of the part.
* If the difference between the live numbers and the reference numbers falls below a tight threshold (e.g., `< 0.2`), the pipeline confidently declares a match, regardless of how the part fell onto the belt!
