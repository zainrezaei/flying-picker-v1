# The Math Behind the Flying Picker Vision Pipeline

This document breaks down the mathematical concepts and algorithms powering the Flying Picker's vision system. It details how the software translates raw camera pixels into physical, millimeter-accurate coordinates for the robot to use.

---

## 1. Object Detection & Confidence Metrics
Before sending anything to the robot, the system evaluates how "perfect" an object looks using geometric comparisons. 

### Convex Hull & Solidity
Imagine stretching a tight rubber band around the outer boundary of an object. The shape this rubber band makes is called the **Convex Hull**.
* **Solidity** measures how solid or "bumpy" the contour is.
* **Math:** `Solidity = Area(Contour) / Area(ConvexHull)`
* **Example:** A perfect rectangle has a solidity of 1.0 (100%). A star shape has a low solidity because the hull stretches over its points, leaving empty gaps.

### Bounding Rectangularity
We calculate the smallest possible bounding rectangle (`minAreaRect`) that fits around the object.
* **Rectangularity** measures how well the object fills this box.
* **Math:** `Rectangularity = Area(Contour) / Area(BoundingBox)`

### Final Confidence Score
The pipeline combines these to generate a 0.0 to 1.0 confidence score representing how closely the blob resembles a geometric, pickable solid object.
* **Score:** `Confidence = (Solidity + Rectangularity) / 2.0`

---

## 2. Center of Gravity & Rotation (Image Moments)
Using a bounding box for the object's rotation angle can cause a major issue: on symmetrical shapes like squares (or near-squares), the bounding box can sometimes snap to the flat edges, and sometimes snap to the *corners*, creating a "diamond" effect where the angle wildly jumps between 0° and 45°.

To solve this, we use **Image Moments** to calculate the object's true Principal Axis of Inertia.

### The Centroid (Center of Mass)
Moments ($m_{pq}$) act as a weighted average of pixel intensities over the shape. 
* **Area ($m_{00}$):** The total sum of all pixels.
* **Centroid:** 
  $$ C_x = \frac{m_{10}}{m_{00}}, \quad C_y = \frac{m_{01}}{m_{00}} $$
This guarantees the exact balance point of the object, completely immune to noisy contours.

### Principal Axis of Inertia (The True Angle)
We extract the second-order central moments ($\mu_{20}, \mu_{02}, \mu_{11}$), which measure the variance—or how the shape's "pixel mass" is spread outwards from the centroid.
* $\mu_{20}$: Horizontal spread.
* $\mu_{02}$: Vertical spread.
* $\mu_{11}$: Diagonal spread (covariance).

Using these, we accurately determine the axis along which the object is longest:
* **Math:** 
  $$ \theta = \frac{1}{2} \arctan \left( \frac{2\mu_{11}}{\mu_{20} - \mu_{02}} \right) $$

**Result:** The rotation angle ($\theta$) stays perfectly stable as the object rotates, completely resolving the spinning bounding-box bug.

---

## 3. Pixel-to-World Translation (Homography)
Cameras have perspective distortion. An object on the far right of the camera frame is further away than an object exactly in the center. We cannot simply multiply pixels by a standard mm/pixel ratio.

**Homography** solves this by projecting the 2D pixel plane onto the 2D physical conveyor plane using a $3 \times 3$ Transformation Matrix ($H$). 

$$ \begin{bmatrix} X' \\ Y' \\ W \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} $$

* **Pixel Coordinates:** $(x, y)$
* **World Coordinates (mm):** $X_{mm} = X' / W$, and $Y_{mm} = Y' / W$.

This transformation flattens perspective, meaning coordinates generated on the extreme edge of the camera accurately reflect the real-world belt millimeters.

---

## 4. Belt Motion Compensation (Kinematics)
The vision system processes frames with a slight delay, and the robot physically takes time to move to the coordinate. Because the conveyor belt is continuously moving, the part travels in the Y direction while the robot moves.

We use basic linear kinematics to project the parts coordinates forward in time:
* **Formula:** 
  $$ Y_{pick} = Y_{detect} + (V_{belt} \times t_{delay}) $$
* $V_{belt}$: Speed of the conveyor belt in mm/second.
* $t_{delay}$: The time offset representing Camera Latency + Processing Time + Robot Travel Time.

This generates a "predicted coordinate" completely decoupled from where the camera actually saw the part, giving the robot the correct destination target.

---

## 5. Shape Classification (Hu Moments)
To distinguish between Part 1, Part 2, etc., we calculate **Hu Moments**.

Hu Moments are seven mathematical constants derived from the central moments mentioned earlier. Their key feature is that they are **invariant** to:
1. **Scale** (Distance from camera)
2. **Translation** (Position on belt)
3. **Rotation** (Angle of the part)

When a contour is detected, the pipeline compares its 7 Hu Moments to our saved reference images using a logarithmic absolute-difference metric. 
* If the difference is below our threshold (e.g., `< 0.2`), it declares a match, regardless of how the part fell onto the belt!
