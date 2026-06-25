Below is the comprehensive and extremely detailed plan covering traffic theory, kinematics mathematics, and Blender API programming techniques.
*https://docs.blender.org/*

Note on **"not observing the area inside the intersection"**. Technically, this turns the intersection into a **Black Box**. Vehicles will disappear at the In-camera, undergo a Time Delay proportional to their speed, then reappear at the Out-camera. This eliminates the need to model complex turning maneuvers, but requires extremely precise timing calculations for reappearance.

$$\text{Total width of one road axis} = (4 \text{ lanes} \times 3.5\text{ m}) + 2.0\text{ m (Median)} + (4 \text{ lanes} \times 3.5\text{ m}) = 30\text{ meters}$$
Intersection Box Dimensions: The intersection area will be a square measuring 30m x 30m

---

## PHASE 1: TRAFFIC KINEMATICS MATHEMATICS AND LOGIC

*Objective: Master the physical rules of vehicle movement to write accurate motion code.*

### 1.1. Visibility Boundaries (Stop Lines)

* **In-Camera:** Only records the approach road from afar up to the **Red Light Stop Line**.
* **Out-Camera:** Only records the road starting from the **Opposite Crosswalk** extending outward.
* **Blind Zone:** The intersection area where traffic streams cross.

### 1.2. Blind Zone Delay Time Formula

When a vehicle reaches the end of the incoming lane, it disappears. The time the vehicle is "invisible" inside the intersection before appearing at the outgoing lane is calculated using the basic kinematics formula:

$$\Delta t = \frac{d_{\text{intersection}}}{v}$$

* $v$: Vehicle speed when traversing the intersection (typically steady-state speed, as vehicles do not stop mid-intersection unless stuck in traffic).
* **Example:** A vehicle traveling at $40 \text{ km/h} \approx 11.1 \text{ m/s}$ crossing a $16\text{m}$ wide intersection will take approximately $1.44\text{ s}$. At 24 fps, the vehicle must be "invisible" for exactly 35 frames before appearing in the output video.

---

## PHASE 2: BLENDER FOUNDATIONS FOR PROGRAMMERS

*Objective: Understand how Blender manages data as code to manipulate it without opening the UI.*

### 2.1. Blender Data Structure (`bpy.data`)

Blender manages everything as Data-blocks. You need to master 3 concepts:

* **Objects (`bpy.data.objects`):** Entities with position and rotation in space (Camera, Vehicle, Road surface).
* **Meshes (`bpy.data.meshes`):** Core geometry (car body, license plate) contained within an Object.
* **Materials (`bpy.data.materials`):** Materials defining color, glossiness, image texture (Texture) of the license plate.

### 2.2. Coordinate System and Units

* Blender defaults to a **Right-Handed Z-Up** coordinate system (X-axis points right, Y-axis points backward, Z-axis points up).
* Default units are **Meters** and **Radians** (for rotation angles). You must always use `math.radians(degrees)` when writing code to rotate cameras or vehicles.

---

## PHASE 3: MASTERING THE BLENDER PYTHON API (`bpy`)

*Objective: Write scripts to automate the entire data lifecycle.*

### 3.1. Keyframing Techniques via Code

To make a vehicle move, you don't randomly change its position — you assign coordinates to the Timeline and select an Interpolation type:

* **Linear (Constant velocity):** Used for vehicles traveling at steady speed.
* **Bezier (Smooth):** Used for vehicles decelerating at stop lines or accelerating when leaving the intersection.

### 3.2. Dynamic Texture Mapping (Automatic License Plate Swapping)

To support LPR, each spawned vehicle must have a unique license plate.

* **Logic:** The code will find the material named `LicensePlate_Mat`, access its `Image Texture` node, and load a new license plate image file before rendering the next frame.

---

## PHASE 4: CAMERA POSITIONING

*Objective: Configure cameras to eliminate graphical blind zones and force a viewing angle similar to real CCTV.*

### 4.1. Camera Optics (Focal Length)

* This is characteristic of a **Telephoto Lens**.
* **Code Configuration:** Set the property `camera.data.lens = 60` or `85` (instead of the default 35). A longer focal length flattens the space, making license plates appear clearer — highly beneficial for LPR model testing.

### 4.2. Frustum Clipping

To ensure the camera only sees the incoming/outgoing lane and not the intersection area:

* Position the camera behind the stop line.
* Adjust the pitch angle just enough so the bottom edge of the video frame aligns exactly with the stop line (for In-Camera) or the crosswalk (for Out-Camera).

---

## PHASE 5: BUILDING A COMPLETE DATA PIPELINE

*Objective: Create a closed-loop automated data generation system.*

```
[Vehicle/Plate Parameter Table] ──> [Python Script] ──> [Blender Headless] ──> [8 Video .mp4] + [Mapping JSON File]

```

To complete the project, the final code file will manage the following logical structure:

1. **Scenario Generator:** Creates a random list of vehicles (Vehicle 1: red, plate X, departs at frame 1 heading South straight; Vehicle 2: blue, plate Y, departs at frame 20 heading East turning right...).
2. **Simulation Runner:** Applies the formulas from Phase 1 to calculate the appearance and disappearance coordinates of each vehicle on their respective lanes.
3. **Render & Export Metadata Function:** Renders 8 video streams. Simultaneously exports a `metadata.json` file storing the exact $XYZ$ position of each vehicle at every frame as absolute Ground Truth data for future testing systems.

---

The first phase to implement immediately is **preparing 3D model assets** by exploring and reading the appropriate 3D format files of real-world car models in the models folder.