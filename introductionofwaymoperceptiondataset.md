Data Labels
The perception dataset contains independently-generated labels for lidar and camera data, not simply projections.


3D Lidar Labels
We provide 3D bounding box labels in lidar data. The lidar labels are 3D 7-DOF bounding boxes in the vehicle frame with globally unique tracking IDs. The following objects have 3D labels: vehicles, pedestrians, cyclists, signs. The bounding boxes have zero pitch and zero roll. Heading is the angle (in radians, normalized to [-π, π]) needed to rotate the vehicle frame +X axis about the Z axis to align with the vehicle's forward axis. Each scene may include an area that is not labeled, which is called a “No Label Zone" (NLZ). These capture areas such as the opposite side of a highway. See our label specifications document for details. NLZs are represented as polygons in the global frame. These polygons are not necessarily convex. In addition to these polygons, each lidar point is annotated with a boolean to indicate whether it is in an NLZ or not. Our metrics computation code requires the user to provide information about whether the prediction result is overlapping with any NLZ. Users can get this information by checking whether their prediction overlaps with any NLZ-annotated lidar points (on both 1st and 2nd returns).

2D Camera Labels
We provide 2D bounding box labels in the camera images. The camera labels are tight-fitting, axis-aligned 2D bounding boxes with globally unique tracking IDs. The bounding boxes cover only the visible parts of the objects. The following objects have 2D labels: vehicles, pedestrians, cyclists. We do not provide object track correspondences across cameras.

Details
See the label definition proto and the label specifications document for more details.


Key Points
We provide a collection of 14 key points from across the human body – including nose, right and left shoulders, elbows, wrists, hips, knees, and ankle – or a select subset of 2D and 3D bounding box labels. For each key point, we also provide a visibility/occlusion attribute. Both types of key points were manually labeled and verified: 2D key points - using camera images, 3D key points - using lidar point clouds. The key points help us understand pedestrian movement, but don’t identify specific individuals. We released 172.6K object annotations with key points for camera objects (2D key points) and 10K object annotations - for laser objects (3D key points). This data can be used to train semi and weakly supervised models for 3D human body pose understanding in the wild. See the key point definition proto and the tutorial for more details.

2D-to-3D correspondence
Objects labeled on camera images with 2D bounding boxes, which are also labeled with 3D bounding boxes on the lidar point cloud, contain corresponding objects IDs. This correspondence between 2D and 3D bounding boxes was labeled and verified by human operators. Note that due to occlusions and differences in fields of view between camera and lidar sensors some objects do not have this correspondence established.


3D Segmentation
Dense labels for every LiDAR point with rich semantics - 23 classes as listed below. The labels are provided at 2Hz for the entire dataset captured by high-resolution LiDAR sensors.We include the following 23 fine-grained categories: Car, Truck, Bus, Motorcyclist, Bicyclist, Pedestrian, Sign, Traffic Light, Pole, Construction Cone, Bicycle, Motorcycle, Building, Vegetation, Tree Trunk, Curb, Road, Lane Marker, Walkable, Sidewalk, Other Ground, Other Vehicle, Undefined

2D Video Panoptic Segmentation
We provide semantic segmentation and instance segmentation labels for a subset of 100k camera images, grouped into 2,860 temporal sequences captured by five cameras. For the training set, we subsample 700 run segments and label four 5-frame groups in each. For the validation and test sets, we subsample 10 and 20 run segments, respectively, and label densely at 100 frames at 5Hz. We provide labels for 28 categories. The dataset offers diversity in object classes, geographical locations, weather, and time of day. We include the following 28 fine-grained categories: Car, Bus, Truck, Other Large Vehicle, Trailer, Ego Vehicle, Motorcycle, Bicycle, Pedestrian, Cyclist, Motorcyclist, Ground Animal, Bird, Pole, Sign, Traffic Light, Construction Cone, Pedestrian Object, Building, Road, Sidewalk, Road Marker, Lane Marker, Vegetation, Sky, Ground, Static, Dynamic. Instance segmentation labels are provided for the Vehicle, Pedestrian and Cyclist classes, and are consistent both across cameras and over time.

Coordinate Systems
We use the following coordinate systems in the dataset.

Global frame
The origin of this frame is set to the vehicle position when the vehicle starts. It is an ‘East-North-Up’ coordinate frame. ‘Up(z)’ is aligned with the gravity vector, positive upwards. ‘East(x)’ points directly east along the line of latitude. ‘North(y)’ points towards the north pole.

Vehicle frame
The x-axis is positive forwards, y-axis is positive to the left, z-axis is positive upwards. A vehicle pose defines the transform from the vehicle frame to the global frame.

Sensor frames
Each sensor comes with an extrinsic transform that defines the transform from the sensor frame to the vehicle frame.The camera frame is placed in the center of the camera lens. The x-axis points down the lens barrel out of the lens. The z-axis points up. The y/z plane is parallel to the camera plane. The coordinate system is right handed.The lidar sensor frame has the z-axis pointing upward with the x/y plane depending on the lidar position.

Lidar Spherical Coordinates
The lidar spherical coordinate system is based on the Cartesian coordinate system in lidar sensor frame. A point (x, y, z) in lidar Cartesian coordinates can be uniquely translated to a (range, azimuth, inclination) tuple in lidar spherical coordinates.

Lidar Data
The dataset contains data from five lidars - one mid-range lidar (top) and four short-range lidars (front, side left, side right, and rear)For the purposes of this dataset, the following limitations were applied to lidar data:

Range of the mid-range lidar truncated to a maximum of 75 meters

Range of the short-range lidars truncated to a maximum of 20 meters

The strongest two intensity returns are provided for all five lidars

An extrinsic calibration matrix transforms the lidar frame to the vehicle frame. The mid-range lidar has a non-uniform inclination beam angle pattern. A 1D tensor is available to get the exact inclination of each beam. The point cloud of each lidar is encoded as a range image. Two range images are provided for each lidar, one for each of the two strongest returns. It has 4 channels:

channel 0: range (see spherical coordinate system definition)

channel 1: lidar intensity

channel 2: lidar elongation

channel 3: is_in_nlz (1 = in, -1 = not in)

Lidar elongation refers to the elongation of the pulse beyond its nominal width. Returns with long pulse elongation, for example, indicate that the laser reflection is potentially smeared or refracted, such that the return pulse is elongated in time.In addition to the basic 4 channels, we also provide another 6 channels for lidar to camera projection. The projection method used takes rolling shutter effect into account:

channel 0: camera name

channel 1: x (axis along image width)

channel 2: y (axis along image height)

channel 3: camera name of 2nd projection (set to UNKNOWN if no projection)

channel 4: x (axis along image width)

channel 5: y (axis along image height)

A range image represents a lidar point cloud in the spherical coordinate system based on the following rules:

Each row corresponds to an inclination. Row 0 (top of the image) corresponds to the maximum inclination.

Each column corresponds to an azimuth. Column 0 (left of the image) corresponds to -x-axis (i.e. the opposite of forward direction). The center of the image corresponds to the +x-axis (i.e. the forward direction). Note that an azimuth correction is needed to make sure the center of the image corresponds to the +x-axis.

Example range image

Camera Data
The dataset contains images from five cameras associated with five different directions. They are front, front left, front right, side left, and side right. One camera image is provided for each pair in JPEG format. In addition to the image bytes, we also provide the vehicle pose, the velocity corresponding to the exposure time of the image center and rolling shutter timing information. This information is useful to customize the lidar to camera projection, if needed.


Object Assets
Modeling the 3D world from sensor data for simulation is a scalable way of developing testing and validation environments for robotic learning problems such as autonomous driving. We provide a large-scale object-centric assets dataset containing over 1.2M images and lidar observations of two major categories (vehicles and pedestrians) from the released Waymo Open Dataset Perception Dataset (v2.0.0). We hope this data will enable and advance research on 3D point cloud reconstruction and completion, object NeRF reconstruction, and generative object assets to address the real-world driving challenges with occlusions, partial observations, lighting-variations, and long-tail distributions. We leverage the existing 3D lidar box annotations to exclude objects beyond certain distances to the surveying vehicle in each data frame (e.g., 40m for pedestrians and 60m for vehicles, respectively). At a given frame, we project 3D point clouds within each 3D bounding box to the most visible camera and extract the centering patch to build our single-view 2D image collections. Furthermore, we provide 2D panoptic segmentation auto-labels and refined box pose for the vehicle category. We use the sensor calibrations to compute ray directions for each 2D pixel, taking into account the camera rolling shutter effects, ego-motion and object motion.Key features of this dataset includes:

1.2M object frames from 31K unique object instances that covers 2 classes (vehicles and pedestrians)

Extracted perception objects from multi-sensor data: all 5 cameras and the top lidar

Lidar features include 3D point cloud sequences that support 3D object shape reconstruction. We additionally provide refined box pose through point cloud shape registration for all objects in the vehicle category

Camera features include sequences of camera patches from the most_visible_camera, projected lidar returns on the corresponding camera, per-pixel camera rays information, and auto-labeled 2D panoptic segmentation that supports object NeRF reconstruction, detailed in the paper

Please see the new object assets tutorial on GitHub.

Maps
We provide maps in the v1.4.2 release. This includes lane centers, lane boundaries, road boundaries, crosswalks, speed bumps, stop signs, and entrances to driveways. Map features are defined as 3D polylines or polygons. See the map proto definitions for full details.

Modular format
In the v2.0.0 release, we introduce a major update to the dataset format. The new format, based on Apache Parquet column-oriented file format, separates the data into multiple tables, allowing users to selectively download only the components of the dataset that they need for their specific use case. This modular format offers a significant advantage over the previous format by reducing the amount of data that needs to be downloaded and processed, saving time and resources. Check out the tutorial for more information. The features supported in modular format include: sensor data, bounding boxes, keypoints, camera-to-lidar object association and segmentation. The only feature that is not currently supported in the modular dataset format is maps, which can be accessed in the v1.4.2 dataset.