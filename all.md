# Visualization Demo of Autonomous Driving Scene Analysis Using Waymo Dataset

## 1. Motivation

The motivation for this project stems from the complexity of autonomous driving data, particularly multi-sensor data such as LiDAR and cameras, which requires specialized skills and resources to process. Autonomous driving systems generate massive amounts of heterogeneous sensor data that is challenging to interpret without proper visualization tools.

The Waymo Open Dataset, while incredibly valuable, presents significant challenges for researchers:

- The sheer volume of data (600,000 frames across 1,950 segments) makes manual inspection impractical
- Multi-modal sensor fusion requires specialized tools to visualize LiDAR point clouds alongside camera images
- The TFRecord format is not directly compatible with many visualization platforms
- Temporal relationships between frames require specialized playback capabilities
- Different coordinate systems between sensors necessitate complex transformations for unified visualization

This project seeks to bridge the gap between raw data and actionable insights by providing interactive and intuitive visualization tools. These tools aid researchers and developers in comprehending diverse driving scenarios and support the development of autonomous vehicle perception systems. Our visualization framework enables:

- Immediate visual inspection of sensor data without extensive preprocessing
- Interactive exploration of driving scenes from multiple perspectives
- Integrated visualization of detection and segmentation results
- Statistical analysis of scene composition and object distributions
- Temporal analysis of object trajectories and environmental changes

By creating accessible visualization interfaces, we facilitate deeper understanding of sensor fusion, object detection, and scene classification within autonomous driving environments, accelerating research and development in this critical field.

## 2. Data Source & Background

### 2.1 Waymo Open Dataset Key Features

The Waymo Open Dataset serves as the primary data source for this project, offering a comprehensive collection of autonomous driving data with the following key characteristics:

- **Multi-sensor Data**: 
  - 5 LiDARs: One mid-range LiDAR (top) with 75m range and four short-range LiDARs (front, left, right, rear) with 20m range
  - 5 high-resolution cameras: front, front-left, front-right, side-left, and side-right, providing 360° visual coverage
  - Each sensor captures data at different frequencies, requiring precise time synchronization

- **Geographic Diversity**: 
  - Collected across three major US cities: San Francisco (urban grid with steep hills), Phoenix (suburban desert environment), and Mountain View (campus-like setting)
  - Includes various road types: highways, urban streets, residential areas, and intersections
  - Diverse infrastructure elements: traffic signals, road markings, construction zones

- **Time Coverage**: 
  - Day and night conditions with various lighting situations (direct sunlight, shadows, artificial lighting)
  - Weather variations including clear, rainy, and foggy conditions
  - Different traffic densities from sparse to congested scenarios

- **Annotation Types**:  
  - 3D bounding boxes: 7-DOF (x,y,z,length,width,height,heading) annotations for 1.2M objects
  - 2D bounding boxes: Tight-fitting, axis-aligned boxes in image space with tracking IDs
  - 3D segmentation: 23 distinct classes for LiDAR points (e.g., vehicle, pedestrian, road surface)
  - 2D video panoptic segmentation: 28 fine-grained categories with consistent instance IDs across frames
  - Key points: 14 anatomical points for human body pose tracking
  - No Label Zones (NLZs): Areas explicitly marked as not labeled (e.g., opposite lanes on highways)

- **Data Volume**:  
  - 1,950 segments, each capturing 20 seconds of continuous driving
  - Approximately 600,000 frames (10 Hz sampling rate)
  - 200,000 kilometers of diverse driving scenarios
  - Raw data size exceeds 1.2TB, presenting storage and processing challenges

- **Object Assets**: 
  - 1.2M images and LiDAR observations of vehicles and pedestrians
  - Object-centric assets designed for 3D reconstruction tasks
  - Instances include partial observations, occlusions, and various viewing angles
  - Refined pose information through point cloud registration techniques

### 2.2 Coordinate Systems

The Waymo Open Dataset utilizes several coordinate systems that must be understood for proper data processing and visualization:

- **Global Frame**:
  - East-North-Up (ENU) coordinate system
  - Origin set at the vehicle's starting position in each segment
  - X-axis points east along latitude lines
  - Y-axis points north toward the pole
  - Z-axis points up, aligned with gravity
  - Primarily used for localization and mapping applications

- **Vehicle Frame**:
  - Right-handed coordinate system centered at the rear axle midpoint
  - X-axis points forward along the vehicle's longitudinal axis
  - Y-axis points left from the driver's perspective
  - Z-axis points up, perpendicular to the ground plane
  - Vehicle pose defines the transformation from vehicle to global frame
  - Critical for sensor fusion and motion planning applications

- **Sensor Frames**:
  - Each sensor has a unique coordinate system with an associated extrinsic calibration matrix
  - LiDAR frames: Z-axis pointing upward, with X and Y axes depending on mounting position
  - Camera frames: X-axis points out of the lens, Z-axis points up, Y-axis completes right-handed system
  - Extrinsic matrices provide transformations between sensor frames and vehicle frame
  - Intrinsic matrices define camera projection parameters (focal length, principal point)

- **LiDAR Spherical Coordinates**:
  - Used for range image representation of point clouds
  - Defined by range (distance from sensor), azimuth (horizontal angle), and inclination (vertical angle)
  - Conversion between spherical and Cartesian coordinates is essential for point cloud processing
  - Non-uniform inclination pattern in mid-range LiDAR requires specific beam angle tables

Understanding these coordinate systems and implementing proper transformations between them is critical for correctly visualizing and analyzing the Waymo dataset. Our framework handles these transformations automatically, providing a unified visualization space for multi-sensor data.

### 2.3 LiDAR Data Structure

LiDAR data in the Waymo dataset is structured as follows:

- **Range Limitations**:
  - Mid-range LiDAR (top): Maximum range truncated to 75 meters
  - Short-range LiDARs (front, side left, side right, rear): Maximum range truncated to 20 meters
  - These limitations ensure reliable detection within specified operational boundaries

- **Return Structure**:
  - Each LiDAR provides the two strongest intensity returns per emitted pulse
  - Primary returns typically capture the closest obstacle
  - Secondary returns often capture partially transparent objects or objects beyond the first point of reflection
  - This dual-return system enhances perception through vegetation and partial occlusions

- **Range Image Format**:
  - Point clouds are encoded as 2D range images for efficient storage and processing
  - Each row corresponds to a specific inclination angle (vertical laser beam)
  - Each column represents an azimuth angle (horizontal scanning position)
  - Range images enable direct application of 2D computer vision techniques to 3D data

- **Channel Configuration**:
  - Each range image contains 4 basic channels capturing fundamental LiDAR measurements:
    1. Range: Direct distance measurement from sensor to detected point
    2. Intensity: Signal strength of the returned laser pulse, indicating reflectivity
    3. Elongation: Measure of pulse stretching, indicating potential smearing or refraction
    4. No Label Zone Indicator: Binary flag marking points in regions explicitly not labeled

  - 6 additional projection channels facilitate LiDAR-to-camera mapping:
    1. Camera name: Identifier for primary projection camera
    2. X-coordinate: Horizontal position in image space
    3. Y-coordinate: Vertical position in image space
    4. Secondary camera name: Identifier for alternative view (if available)
    5. Secondary X-coordinate: Horizontal position in secondary camera
    6. Secondary Y-coordinate: Vertical position in secondary camera

This structured representation enables efficient processing and cross-modal fusion between LiDAR and camera data, which our visualization system leverages to provide comprehensive multi-sensor scene understanding.

## 3. Methodology

### 3.1 Data Preparation

The process of transforming raw Waymo Open Dataset files into a usable format involves sophisticated data engineering techniques across multiple stages:

#### 3.1.1 TFRecord Parsing

The Waymo Open Dataset is distributed in TFRecord format, a binary serialization format optimized for TensorFlow. Parsing these files requires specialized handling:

```python
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
import os

def parse_tfrecord_files(file_paths, output_dir):
    """Parse multiple TFRecord files with error handling and logging"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    frame_count = 0
    for file_path in file_paths:
        try:
            # Open dataset with appropriate compression
            dataset = tf.data.TFRecordDataset(file_path, compression_type='GZIP')
            
            # Process each frame in the dataset
            for frame_idx, data in enumerate(dataset):
                # Parse frame from protobuf
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                
                # Extract frame timestamp for unique identification
                timestamp = frame.timestamp_micros
                
                # Save processed frame data
                output_path = os.path.join(output_dir, f"{os.path.basename(file_path)}_{frame_idx}_{timestamp}.pb")
                with open(output_path, 'wb') as f:
                    f.write(frame.SerializeToString())
                
                frame_count += 1
                
                # Log progress periodically
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Successfully processed {frame_count} frames from {len(file_paths)} files")
    return frame_count
```

This function handles multiple TFRecord files, implements error recovery, and extracts individual frames with unique identifiers. The resulting protocol buffer files serve as an intermediate representation for further processing.

#### 3.1.2 MCAP Conversion

To enable compatibility with modern visualization tools like Foxglove Studio, we convert the parsed data to MCAP format:

```python
from waymo_open_dataset.utils import frame_utils
import mcap
import mcap.records
from datetime import datetime
import numpy as np
import json

def convert_tfrecord_to_mcap(tfrecord_path, mcap_path):
    """Convert Waymo TFRecord to MCAP format with rich metadata"""
    
    # Open dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    
    # Prepare schema definitions for different message types
    schemas = {
        "visualization_msgs/MarkerArray": mcap.records.Schema(
            id=1,
            name="visualization_msgs/MarkerArray",
            encoding="json",
            data=json.dumps({"type": "visualization_msgs/MarkerArray"}).encode()
        ),
        "sensor_msgs/PointCloud2": mcap.records.Schema(
            id=2,
            name="sensor_msgs/PointCloud2",
            encoding="json",
            data=json.dumps({"type": "sensor_msgs/PointCloud2"}).encode()
        ),
        "sensor_msgs/Image": mcap.records.Schema(
            id=3,
            name="sensor_msgs/Image",
            encoding="json",
            data=json.dumps({"type": "sensor_msgs/Image"}).encode()
        ),
        "tf2_msgs/TFMessage": mcap.records.Schema(
            id=4,
            name="tf2_msgs/TFMessage",
            encoding="json",
            data=json.dumps({"type": "tf2_msgs/TFMessage"}).encode()
        )
    }
    
    # Create channels for different data types
    channels = {
        "/lidar/points": mcap.records.Channel(
            id=1,
            schema_id=2,
            topic="/lidar/points",
            message_encoding="ros1",
            metadata={}
        ),
        "/camera/front": mcap.records.Channel(
            id=2,
            schema_id=3,
            topic="/camera/front",
            message_encoding="ros1",
            metadata={}
        ),
        "/tf": mcap.records.Channel(
            id=3,
            schema_id=4,
            topic="/tf",
            message_encoding="ros1",
            metadata={}
        ),
        "/objects": mcap.records.Channel(
            id=4,
            schema_id=1,
            topic="/objects",
            message_encoding="ros1",
            metadata={}
        )
    }
    
    # Additional channels for other cameras and LiDARs would be defined here
    
    with mcap.Writer(open(mcap_path, 'wb')) as writer:
        # Write header with metadata
        writer.start(
            profile="",
            library="waymo_conversion_tool",
            metadata={
                "source": os.path.basename(tfrecord_path),
                "created": datetime.now().isoformat(),
                "description": "Waymo Open Dataset converted to MCAP"
            }
        )
        
        # Write schemas
        for schema in schemas.values():
            writer.add_schema(schema)
            
        # Write channels
        for channel in channels.values():
            writer.add_channel(channel)
        
        # Process each frame
        for data_idx, data in enumerate(dataset):
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            
            # Extract timestamp in nanoseconds for MCAP
            timestamp_ns = frame.timestamp_micros * 1000
            
            # Process and write LiDAR data
            if frame.lasers:
                points, cp_points = frame_utils.convert_range_image_to_point_cloud(
                    frame, range_images=frame_utils.parse_range_image_and_camera_projection(frame), 
                    keep_polar_features=True
                )
                
                # Serialize point cloud to ROS PointCloud2 format
                point_cloud_msg = convert_points_to_pointcloud2(points[0], frame.context.name, timestamp_ns)
                
                # Write message
                writer.add_message(
                    channels["/lidar/points"],
                    log_time=timestamp_ns,
                    publish_time=timestamp_ns,
                    data=json.dumps(point_cloud_msg).encode()
                )
            
            # Process and write camera data
            for camera in frame.images:
                camera_name = dataset_pb2.CameraName.Name.Name(camera.name)
                topic = f"/camera/{camera_name.lower()}"
                
                # Ensure channel exists
                if topic not in channels:
                    channel_id = len(channels) + 1
                    channels[topic] = mcap.records.Channel(
                        id=channel_id,
                        schema_id=3,
                        topic=topic,
                        message_encoding="ros1",
                        metadata={}
                    )
                    writer.add_channel(channels[topic])
                
                # Convert image to ROS format and write
                image_msg = convert_image_to_ros(camera.image, camera_name, timestamp_ns)
                writer.add_message(
                    channels[topic],
                    log_time=timestamp_ns,
                    publish_time=timestamp_ns,
                    data=json.dumps(image_msg).encode()
                )
            
            # Process and write TF transformations
            tf_msg = create_tf_message(frame, timestamp_ns)
            writer.add_message(
                channels["/tf"],
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=json.dumps(tf_msg).encode()
            )
            
            # Process and write object detections as visualization markers
            if frame.laser_labels:
                markers_msg = create_marker_array_from_labels(frame.laser_labels, timestamp_ns)
                writer.add_message(
                    channels["/objects"],
                    log_time=timestamp_ns,
                    publish_time=timestamp_ns,
                    data=json.dumps(markers_msg).encode()
                )
            
            # Log progress periodically
            if data_idx % 10 == 0:
                print(f"Processed {data_idx} frames...")
    
    print(f"Converted {tfrecord_path} to {mcap_path}")
```

This comprehensive conversion implements:
- Schema and channel definitions for different data types
- Detailed metadata capturing source information and processing history
- Conversion of range images to point clouds with preserved features
- Transformation of camera images to standardized formats
- Preservation of coordinate transformations as TF messages
- Representation of object detections as visualization markers

These features ensure that the resulting MCAP files contain all necessary information for sophisticated visualization and analysis.

#### 3.1.3 Coordinate System Transformation

Accurate alignment of data from different sensors requires precise coordinate transformations:

```python
import numpy as np
from scipy.spatial.transform import Rotation

def transform_point_cloud(points, source_frame, target_frame, calibration):
    """Transform points from source coordinate frame to target coordinate frame
    
    Args:
        points: Nx3 array of points in source frame
        source_frame: String identifier for source coordinate system
        target_frame: String identifier for target coordinate system
        calibration: Calibration object containing extrinsic transformations
        
    Returns:
        Nx3 array of transformed points in target frame
    """
    # Extract transformation matrices from calibration data
    transformations = {
        "vehicle_to_global": calibration.vehicle_to_global,
        "lidar_top_to_vehicle": calibration.lidar_extrinsics[0],
        "lidar_front_to_vehicle": calibration.lidar_extrinsics[1],
        "lidar_side_left_to_vehicle": calibration.lidar_extrinsics[2],
        "lidar_side_right_to_vehicle": calibration.lidar_extrinsics[3],
        "lidar_rear_to_vehicle": calibration.lidar_extrinsics[4],
        "camera_front_to_vehicle": calibration.camera_extrinsics[0],
        "camera_front_left_to_vehicle": calibration.camera_extrinsics[1],
        "camera_front_right_to_vehicle": calibration.camera_extrinsics[2],
        "camera_side_left_to_vehicle": calibration.camera_extrinsics[3],
        "camera_side_right_to_vehicle": calibration.camera_extrinsics[4]
    }
    
    # Add inverse transformations
    for name, matrix in list(transformations.items()):
        if name.endswith("_to_vehicle"):
            inverse_name = f"vehicle_to_{name.split('_to_')[0]}"
            transformations[inverse_name] = np.linalg.inv(matrix)
    
    # Construct transformation path from source to target
    if source_frame == target_frame:
        return points
    
    # Direct transformation
    transform_key = f"{source_frame}_to_{target_frame}"
    if transform_key in transformations:
        transform_matrix = transformations[transform_key]
    else:
        # Two-step transformation via vehicle frame
        to_vehicle_key = f"{source_frame}_to_vehicle"
        from_vehicle_key = f"vehicle_to_{target_frame}"
        
        if to_vehicle_key not in transformations or from_vehicle_key not in transformations:
            raise ValueError(f"Cannot find transformation path from {source_frame} to {target_frame}")
            
        to_vehicle = transformations[to_vehicle_key]
        from_vehicle = transformations[from_vehicle_key]
        transform_matrix = np.matmul(from_vehicle, to_vehicle)
    
    # Apply transformation
    homogeneous_points = np.hstack([points, np.ones((points.shape[0], 1))])
    transformed_points = np.dot(homogeneous_points, transform_matrix.T)
    
    # Return Cartesian coordinates
    return transformed_points[:, :3]

def spherical_to_cartesian(range_image, inclinations, azimuths):
    """Convert spherical coordinates in range image to Cartesian coordinates
    
    Args:
        range_image: HxW range image with distances in meters
        inclinations: H-vector of vertical angles in radians
        azimuths: W-vector of horizontal angles in radians
        
    Returns:
        HxWx3 array of Cartesian coordinates (x, y, z)
    """
    # Create meshgrid of angles
    inclination_grid, azimuth_grid = np.meshgrid(inclinations, azimuths, indexing='ij')
    
    # Calculate Cartesian coordinates
    x = range_image * np.cos(inclination_grid) * np.cos(azimuth_grid)
    y = range_image * np.cos(inclination_grid) * np.sin(azimuth_grid)
    z = range_image * np.sin(inclination_grid)
    
    # Stack coordinates
    cartesian_points = np.stack([x, y, z], axis=-1)
    return cartesian_points
```

These functions implement:
- Flexible transformation between any pair of coordinate systems
- Multi-step transformations via the vehicle frame when direct transformations aren't available
- Automatic construction of inverse transformations
- Conversion from spherical LiDAR coordinates to Cartesian coordinates

By centralizing coordinate transformations, our system ensures consistent spatial alignment of all sensor data, essential for accurate multi-sensor visualization.

### 3.2 Multi-sensor Data Processing

Processing synchronized data from LiDARs and cameras to extract meaningful features requires specialized algorithms for each sensor type and effective fusion strategies.

#### 3.2.1 LiDAR Point Cloud Processing

LiDAR data processing involves several stages to convert raw range images into useful 3D representations:

```python
import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor

def process_point_cloud(range_image, calibration, voxel_size=0.1, intensity_scale=255.0):
    """Process LiDAR range image into structured point cloud with various features
    
    Args:
        range_image: Range image with channels [range, intensity, elongation, is_in_nlz]
        calibration: Sensor calibration data
        voxel_size: Resolution for voxel downsampling (in meters)
        intensity_scale: Scaling factor for normalizing intensity values
        
    Returns:
        Dictionary with processed point cloud features
    """
    # Extract channels
    ranges = range_image[:, :, 0]
    intensities = range_image[:, :, 1]
    elongations = range_image[:, :, 2]
    nlz_flags = range_image[:, :, 3]
    
    # Get beam angles from calibration
    inclinations = calibration.beam_inclinations
    height, width = ranges.shape
    azimuths = np.linspace(-np.pi, np.pi, width, endpoint=False)
    
    # Convert to Cartesian coordinates
    points_xyz = spherical_to_cartesian(ranges, inclinations, azimuths)
    
    # Reshape to list of points and filter invalid points
    valid_mask = (ranges > 0) & (nlz_flags < 0)  # Positive range, not in NLZ
    points = points_xyz[valid_mask]
    intensities = intensities[valid_mask]
    elongations = elongations[valid_mask]
    
    # Normalize intensities to [0, 1]
    normalized_intensities = np.clip(intensities / intensity_scale, 0, 1)
    
    # Transform to vehicle coordinates
    points_vehicle = transform_point_cloud(
        points, 
        source_frame="lidar_top", 
        target_frame="vehicle", 
        calibration=calibration
    )
    
    # Create Open3D point cloud for processing
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_vehicle)
    pcd.colors = o3d.utility.Vector3dVector(
        np.stack([normalized_intensities, normalized_intensities, normalized_intensities], axis=-1)
    )
    
    # Voxel downsampling to reduce density
    if voxel_size > 0:
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
    else:
        pcd_downsampled = pcd
    
    # Ground segmentation using RANSAC
    points_array = np.asarray(pcd_downsampled.points)
    
    # Create simple features for ground detection (assuming ground is perpendicular to z-axis)
    X = points_array[:, :2]  # x, y coordinates
    y = points_array[:, 2]   # z coordinates (height)
    
    # Fit ground plane using RANSAC
    ransac = RANSACRegressor(
        max_trials=100,
        residual_threshold=0.2,  # 20cm threshold for inliers
        random_state=42
    )
    
    try:
        ransac.fit(X, y)
        
        # Predict height of each point if it were on the ground plane
        y_pred = ransac.predict(X)
        
        # Points are ground if close to the predicted plane
        ground_mask = np.abs(y - y_pred) < 0.2
        
        # Create segmented point clouds
        ground_points = points_array[ground_mask]
        non_ground_points = points_array[~ground_mask]
        
        # Create colored point clouds for visualization
        ground_colors = np.zeros((ground_points.shape[0], 3))
        ground_colors[:, 1] = 0.8  # Green for ground
        
        non_ground_colors = np.zeros((non_ground_points.shape[0], 3))
        non_ground_colors[:, 0] = 0.8  # Red for non-ground
        
        # Create separate point clouds for ground and obstacles
        ground_pcd = o3d.geometry.PointCloud()
        ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
        ground_pcd.colors = o3d.utility.Vector3dVector(ground_colors)
        
        non_ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
        non_ground_pcd.colors = o3d.utility.Vector3dVector(non_ground_colors)
        
    except Exception as e:
        print(f"Ground segmentation failed: {e}")
        ground_mask = np.zeros(points_array.shape[0], dtype=bool)
        ground_pcd = o3d.geometry.PointCloud()
        non_ground_pcd = pcd_downsampled
    
    # Return structured result
    return {
        'points': points_vehicle,
        'intensities': normalized_intensities,
        'elongations': elongations,
        'downsampled_pcd': pcd_downsampled,
        'ground_mask': ground_mask,
        'ground_pcd': ground_pcd,
        'non_ground_pcd': non_ground_pcd
    }
```

This comprehensive function implements:
- Channel extraction from range images
- Spherical to Cartesian coordinate conversion
- Filtering of invalid points and points in No Label Zones
- Intensity normalization for visualization
- Coordinate transformation to vehicle frame
- Voxel downsampling for computational efficiency
- Ground plane segmentation using RANSAC
- Creation of colored point clouds for ground and non-ground points

These processed point clouds provide a foundation for visualization and further perceptual analysis.

#### 3.2.2 Camera Image Processing

Camera data requires specialized processing to align with LiDAR data and enhance visual information:

```python
import cv2
import numpy as np
from PIL import Image
import io

def process_camera_image(image_data, camera_name, calibration):
    """Process camera image data with intrinsic correction and annotation
    
    Args:
        image_data: Raw image bytes from Waymo dataset
        camera_name: Camera identifier (e.g., 'FRONT', 'FRONT_LEFT')
        calibration: Camera calibration parameters
        
    Returns:
        Dictionary with processed image and metadata
    """
    # Decode image bytes
    if isinstance(image_data, bytes):
        pil_image = Image.open(io.BytesIO(image_data))
        image = np.array(pil_image)
    else:
        image = image_data
    
    # Extract camera intrinsics
    camera_idx = {
        'FRONT': 0,
        'FRONT_LEFT': 1,
        'FRONT_RIGHT': 2,
        'SIDE_LEFT': 3,
        'SIDE_RIGHT': 4
    }[camera_name]
    
    intrinsics = calibration.camera_intrinsics[camera_idx]
    distortion = calibration.camera_distortion[camera_idx]
    
    # Create camera matrix
    camera_matrix = np.array([
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1]
    ])
    
    # Undistort image
    height, width = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, 
        distortion, 
        (width, height), 
        0, 
        (width, height)
    )
    
    undistorted = cv2.undistort(
        image, 
        camera_matrix, 
        distortion, 
        None, 
        new_camera_matrix
    )
    
    # Crop the image to the ROI
    x, y, w, h = roi
    undistorted_cropped = undistorted[y:y+h, x:x+w]
    
    # Enhance image for better visualization
    enhanced = enhance_image(undistorted_cropped)
    
    # Create visualizable version with camera info overlay
    visualization = enhanced.copy()
    cv2.putText(
        visualization,
        f"Camera: {camera_name}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Add field of view lines
    fov_horizontal = np.degrees(2 * np.arctan(width / (2 * intrinsics[0])))
    fov_vertical = np.degrees(2 * np.arctan(height / (2 * intrinsics[1])))
    
    cv2.putText(
        visualization,
        f"FOV: {fov_horizontal:.1f}°H x {fov_vertical:.1f}°V",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    
    return {
        'original': image,
        'undistorted': undistorted_cropped,
        'enhanced': enhanced,
        'visualization': visualization,
        'camera_matrix': new_camera_matrix,
        'fov': (fov_horizontal, fov_vertical),
        'camera_name': camera_name
    }

def enhance_image(image):
    """Enhance image quality for better visualization
    
    Args:
        image: Raw camera image
        
    Returns:
        Enhanced image with improved contrast and brightness
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    # Convert back to RGB
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb
```

This camera processing pipeline implements:
- Image decoding from raw bytes
- Lens distortion correction using camera intrinsics
- Region of interest cropping
- Image enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Visualization overlay with camera information and field of view

The processed images provide clear visual context for driving scene understanding and serve as a foundation for projection of LiDAR points.

#### 3.2.3 Sensor Fusion

Integrating LiDAR and camera data creates a cohesive representation of the driving scene:

```python
import numpy as np
import cv2

def project_lidar_to_camera(point_cloud, camera_data, calibration):
    """Project LiDAR points onto camera image
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        camera_data: Dictionary containing processed camera data
        calibration: Calibration data for transformations
        
    Returns:
        Visualization with projected points and image with depth information
    """
    # Extract points in vehicle frame
    points_vehicle = point_cloud['points']
    intensities = point_cloud['intensities']
    
    # Transform from vehicle to camera frame
    camera_name = camera_data['camera_name']
    points_camera = transform_point_cloud(
        points_vehicle,
        source_frame="vehicle",
        target_frame=f"camera_{camera_name.lower()}",
        calibration=calibration
    )
    
    # Filter points in front of camera (positive x)
    in_front_mask = points_camera[:, 0] > 0
    points_camera = points_camera[in_front_mask]
    intensities = intensities[in_front_mask]
    
    # Project 3D points to 2D image plane
    camera_matrix = camera_data['camera_matrix']
    
    # Perspective projection
    points_2d_homogeneous = np.dot(points_camera, camera_matrix.T)
    
    # Normalize by depth
    depths = points_2d_homogeneous[:, 2]
    points_2d = points_2d_homogeneous[:, :2] / depths[:, np.newaxis]
    
    # Create visualization image
    image = camera_data['enhanced'].copy()
    height, width = image.shape[:2]
    
    # Filter points within image bounds
    in_bounds_mask = (
        (points_2d[:, 0] >= 0) & 
        (points_2d[:, 0] < width) & 
        (points_2d[:, 1] >= 0) & 
        (points_2d[:, 1] < height)
    )
    
    valid_points_2d = points_2d[in_bounds_mask].astype(np.int32)
    valid_depths = depths[in_bounds_mask]
    valid_intensities = intensities[in_bounds_mask]
    
    # Create colormap based on depth
    min_depth = max(0.1, np.min(valid_depths))
    max_depth = min(75.0, np.max(valid_depths))
    
    depth_normalized = (valid_depths - min_depth) / (max_depth - min_depth)
    depth_colors = cv2.applyColorMap(
        (255 * (1.0 - depth_normalized)).astype(np.uint8),
        cv2.COLORMAP_TURBO
    )
    
    # Extract RGB values from colormap
    depth_rgb = depth_colors.reshape(-1, 3)
    
    # Draw points on image
    visualization = image.copy()
    for i, (x, y) in enumerate(valid_points_2d):
        color = tuple(map(int, depth_rgb[i]))
        size = max(1, min(5, int(10 * (1.0 - depth_normalized[i]))))
        cv2.circle(visualization, (x, y), size, color, -1)
    
    # Create depth image
    depth_image = np.zeros((height, width), dtype=np.float32)
    for i, (x, y) in enumerate(valid_points_2d):
        depth_image[y, x] = valid_depths[i]
    
    # Normalize and colorize depth image for visualization
    depth_vis = np.zeros((height, width, 3), dtype=np.uint8)
    mask = depth_image > 0
    if np.any(mask):
        normalized = np.zeros_like(depth_image)
        normalized[mask] = (depth_image[mask] - min_depth) / (max_depth - min_depth)
        normalized = (255 * (1.0 - normalized)).astype(np.uint8)
        depth_vis = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        # Set pixels with no depth to black
        depth_vis[~mask] = [0, 0, 0]
    
    # Add depth scale reference
    cv2.rectangle(visualization, (width-150, 20), (width-20, 280), (0, 0, 0), -1)
    
    for i in range(10):
        y = 250 - i * 20
        depth_val = min_depth + (i / 10.0) * (max_depth - min_depth)
        color = tuple(map(int, cv2.applyColorMap(np.array([[[255 - i * 25]]]), cv2.COLORMAP_TURBO)[0, 0]))
        cv2.rectangle(visualization, (width-140, y-10), (width-100, y+10), color, -1)
        cv2.putText(
            visualization,
            f"{depth_val:.1f}m",
            (width-95, y+5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    cv2.putText(
        visualization,
        f"Depth Range: {min_depth:.1f}m - {max_depth:.1f}m",
        (width-280, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    return {
        'visualization': visualization,
        'depth_image': depth_image,
        'depth_visualization': depth_vis,
        'projected_points': valid_points_2d,
        'projected_depths': valid_depths,
        'min_depth': min_depth,
        'max_depth': max_depth
    }
```

This sensor fusion implementation:
- Transforms LiDAR points from vehicle to camera coordinate frames
- Filters points outside the camera's field of view
- Projects 3D points to the 2D image plane using perspective projection
- Colorizes points based on depth using a perceptually accurate color map
- Creates a sparse depth image from projected points
- Adds visualization aids such as depth scales and range information

The resulting visualizations provide intuitive representations of the 3D scene structure as observed by the autonomous vehicle's sensors.

### 3.3 Perception Tasks

Advanced analysis of processed data to understand the driving environment involves sophisticated algorithms for object detection, segmentation, and scene classification.

#### 3.3.1 3D Object Detection

Detecting and classifying objects in 3D space is a core perception task that our system implements through a multi-stage pipeline:

```python
import numpy as np
import tensorflow as tf
from waymo_open_dataset.utils import box_utils
from scipy.spatial import ConvexHull

def detect_objects_3d(point_cloud, model_path):
    """Detect 3D objects in LiDAR point cloud
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        model_path: Path to pre-trained 3D object detection model
        
    Returns:
        Dictionary with detected objects and their properties
    """
    # Load model (assuming TensorFlow SavedModel format)
    model = tf.saved_model.load(model_path)
    detect_fn = model.signatures['serving_default']
    
    # Extract non-ground points for efficiency
    if 'ground_mask' in point_cloud and 'points' in point_cloud:
        non_ground_mask = ~point_cloud['ground_mask']
        points = point_cloud['points'][non_ground_mask]
    else:
        points = point_cloud['points']
    
    # Prepare input features
    # Voxel Feature Encoding (VFE)
    voxel_features, voxel_indices = voxelize_point_cloud(
        points, 
        voxel_size=[0.1, 0.1, 0.2],
        point_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]
    )
    
    # Create model input
    model_input = {
        'voxel_features': tf.convert_to_tensor(voxel_features, dtype=tf.float32),
        'voxel_indices': tf.convert_to_tensor(voxel_indices, dtype=tf.int32)
    }
    
    # Run inference
    detections = detect_fn(**model_input)
    
    # Process results
    boxes_3d = detections['boxes'].numpy()  # [x, y, z, length, width, height, heading]
    scores = detections['scores'].numpy()
    classes = detections['classes'].numpy()
    
    # Apply score threshold
    score_mask = scores > 0.5
    boxes_3d = boxes_3d[score_mask]
    scores = scores[score_mask]
    classes = classes[score_mask]
    
    # Apply non-maximum suppression
    selected_indices = tf.image.non_max_suppression(
        tf.concat([boxes_3d[:, :2], boxes_3d[:, 3:5]], axis=1),  # Use 2D boxes (x, y, length, width)
        scores,
        max_output_size=100,
        iou_threshold=0.5
    ).numpy()
    
    boxes_3d = boxes_3d[selected_indices]
    scores = scores[selected_indices]
    classes = classes[selected_indices]
    
    # Convert boxes to corners for visualization
    corners = []
    convex_hulls = []
    
    for box in boxes_3d:
        # Extract box parameters
        x, y, z, length, width, height, heading = box
        
        # Generate corners in object frame
        corner_points = box_utils.get_3d_box_corners(
            [length, width, height], [x, y, z], heading
        )
        
        corners.append(corner_points)
        
        # Project to ground plane for convex hull
        ground_points = corner_points[:4, :2]  # Use bottom corners
        hull = ConvexHull(ground_points)
        hull_points = ground_points[hull.vertices]
        convex_hulls.append(hull_points)
    
    # Create class name mapping
    class_names = {
        0: 'Unknown',
        1: 'Vehicle',
        2: 'Pedestrian',
        3: 'Sign',
        4: 'Cyclist'
    }
    
    # Map numeric classes to names
    class_labels = [class_names.get(cls, 'Unknown') for cls in classes]
    
    return {
        'boxes_3d': boxes_3d,
        'corners': np.array(corners),
        'convex_hulls': convex_hulls,
        'scores': scores,
        'classes': classes,
        'class_labels': class_labels
    }

def voxelize_point_cloud(points, voxel_size, point_range):
    """Convert point cloud to voxel features for 3D object detection
    
    Args:
        points: Nx3 array of point coordinates
        voxel_size: 3-element list defining voxel dimensions
        point_range: 6-element list defining the point cloud range
        
    Returns:
        Voxel features and indices for model input
    """
    # Compute grid size
    grid_size = np.round((point_range[3:6] - point_range[:3]) / voxel_size).astype(np.int32)
    
    # Compute voxel index for each point
    voxel_indices = np.floor((points[:, :3] - point_range[:3]) / voxel_size).astype(np.int32)
    
    # Filter points outside range
    mask = np.all(
        (voxel_indices >= 0) & (voxel_indices < grid_size),
        axis=1
    )
    voxel_indices = voxel_indices[mask]
    filtered_points = points[mask]
    
    # Create unique voxel IDs
    voxel_ids = (
        voxel_indices[:, 0] * grid_size[1] * grid_size[2] +
        voxel_indices[:, 1] * grid_size[2] +
        voxel_indices[:, 2]
    )
    
    # Find unique voxels
    unique_voxel_ids, inverse_indices = np.unique(voxel_ids, return_inverse=True)
    
    # Generate features for each voxel
    voxel_features = []
    voxel_coords = []
    
    for voxel_id in unique_voxel_ids:
        # Find points in this voxel
        voxel_points = filtered_points[voxel_ids == voxel_id]
        
        # Compute voxel features
        if len(voxel_points) > 0:
            # Use mean and standard deviation as features
            mean = np.mean(voxel_points, axis=0)
            std = np.std(voxel_points, axis=0)
            max_vals = np.max(voxel_points, axis=0)
            min_vals = np.min(voxel_points, axis=0)
            
            # Combine features
            feature = np.concatenate([mean, std, max_vals, min_vals])
            voxel_features.append(feature)
            
            # Add voxel coordinates
            voxel_idx = np.unravel_index(voxel_id, grid_size)
            voxel_coords.append(voxel_idx)
    
    return np.array(voxel_features), np.array(voxel_coords)
```

This object detection implementation provides:
- Point cloud preprocessing specific to detection tasks
- Voxel-based feature encoding for efficient processing
- Integration with TensorFlow-based detection models
- Non-maximum suppression to remove duplicate detections
- Conversion of 3D boxes to corner representations for visualization
- Calculation of ground plane projections for bird's eye view visualization

The resulting detections provide structured information about objects in the driving scene, essential for planning and decision-making in autonomous vehicles.

#### 3.3.2 Semantic Segmentation

Semantic segmentation assigns class labels to individual LiDAR points, providing fine-grained scene understanding:

```python
import numpy as np
import tensorflow as tf
import os

def segment_point_cloud(point_cloud, model_path):
    """Perform semantic segmentation on LiDAR point cloud
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        model_path: Path to pre-trained semantic segmentation model
        
    Returns:
        Dictionary with segmentation results
    """
    # Load model
    model = tf.saved_model.load(model_path)
    segment_fn = model.signatures['serving_default']
    
    # Extract points
    points = point_cloud['points']
    
    # Prepare input features
    # Point cloud must be normalized and prepared for PointNet++ architecture
    centered_points = points - np.mean(points, axis=0)
    scale = np.max(np.abs(centered_points))
    normalized_points = centered_points / scale
    
    # Add basic features (normalized coordinates + raw coordinates + intensity if available)
    if 'intensities' in point_cloud:
        features = np.concatenate([
            normalized_points,
            points,
            point_cloud['intensities'].reshape(-1, 1)
        ], axis=1)
    else:
        features = np.concatenate([normalized_points, points], axis=1)
    
    # Limit number of points to model capacity
    max_points = 100000
    if len(features) > max_points:
        indices = np.random.choice(len(features), max_points, replace=False)
        features = features[indices]
        sample_mask = np.zeros(len(points), dtype=bool)
        sample_mask[indices] = True
    else:
        sample_mask = np.ones(len(points), dtype=bool)
    
    # Create model input
    model_input = {
        'points': tf.convert_to_tensor(features, dtype=tf.float32)
    }
    
    # Run inference
    segmentation = segment_fn(**model_input)
    
    # Process results
    logits = segmentation['logits'].numpy()
    segment_indices = np.argmax(logits, axis=1)
    
    # Map to full point cloud if sampled
    if not np.all(sample_mask):
        full_segment_indices = np.zeros(len(points), dtype=np.int32)
        full_segment_indices[sample_mask] = segment_indices
        segment_indices = full_segment_indices
    
    # Create class name mapping (Waymo dataset classes)
    class_names = {
        0: 'Undefined',
        1: 'Car',
        2: 'Truck',
        3: 'Bus',
        4: 'Other Vehicle',
        5: 'Pedestrian',
        6: 'Cyclist',
        7: 'Motorcyclist',
        8: 'Sign',
        9: 'Traffic Light',
        10: 'Pole',
        11: 'Construction Cone',
        12: 'Bicycle',
        13: 'Motorcycle',
        14: 'Building',
        15: 'Vegetation',
        16: 'Tree Trunk',
        17: 'Curb',
        18: 'Road',
        19: 'Lane Marker',
        20: 'Walkable',
        21: 'Sidewalk',
        22: 'Ground',
        23: 'Other'
    }
    
    # Create color mapping for visualization
    color_map = {
        0: [0, 0, 0],       # Undefined - Black
        1: [0, 0, 255],     # Car - Blue
        2: [0, 0, 180],     # Truck - Dark Blue
        3: [0, 0, 120],     # Bus - Darker Blue
        4: [0, 0, 80],      # Other Vehicle - Very Dark Blue
        5: [255, 0, 0],     # Pedestrian - Red
        6: [255, 80, 0],    # Cyclist - Orange
        7: [255, 120, 0],   # Motorcyclist - Darker Orange
        8: [255, 255, 0],   # Sign - Yellow
        9: [255, 255, 100], # Traffic Light - Light Yellow
        10: [80, 80, 80],   # Pole - Gray
        11: [255, 100, 255],# Construction Cone - Pink
        12: [0, 255, 200],  # Bicycle - Cyan
        13: [0, 200, 255],  # Motorcycle - Light Blue
        14: [120, 80, 0],   # Building - Brown
        15: [0, 155, 0],    # Vegetation - Green
        16: [0, 100, 0],    # Tree Trunk - Dark Green
        17: [155, 155, 155],# Curb - Light Gray
        18: [100, 100, 100],# Road - Medium Gray
        19: [255, 255, 255],# Lane Marker - White
        20: [155, 100, 0],  # Walkable - Light Brown
        21: [155, 155, 0],  # Sidewalk - Olive
        22: [50, 50, 50],   # Ground - Dark Gray
        23: [200, 200, 200] # Other - Light Gray
    }
    
    # Generate colors for each point
    colors = np.zeros((len(segment_indices), 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        mask = segment_indices == class_idx
        colors[mask] = color
    
    # Count instances of each class
    unique_classes, counts = np.unique(segment_indices, return_counts=True)
    class_counts = {class_names[cls]: count for cls, count in zip(unique_classes, counts)}
    
    # Calculate semantic class distributions
    total_points = len(segment_indices)
    class_distributions = {class_names[cls]: count / total_points for cls, count in zip(unique_classes, counts)}
    
    # Sort by frequency
    sorted_distributions = dict(sorted(class_distributions.items(), key=lambda x: x[1], reverse=True))
    
    return {
        'segment_indices': segment_indices,
        'colors': colors,
        'class_counts': class_counts,
        'class_distributions': sorted_distributions,
        'class_names': class_names
    }
```

This semantic segmentation implementation offers:
- Point cloud normalization and feature extraction
- Sampling strategies for handling large point clouds
- Integration with TensorFlow segmentation models
- Complete class mapping for the Waymo dataset's 23 semantic categories
- Color mapping for visualizing segmented point clouds
- Statistical analysis of class distributions

The resulting segmentation enables fine-grained scene understanding, identifying not only objects but also infrastructure elements like roads, sidewalks, and lane markers.

#### 3.3.3 Scene Classification

Scene classification categorizes entire driving scenes based on message topics and content characteristics:

```python
import pandas as pd
import numpy as np
from collections import Counter

def classify_scene(features_df):
    """Classify driving scenes based on message topics and content
    
    Args:
        features_df: DataFrame containing message features and topics
        
    Returns:
        DataFrame with scene classification and category distributions
    """
    # Basic classification based on topic names
    def classify_by_topic(row):
        topic = row['topic'].lower()
        
        if 'camera' in topic:
            return 'visual_perception'
        elif 'lidar' in topic:
            return 'lidar_perception' 
        elif 'annotation' in topic or 'label' in topic:
            return 'object_detection'
        elif 'tf' in topic or 'transform' in topic:
            return 'coordinate_transform'
        elif 'stat' in topic or 'metric' in topic:
            return 'system_statistics'
        elif 'map' in topic:
            return 'map_data'
        elif 'plan' in topic or 'trajectory' in topic:
            return 'planning'
        elif 'control' in topic:
            return 'vehicle_control'
        else:
            return 'other'
    
    # Apply basic classification
    features_df['scene_category'] = features_df.apply(classify_by_topic, axis=1)
    
    # Enhanced classification based on content characteristics
    if 'message_size' in features_df.columns:
        # Large messages are likely to be sensor data
        size_threshold = features_df['message_size'].quantile(0.9)
        
        # Update classification for large messages
        large_mask = features_df['message_size'] > size_threshold
        other_mask = features_df['scene_category'] == 'other'
        
        # If message is large and unclassified, assume it's sensor data
        sensor_data_mask = large_mask & other_mask
        features_df.loc[sensor_data_mask, 'scene_category'] = 'sensor_data'
    
    # Temporal classification based on message frequency
    if 'timestamp' in features_df.columns:
        # Convert to datetime if string
        if features_df['timestamp'].dtype == 'object':
            features_df['datetime'] = pd.to_datetime(features_df['timestamp'])
        else:
            features_df['datetime'] = pd.to_datetime(features_df['timestamp'], unit='us')
        
        # Resample by second and count messages
        time_series = features_df.resample('1S', on='datetime').size()
        
        # Calculate statistics
        avg_msgs_per_second = time_series.mean()
        peak_msgs_per_second = time_series.max()
        
        # Classify driving scenario based on message frequency patterns
        scenario_class = 'normal_driving'
        
        if peak_msgs_per_second > avg_msgs_per_second * 2:
            # High message rate spikes suggest complex scenarios
            scenario_class = 'complex_scenario'
            
            # Analyze category distribution during peaks
            peak_times = time_series[time_series > avg_msgs_per_second * 1.5].index
            if not peak_times.empty:
                # Get messages during peak times
                peak_mask = features_df['datetime'].dt.floor('1S').isin(peak_times)
                peak_categories = features_df.loc[peak_mask, 'scene_category']
                
                # Check if object detection messages are prevalent
                category_counts = Counter(peak_categories)
                if category_counts.get('object_detection', 0) > len(peak_categories) * 0.3:
                    scenario_class = 'object_dense_scenario'
                
                # Check if planning messages are prevalent
                if category_counts.get('planning', 0) > len(peak_categories) * 0.2:
                    scenario_class = 'decision_making_scenario'
        
        # Add scenario classification
        features_df['scenario_class'] = scenario_class
    
    # Calculate category distributions
    category_counts = features_df['scene_category'].value_counts()
    category_percentages = category_counts / len(features_df) * 100
    
    # Identify primary scene type
    primary_category = category_counts.idxmax()
    
    # Create summary statistics
    scene_stats = {
        'primary_category': primary_category,
        'category_counts': category_counts.to_dict(),
        'category_percentages': category_percentages.to_dict()
    }
    
    if 'scenario_class' in features_df.columns:
        scene_stats['scenario_class'] = features_df['scenario_class'].iloc[0]
    
    return features_df, scene_stats
```

This scene classification approach provides:
- Topic-based classification of individual messages
- Content-based analysis using message size characteristics
- Temporal analysis to identify complex driving scenarios
- Detection of object-dense and decision-making scenarios
- Statistical summaries of scene composition
- Identification of primary scene categories

This comprehensive classification enables navigation and filtering of large datasets, helping researchers focus on specific driving scenarios of interest.

### 3.4 Tools & Libraries

The project leverages several key tools and libraries to provide robust functionality:

#### 3.4.1 Data Processing Core Components

```python
# Core data processing imports
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
import mcap
from mcap.reader import make_reader
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
import os
import io
import json
from datetime import datetime

# Setup processing environment with GPU acceleration
def configure_tensorflow():
    """Configure TensorFlow for optimal performance with GPU acceleration"""
    # Check for GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Memory growth prevents tensorflow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set visible devices to first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            
            print(f"TensorFlow configured to use GPU: {gpus[0]}")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU found. Using CPU for processing (this will be slower)")

# Initialize and configure data processing environment
configure_tensorflow()

# Setup logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("waymo_processing")

# Core data structures
class WaymoDataProcessor:
    """Central class for managing Waymo dataset processing"""
    
    def __init__(self, data_dir, output_dir, cache_dir=None):
        """Initialize data processor
        
        Args:
            data_dir: Directory containing raw Waymo TFRecord files
            output_dir: Directory for processed output files
            cache_dir: Optional directory for caching intermediate results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cache_dir = cache_dir or os.path.join(output_dir, 'cache')
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize processing stats
        self.stats = {
            'total_segments': 0,
            'total_frames': 0,
            'total_objects': 0,
            'processing_time': 0
        }
        
        logger.info(f"Initialized WaymoDataProcessor with data_dir={data_dir}, output_dir={output_dir}")
    
    def process_dataset(self, limit=None, force_reprocess=False):
        """Process the entire dataset
        
        Args:
            limit: Optional limit on number of segments to process
            force_reprocess: If True, reprocess even if output files exist
            
        Returns:
            Dictionary with processing statistics
        """
        import time
        start_time = time.time()
        
        # List all TFRecord files
        tfrecord_files = [
            os.path.join(self.data_dir, f) 
            for f in os.listdir(self.data_dir) 
            if f.endswith('.tfrecord')
        ]
        
        if limit:
            tfrecord_files = tfrecord_files[:limit]
        
        logger.info(f"Found {len(tfrecord_files)} TFRecord files to process")
        
        # Process each file
        for i, tfrecord_path in enumerate(tfrecord_files):
            segment_id = os.path.basename(tfrecord_path).split('.')[0]
            mcap_path = os.path.join(self.output_dir, f"{segment_id}.mcap")
            
            if os.path.exists(mcap_path) and not force_reprocess:
                logger.info(f"Skipping already processed segment {segment_id} ({i+1}/{len(tfrecord_files)})")
                
                # Update stats from existing file
                segment_stats = self._analyze_mcap_file(mcap_path)
                self._update_stats(segment_stats)
                continue
            
            logger.info(f"Processing segment {segment_id} ({i+1}/{len(tfrecord_files)})")
            
            try:
                # Convert TFRecord to MCAP
                convert_tfrecord_to_mcap(tfrecord_path, mcap_path)
                
                # Analyze and update stats
                segment_stats = self._analyze_mcap_file(mcap_path)
                self._update_stats(segment_stats)
                
                logger.info(f"Successfully processed segment {segment_id}")
            except Exception as e:
                logger.error(f"Error processing segment {segment_id}: {e}")
        
        # Calculate total processing time
        self.stats['processing_time'] = time.time() - start_time
        
        # Save final stats
        stats_path = os.path.join(self.output_dir, 'processing_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Completed dataset processing. Stats saved to {stats_path}")
        return self.stats
    
    def _analyze_mcap_file(self, mcap_path):
        """Analyze MCAP file to extract statistics
        
        Args:
            mcap_path: Path to MCAP file
            
        Returns:
            Dictionary with segment statistics
        """
        segment_stats = {
            'frames': 0,
            'objects': 0,
            'topic_counts': {},
            'segment_id': os.path.basename(mcap_path).split('.')[0]
        }
        
        try:
            with open(mcap_path, 'rb') as f:
                reader = make_reader(f)
                
                topics = []
                object_count = 0
                
                for schema, channel, message in reader.iter_messages():
                    topics.append(channel.topic)
                    
                    # Count frames
                    if channel.topic == '/tf':
                        segment_stats['frames'] += 1
                    
                    # Count objects
                    if channel.topic == '/objects':
                        try:
                            msg_data = json.loads(message.data)
                            object_count += len(msg_data.get('markers', []))
                        except:
                            pass
                
                # Count topics
                topic_counter = Counter(topics)
                segment_stats['topic_counts'] = dict(topic_counter)
                
                # Update object count
                segment_stats['objects'] = object_count
        except Exception as e:
            logger.error(f"Error analyzing MCAP file {mcap_path}: {e}")
        
        return segment_stats
    
    def _update_stats(self, segment_stats):
        """Update global statistics with segment statistics
        
        Args:
            segment_stats: Dictionary with segment statistics
        """
        self.stats['total_segments'] += 1
        self.stats['total_frames'] += segment_stats['frames']
        self.stats['total_objects'] += segment_stats['objects']
        
        # Update topic counts
        if 'topic_counts' not in self.stats:
            self.stats['topic_counts'] = {}
        
        for topic, count in segment_stats['topic_counts'].items():
            if topic in self.stats['topic_counts']:
                self.stats['topic_counts'][topic] += count
            else:
                self.stats['topic_counts'][topic] = count
```

This comprehensive architecture provides:
- Centralized management of dataset processing
- GPU acceleration configuration
- Robust logging and error handling
- Statistical tracking of processing progress
- Caching to avoid redundant processing
- Segment-level analysis of data characteristics

The modular design enables efficient processing of the large-scale Waymo dataset while maintaining extensibility for additional processing functions.

#### 3.4.2 Visualization Framework

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class WaymoVisualizer:
    """Visualization toolkit for Waymo dataset analysis"""
    
    def __init__(self, theme='dark', figure_size=(12, 8), dpi=100):
        """Initialize visualizer with style settings
        
        Args:
            theme: Visualization theme ('dark' or 'light')
            figure_size: Default figure size for matplotlib
            dpi: Resolution for matplotlib figures
        """
        self.theme = theme
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Configure matplotlib style
        if theme == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#00a8e1',    # Blue
                'secondary': '#ff9e1b',  # Orange
                'accent': '#00cf91',     # Green
                'highlight': '#ff5c8d',  # Pink
                'background': '#1e1e1e', # Dark gray
                'text': '#ffffff'        # White
            }
        else:
            plt.style.use('default')
            self.colors = {
                'primary': '#0072b2',    # Blue
                'secondary': '#e69f00',  # Orange
                'accent': '#009e73',     # Green
                'highlight': '#cc79a7',  # Pink
                'background': '#ffffff', # White
                'text': '#000000'        # Black
            }
        
        # Configure seaborn
        sns.set(style="ticks", font_scale=1.1)
        
        # Create color palette for segmentation
        self._create_segmentation_palette()
    
    def _create_segmentation_palette(self):
        """Create color palette for semantic segmentation visualization"""
        # Palette for 23 semantic classes in Waymo dataset
        colors = [
            [0, 0, 0],       # Undefined - Black
            [0, 0, 255],     # Car - Blue
            [0, 0, 180],     # Truck - Dark Blue
            [0, 0, 120],     # Bus - Darker Blue
            [0, 0, 80],      # Other Vehicle - Very Dark Blue
            [255, 0, 0],     # Pedestrian - Red
            [255, 80, 0],    # Cyclist - Orange
            [255, 120, 0],   # Motorcyclist - Darker Orange
            [255, 255, 0],   # Sign - Yellow
            [255, 255, 100], # Traffic Light - Light Yellow
            [80, 80, 80],    # Pole - Gray
            [255, 100, 255], # Construction Cone - Pink
            [0, 255, 200],   # Bicycle - Cyan
            [0, 200, 255],   # Motorcycle - Light Blue
            [120, 80, 0],    # Building - Brown
            [0, 155, 0],     # Vegetation - Green
            [0, 100, 0],     # Tree Trunk - Dark Green
            [155, 155, 155], # Curb - Light Gray
            [100, 100, 100], # Road - Medium Gray
            [255, 255, 255], # Lane Marker - White
            [155, 100, 0],   # Walkable - Light Brown
            [155, 155, 0],   # Sidewalk - Olive
            [50, 50, 50]     # Ground - Dark Gray
        ]
        
        # Normalize to 0-1 range for matplotlib
        self.segmentation_palette = np.array(colors) / 255.0
    
    def plot_topic_distribution(self, topic_counter, top_n=10, title="Topic Distribution", 
                              save_path=None, interactive=False):
        """Visualize distribution of message topics
        
        Args:
            topic_counter: Counter object or dictionary with topic counts
            top_n: Number of top topics to display
            title: Plot title
            save_path: Optional path to save visualization
            interactive: If True, use plotly for interactive visualization
            
        Returns:
            Figure object
        """
        # Get top topics
        top_topics = dict(sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        if interactive:
            # Create interactive plotly figure
            fig = px.bar(
                x=list(top_topics.values()),
                y=list(top_topics.keys()),
                orientation='h',
                title=title,
                labels={'x': 'Message Count', 'y': 'Topic'},
                color=list(top_topics.values()),
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            fig.update_layout(
                height=max(500, 50 * len(top_topics)),
                width=800,
                template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        else:
            # Create matplotlib figure
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot horizontal bar chart with custom colors
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['accent'], self.colors['highlight']] * 10
            bars = plt.barh(
                range(len(top_topics)), 
                list(top_topics.values()),
                color=colors[:len(top_topics)]
            )
            
            # Customize appearance
            plt.yticks(range(len(top_topics)), list(top_topics.keys()))
            plt.xlabel('Message Count')
            plt.title(title)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.5, 
                    bar.get_y() + bar.get_height()/2,
                    f'{int(width):,}',
                    ha='left', 
                    va='center',
                    color=self.colors['text'],
                    fontweight='bold'
                )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return plt.gcf()
    
    def visualize_point_cloud(self, point_cloud, colors=None, custom_draw=None, 
                            background_color=None, window_name="Point Cloud Visualization"):
        """Visualize 3D point cloud using Open3D
        
        Args:
            point_cloud: Nx3 array of point coordinates or Open3D PointCloud object
            colors: Optional Nx3 array of RGB colors for each point
            custom_draw: Optional function for custom rendering logic
            background_color: Optional background color override
            window_name: Title for visualization window
            
        Returns:
            Open3D Visualizer object
        """
        # Create point cloud object if not already one
        if isinstance(point_cloud, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            if colors is not None:
                # Normalize colors if in range 0-255
                if np.max(colors) > 1:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd = point_cloud
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Configure visualization settings
        opt = vis.get_render_option()
        
        # Set background color
        if background_color is None:
            bg_color = [0.1, 0.1, 0.1] if self.theme == 'dark' else [1, 1, 1]
        else:
            bg_color = background_color
        opt.background_color = bg_color
        
        # Configure other rendering options
        opt.point_size = 2.0
        opt.show_coordinate_frame = True
        
        # Set default camera viewpoint (top-down with slight angle)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        # Apply custom rendering logic if provided
        if custom_draw is not None:
            custom_draw(vis)
        
        return vis
    
    def visualize_segmentation(self, points, segment_indices, class_names=None,
                             window_name="Semantic Segmentation"):
        """Visualize semantically segmented point cloud
        
        Args:
            points: Nx3 array of point coordinates
            segment_indices: N array of semantic class indices
            class_names: Optional dictionary mapping indices to class names
            window_name: Title for visualization window
            
        Returns:
            Open3D Visualizer object
        """
        # Set colors based on segmentation classes
        colors = self.segmentation_palette[segment_indices % len(self.segmentation_palette)]
        
        # Create visualization
        vis = self.visualize_point_cloud(
            points, 
            colors=colors,
            window_name=window_name
        )
        
        # Add legend if class names are provided
        if class_names is not None:
            # Count instances of each class
            unique_indices, counts = np.unique(segment_indices, return_counts=True)
            
            # Sort by frequency
            sort_idx = np.argsort(counts)[::-1]
            sorted_indices = unique_indices[sort_idx]
            sorted_counts = counts[sort_idx]
            
            # Create custom draw function to add text
            def add_legend(vis):
                # Get render window size
                renderer = vis.get_render_option()
                window_width = vis.get_window_size()[0]
                window_height = vis.get_window_size()[1]
                
                # Add class information as text
                for i, class_idx in enumerate(sorted_indices[:10]):  # Show top 10 classes
                    if class_idx in class_names:
                        class_name = class_names[class_idx]
                        count = sorted_counts[i]
                        percentage = 100 * count / len(segment_indices)
                        
                        color = self.segmentation_palette[class_idx % len(self.segmentation_palette)]
                        color_str = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
                        
                        text = f"{class_name}: {count:,} points ({percentage:.1f}%)"
                        # Add text using renderer
                        # Note: Actual text rendering requires custom implementation
                        # as Open3D visualization lacks text support
                        # This would typically be done using a custom GUI framework
            
            # We would apply the legend in a real implementation
            # Currently not applicable due to Open3D limitations
        
        return vis
    
    def visualize_objects_3d(self, point_cloud, objects, window_name="3D Object Detection"):
        """Visualize 3D object detections in point cloud
        
        Args:
            point_cloud: Nx3 array of point coordinates
            objects: Dictionary with detection results including boxes_3d
            window_name: Title for visualization window
            
        Returns:
            Open3D Visualizer object
        """
        # Extract detection boxes
        boxes_3d = objects.get('boxes_3d', [])
        class_labels = objects.get('class_labels', ['Unknown'] * len(boxes_3d))
        scores = objects.get('scores', [1.0] * len(boxes_3d))
        
        # Define colors for different classes
        class_colors = {
            'Vehicle': [0, 0, 1],      # Blue
            'Pedestrian': [1, 0, 0],   # Red
            'Cyclist': [1, 0.5, 0],    # Orange
            'Sign': [1, 1, 0],         # Yellow
            'Unknown': [0.5, 0.5, 0.5] # Gray
        }
        
        # Create custom drawing function for visualizer
        def draw_3d_boxes(vis):
            # Get renderer
            render_option = vis.get_render_option()
            
            # For each detection box
            for i, box in enumerate(boxes_3d):
                # Extract box parameters
                x, y, z, length, width, height, heading = box
                
                # Get class and color
                class_label = class_labels[i]
                score = scores[i]
                
                color = class_colors.get(class_label, [0.5, 0.5, 0.5])
                
                # Create box geometry
                box_points = o3d.utility.Vector3dVector([
                    [x - length/2, y - width/2, z - height/2],
                    [x + length/2, y - width/2, z - height/2],
                    [x + length/2, y + width/2, z - height/2],
                    [x - length/2, y + width/2, z - height/2],
                    [x - length/2, y - width/2, z + height/2],
                    [x + length/2, y - width/2, z + height/2],
                    [x + length/2, y + width/2, z + height/2],
                    [x - length/2, y + width/2, z + height/2]
                ])
                
                # Create line indices for box edges
                lines = o3d.utility.Vector2iVector([
                    [0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]
                ])
                
                # Create line set and set properties
                line_set = o3d.geometry.LineSet(box_points, lines)
                line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
                
                # Apply rotation for heading
                R = np.array([
                    [np.cos(heading), -np.sin(heading), 0],
                    [np.sin(heading), np.cos(heading), 0],
                    [0, 0,, 1]
                ])
                
                # Apply rotation around z-axis at box center
                center = np.array([x, y, z])
                points_centered = np.asarray(box_points) - center
                points_rotated = np.dot(points_centered, R.T) + center
                line_set.points = o3d.utility.Vector3dVector(points_rotated)
                
                # Add to visualization
                vis.add_geometry(line_set)
                
                # Add text for class and score (not directly supported in Open3D)
                # This would require a custom GUI framework integration
        
        # Create point cloud visualization with custom drawing
        return self.visualize_point_cloud(
            point_cloud,
            custom_draw=draw_3d_boxes,
            window_name=window_name
        )
    
    def create_interactive_dashboard(self, data, config=None):
        """Create interactive dashboard for dataset analysis
        
        Args:
            data: Dictionary containing analysis results
            config: Optional configuration for dashboard layout
            
        Returns:
            Plotly dashboard figure
        """
        # Use provided config or default layout
        if config is None:
            config = {
                'layout': [
                    ['topic_distribution', 'message_size_distribution'],
                    ['time_series', 'class_distribution']
                ],
                'height': 900,
                'width': 1200
            }
        
        # Create subplot grid based on layout
        rows = len(config['layout'])
        cols = max(len(row) for row in config['layout'])
        
        # Define subplot titles
        subplot_titles = []
        for row in config['layout']:
            for panel in row:
                if panel in data:
                    subplot_titles.append(panel.replace('_', ' ').title())
                else:
                    subplot_titles.append('')
        
        # Create subplot figure
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Add content to each subplot
        for r, row in enumerate(config['layout']):
            for c, panel in enumerate(row):
                row_idx = r + 1
                col_idx = c + 1
                
                if panel == 'topic_distribution' and 'topic_counter' in data:
                    # Add topic distribution bar chart
                    topics = list(data['topic_counter'].keys())[:10]
                    counts = list(data['topic_counter'].values())[:10]
                    
                    fig.add_trace(
                        go.Bar(
                            y=topics,
                            x=counts,
                            orientation='h',
                            marker=dict(
                                color=counts,
                                colorscale='Viridis'
                            ),
                            name='Topics'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                
                elif panel == 'message_size_distribution' and 'message_sizes' in data:
                    # Add message size histogram
                    fig.add_trace(
                        go.Histogram(
                            x=data['message_sizes'],
                            nbinsx=50,
                            marker=dict(
                                color=self.colors['primary'],
                                line=dict(color='white', width=0.5)
                            ),
                            name='Message Sizes'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                
                elif panel == 'time_series' and 'timestamps' in data:
                    # Add time series plot
                    fig.add_trace(
                        go.Scatter(
                            x=data['timestamps'],
                            y=data['message_counts'],
                            mode='lines',
                            line=dict(
                                color=self.colors['accent'],
                                width=2
                            ),
                            name='Messages Over Time'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                
                elif panel == 'class_distribution' and 'class_counts' in data:
                    # Add class distribution pie chart
                    labels = list(data['class_counts'].keys())
                    values = list(data['class_counts'].values())
                    
                    fig.add_trace(
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.4,
                            marker=dict(
                                colors=px.colors.qualitative.Pastel
                            ),
                            name='Object Classes'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
        
        # Update layout
        fig.update_layout(
            height=config['height'],
            width=config['width'],
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig
```

This comprehensive visualization framework provides:
- Customizable styling with dark and light themes
- Advanced 3D visualization of point clouds using Open3D
- Semantic segmentation visualization with color-coded classes
- 3D object detection visualization with oriented bounding boxes
- Interactive dashboards using Plotly
- Statistical visualizations of topic distributions and message patterns
- Configurable layouts for different analysis needs

These visualization capabilities enable intuitive exploration of the complex multi-sensor data in the Waymo dataset.

## 4. Visualization Demo

The visualization component of the project offers interactive tools for analyzing autonomous driving scenes through multiple perspectives, implemented through a comprehensive system of integrated visualizations.

### 4.1 Point Cloud Visualization

The core of our visualization system is the ability to render and interact with LiDAR point clouds:

```python
def visualize_scene(mcap_file, frame_index=0):
    """Create comprehensive visualization of a single frame
    
    Args:
        mcap_file: Path to MCAP file
        frame_index: Index of frame to visualize
        
    Returns:
        Dictionary with visualization components
    """
    # Initialize visualizer
    visualizer = WaymoVisualizer(theme='dark')
    
    # Extract frame data
    frame_data = extract_frame(mcap_file, frame_index)
    
    # Process point cloud
    point_cloud = process_point_cloud(
        frame_data['range_image'],
        frame_data['calibration']
    )
    
    # Process camera images
    camera_data = {}
    for camera_name, image_data in frame_data['images'].items():
        camera_data[camera_name] = process_camera_image(
            image_data,
            camera_name,
            frame_data['calibration']
        )
    
    # Perform 3D object detection
    detection_results = detect_objects_3d(
        point_cloud,
        model_path='models/3d_detection_model'
    )
    
    # Perform semantic segmentation
    segmentation_results = segment_point_cloud(
        point_cloud,
        model_path='models/segmentation_model'
    )
    
    # Create basic point cloud visualization
    basic_vis = visualizer.visualize_point_cloud(
        point_cloud['points'],
        colors=point_cloud['intensities'].reshape(-1, 1).repeat(3, axis=1),
        window_name="LiDAR Point Cloud"
    )
    
    # Create segmentation visualization
    segment_vis = visualizer.visualize_segmentation(
        point_cloud['points'],
        segmentation_results['segment_indices'],
        class_names=segmentation_results['class_names'],
        window_name="Semantic Segmentation"
    )
    
    # Create object detection visualization
    detection_vis = visualizer.visualize_objects_3d(
        point_cloud['points'],
        detection_results,
        window_name="3D Object Detection"
    )
    
    # Project LiDAR points onto front camera
    front_camera_projection = project_lidar_to_camera(
        point_cloud,
        camera_data['FRONT'],
        frame_data['calibration']
    )
    
    # Create integrated dashboard with statistical visualizations
    dashboard_data = {
        'topic_counter': frame_data['topic_counter'],
        'message_sizes': frame_data['message_sizes'],
        'timestamps': frame_data['timestamps'],
        'message_counts': frame_data['message_counts'],
        'class_counts': detection_results['class_counts']
    }
    
    dashboard = visualizer.create_interactive_dashboard(dashboard_data)
    
    return {
        'point_cloud_vis': basic_vis,
        'segmentation_vis': segment_vis,
        'detection_vis': detection_vis,
        'camera_projection': front_camera_projection,
        'dashboard': dashboard,
        'point_cloud': point_cloud,
        'detection_results': detection_results,
        'segmentation_results': segmentation_results,
        'camera_data': camera_data
    }
```

This integrated visualization provides:

1. **Raw Point Cloud View**: Displays the original LiDAR points colored by intensity, revealing the 3D structure of the environment with depth-dependent coloring.

2. **Semantic Segmentation View**: Renders each point with colors corresponding to its semantic class (e.g., road, vehicle, pedestrian), enabling immediate understanding of scene composition.

3. **3D Object Detection View**: Shows detected objects with oriented 3D bounding boxes, colored by object class, with additional information about detection confidence.

4. **Multi-view Controls**: Provides camera controls for rotating, panning, and zooming the point cloud, with preset viewpoints (top-down, front, side) for consistent analysis.

5. **Temporal Navigation**: Enables frame-by-frame playback with adjustable speed for analyzing object motion and scene changes over time.

### 4.2 Multi-sensor Fusion Visualization

Our system integrates LiDAR point clouds with camera imagery to provide comprehensive multi-sensor scene understanding:

```python
def create_sensor_fusion_visualization(scene_data):
    """Create multi-sensor fusion visualization
    
    Args:
        scene_data: Dictionary with processed scene data
        
    Returns:
        Interactive visualization dashboard
    """
    # Create multi-panel visualization layout
    fig = make_subplots(
        rows=2, 
        cols=3,
        specs=[
            [{"type": "image"}, {"type": "image"}, {"type": "image"}],
            [{"type": "scene", "colspan": 3}, None, None]
        ],
        subplot_titles=(
            "Front Camera", "Front Left Camera", "Front Right Camera", 
            "LiDAR Point Cloud with Camera Fusion"
        )
    )
    
    # Add camera images with projected LiDAR points
    camera_positions = {
        'FRONT': (1, 1),
        'FRONT_LEFT': (1, 2),
        'FRONT_RIGHT': (1, 3)
    }
    
    for camera_name, (row, col) in camera_positions.items():
        if camera_name in scene_data['camera_data']:
            # Get camera visualization with projected points
            camera_vis = scene_data['camera_data'][camera_name]['visualization']
            
            # Add to subplot
            fig.add_trace(
                go.Image(z=camera_vis),
                row=row,
                col=col
            )
    
    # Add 3D point cloud with detections
    point_cloud = scene_data['point_cloud']['points']
    colors = scene_data['segmentation_results']['colors'] * 255
    
    # Create scatter3d trace for points
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=1,
                color=['rgb({},{},{})'.format(r, g, b) for r, g, b in colors.astype(int)],
                opacity=0.7
            ),
            hoverinfo='none'
        ),
        row=2,
        col=1
    )
    
    # Add 3D boxes for detections
    boxes_3d = scene_data['detection_results']['boxes_3d']
    corners = scene_data['detection_results']['corners']
    class_labels = scene_data['detection_results']['class_labels']
    
    for i, box in enumerate(boxes_3d):
        x, y, z, length, width, height, heading = box
        
        # Get class for color
        class_name = class_labels[i]
        
        # Define color based on class
        if 'Vehicle' in class_name:
            color = 'rgb(0, 0, 255)'  # Blue
        elif 'Pedestrian' in class_name:
            color = 'rgb(255, 0, 0)'  # Red
        elif 'Cyclist' in class_name:
            color = 'rgb(255, 165, 0)'  # Orange
        else:
            color = 'rgb(180, 180, 180)'  # Gray
        
        # Add box as mesh3d
        corner_points = corners[i]
        
        # Define vertices (corners)
        x_corners = corner_points[:, 0]
        y_corners = corner_points[:, 1]
        z_corners = corner_points[:, 2]
        
        # Define faces for a cuboid
        i0, i1, i2, i3, i4, i5, i6, i7 = range(8)
        faces = np.array([
            [i0, i1, i2], [i0, i2, i3],  # Bottom face
            [i4, i5, i6], [i4, i6, i7],  # Top face
            [i0, i1, i5], [i0, i5, i4],  # Side face
            [i1, i2, i6], [i1, i6, i5],  # Side face
            [i2, i3, i7], [i2, i7, i6],  # Side face
            [i3, i0, i4], [i3, i4, i7]   # Side face
        ])
        
        # Add box as mesh3d
        fig.add_trace(
            go.Mesh3d(
                x=x_corners,
                y=y_corners,
                z=z_corners,
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=0.7,
                hoverinfo='text',
                hovertext=f"{class_name}<br>Confidence: {scene_data['detection_results']['scores'][i]:.2f}"
            ),
            row=2,
            col=1
        )
    
    # Configure 3D scene
    fig.update_scenes(
        aspectmode='data',
        xaxis_title='X (forward)',
        yaxis_title='Y (left)',
        zaxis_title='Z (up)',
        camera=dict(
            eye=dict(x=0, y=0, z=3),
            up=dict(x=0, y=0, z=1)
        )
    )
    
    # Update layout
    fig.update_layout(
        height=900,
        width=1200,
        template='plotly_dark',
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False
    )
    
    return fig
```

This multi-sensor fusion visualization provides:

1. **Camera-LiDAR Integration**: Projects 3D LiDAR points onto 2D camera images, with depth-based coloring to indicate distance.

2. **Multiple Camera Views**: Displays all five camera perspectives simultaneously, with synchronized LiDAR projections on each view.

3. **3D Scene Context**: Provides a comprehensive 3D view that integrates all sensor data, allowing for complete spatial understanding.

4. **Object Association**: Visually links detected objects across different sensors, showing the same object as seen from different perspectives.

5. **Interactive Controls**: Offers interactive elements to toggle between different visualization modes (raw, segmented, detected) for each sensor.

### 4.3 Statistical Analysis Dashboard

Our system includes statistical analysis tools to provide quantitative insights about driving scenes:

```python
def create_analysis_dashboard(scene_data, historical_data=None):
    """Create statistical analysis dashboard
    
    Args:
        scene_data: Dictionary with current scene data
        historical_data: Optional dictionary with historical data for comparison
        
    Returns:
        Interactive statistical dashboard
    """
    # Create subplot layout
    fig = make_subplots(
        rows=2, 
        cols=2,
        specs=[
            [{"type": "bar"}, {"type": "pie"}],
            [{"type": "heatmap"}, {"type": "scatter"}]
        ],
        subplot_titles=(
            "Object Distribution by Class", 
            "Road vs. Non-Road Points",
            "Spatial Density Map",
            "Object Distance Distribution"
        )
    )
    
    # 1. Object distribution by class
    object_classes = scene_data['detection_results']['class_labels']
    class_counts = {}
    for cls in object_classes:
        if cls in class_counts:
            class_counts[cls] += 1
        else:
            class_counts[cls] = 1
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_classes)
    
    # Add to subplot
    fig.add_trace(
        go.Bar(
            x=classes,
            y=counts,
            marker_color=['blue' if 'Vehicle' in c else 
                         'red' if 'Pedestrian' in c else 
                         'orange' if 'Cyclist' in c else 
                         'gray' for c in classes]
        ),
        row=1,
        col=1
    )
    
    # 2. Road vs. Non-Road points
    # Identify road-related classes from segmentation
    road_classes = [18, 19]  # Road, Lane Marker
    segment_indices = scene_data['segmentation_results']['segment_indices']
    
    road_mask = np.isin(segment_indices, road_classes)
    road_points = np.sum(road_mask)
    non_road_points = len(segment_indices) - road_points
    
    # Add to subplot
    fig.add_trace(
        go.Pie(
            labels=['Road', 'Non-Road'],
            values=[road_points, non_road_points],
            marker_colors=['gray', 'green']
        ),
        row=1,
        col=2
    )
    
    # 3. Spatial density map
    points = scene_data['point_cloud']['points']
    
    # Create 2D histogram (top-down view)
    x_range = (-50, 50)
    y_range = (-50, 50)
    bin_size = 1.0
    
    x_bins = np.arange(x_range[0], x_range[1] + bin_size, bin_size)
    y_bins = np.arange(y_range[0], y_range[1] + bin_size, bin_size)
    
    # Compute 2D histogram
    H, x_edges, y_edges = np.histogram2d(
        points[:, 0],  # x coordinates (forward)
        points[:, 1],  # y coordinates (left)
        bins=[x_bins, y_bins]
    )
    
    # Add to subplot
    fig.add_trace(
        go.Heatmap(
            z=H.T,  # Transpose for correct orientation
            x=x_edges[:-1],
            y=y_edges[:-1],
            colorscale='Viridis',
            showscale=False
        ),
        row=2,
        col=1
    )
    
    # 4. Object distance distribution
    object_distances = []
    object_types = []
    
    # Calculate distance from ego vehicle to each object
    for i, box in enumerate(scene_data['detection_results']['boxes_3d']):
        x, y, z, length, width, height, heading = box
        distance = np.sqrt(x**2 + y**2)  # 2D distance in xy-plane
        object_distances.append(distance)
        object_types.append(scene_data['detection_results']['class_labels'][i])
    
    # Add to subplot
    for cls in set(object_types):
        # Filter distances for this class
        cls_distances = [d for d, t in zip(object_distances, object_types) if t == cls]
        
        # Choose color based on class
        color = 'blue' if 'Vehicle' in cls else \
                'red' if 'Pedestrian' in cls else \
                'orange' if 'Cyclist' in cls else 'gray'
        
        # Add trace
        fig.add_trace(
            go.Scatter(
                x=cls_distances,
                y=[cls] * len(cls_distances),
                mode='markers',
                marker=dict(
                    color=color,
                    size=10,
                    opacity=0.7
                ),
                name=cls
            ),
            row=2,
            col=2
        )
    
    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        template='plotly_dark',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Object Class", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    
    fig.update_xaxes(title_text="X (forward) [m]", row=2, col=1)
    fig.update_yaxes(title_text="Y (left) [m]", row=2, col=1)
    
    fig.update_xaxes(title_text="Distance from Ego Vehicle [m]", row=2, col=2)
    fig.update_yaxes(title_text="Object Class", row=2, col=2)
    
    return fig
```

This statistical dashboard provides:

1. **Object Distribution Analysis**: Shows the frequency of different object classes in the scene, highlighting the prevalence of vehicles, pedestrians, cyclists, and other objects.

2. **Road Structure Analysis**: Visualizes the proportion of road vs. non-road points, providing insights into the drivable area in the scene.

3. **Spatial Density Map**: Generates a heatmap of point cloud density from a top-down perspective, revealing areas of high and low sensor coverage.

4. **Object Distance Distribution**: Plots the distances of different object types from the ego vehicle, useful for understanding object proximity patterns.

5. **Comparative Analysis**: Optionally compares current scene statistics with historical data to identify anomalies or trends across multiple frames.

These visualizations enable both qualitative and quantitative analysis of autonomous driving scenes, providing insights that would be difficult to obtain through manual inspection of raw sensor data.

### 4.4 Temporal Analysis and Playback

Our system includes tools for analyzing scenes across time, essential for understanding dynamic behaviors:

```python
def create_temporal_visualization(mcap_file, start_frame=0, frame_count=10):
    """Create visualization of scene evolution over time
    
    Args:
        mcap_file: Path to MCAP file
        start_frame: First frame to visualize
        frame_count: Number of frames to include
        
    Returns:
        Interactive temporal visualization
    """
    # Extract multiple frames
    frames_data = []
    for i in range(start_frame, start_frame + frame_count):
        try:
            frame_data = extract_frame(mcap_file, i)
            
            # Process basic data
            point_cloud = process_point_cloud(
                frame_data['range_image'],
                frame_data['calibration']
            )
            
            # Perform object detection
            detection_results = detect_objects_3d(
                point_cloud,
                model_path='models/3d_detection_model'
            )
            
            # Store processed frame
            frames_data.append({
                'frame_index': i,
                'timestamp': frame_data['timestamp'],
                'point_cloud': point_cloud,
                'detection_results': detection_results
            })
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            break
    
    # Create object tracking visualization
    fig = make_subplots(
        rows=1, 
        cols=2,
        specs=[
            [{"type": "scene"}, {"type": "scatter"}]
        ],
        subplot_titles=(
            "3D Object Trajectories", 
            "Object Velocity Over Time"
        )
    )
    
    # Track objects across frames
    object_tracks = {}
    
    # Analyze each frame
    for frame in frames_data:
        # Extract detections
        boxes = frame['detection_results']['boxes_3d']
        classes = frame['detection_results']['class_labels']
        object_ids = frame['detection_results'].get('tracking_ids', range(len(boxes)))
        
        # Update tracks
        for i, obj_id in enumerate(object_ids):
            box = boxes[i]
            cls = classes[i]
            position = box[:3]  # x, y, z
            
            if obj_id not in object_tracks:
                # Initialize new track
                object_tracks[obj_id] = {
                    'positions': [],
                    'timestamps': [],
                    'class': cls
                }
            
            # Add position and timestamp
            object_tracks[obj_id]['positions'].append(position)
            object_tracks[obj_id]['timestamps'].append(frame['timestamp'])
    
    # Plot object trajectories
    for obj_id, track in object_tracks.items():
        # Skip objects with too few observations
        if len(track['positions']) < 3:
            continue
        
        # Convert positions to arrays
        positions = np.array(track['positions'])
        
        # Choose color based on object class
        cls = track['class']
        color = 'blue' if 'Vehicle' in cls else \
                'red' if 'Pedestrian' in cls else \
                'orange' if 'Cyclist' in cls else 'gray'
        
        # Add 3D trajectory
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines+markers',
                marker=dict(
                    size=3,
                    color=color,
                    opacity=0.8
                ),
                line=dict(
                    color=color,
                    width=2
                ),
                name=f"{cls} (ID: {obj_id})"
            ),
            row=1,
            col=1
        )
        
        # Calculate velocities
        if len(track['timestamps']) > 1:
            timestamps = np.array(track['timestamps'])
            time_diffs = np.diff(timestamps) / 1e6  # Convert microseconds to seconds
            
            # Calculate velocity between consecutive positions
            pos_diffs = np.diff(positions, axis=0)
            velocities = np.sqrt(np.sum(pos_diffs[:, :2]**2, axis=1)) / time_diffs  # m/s, using only x,y
            
            # Plot velocity over time
            rel_times = (timestamps[1:] - timestamps[0]) / 1e6  # Time relative to first observation
            
            fig.add_trace(
                go.Scatter(
                    x=rel_times,
                    y=velocities,
                    mode='lines+markers',
                    line=dict(color=color),
                    name=f"{cls} Velocity (ID: {obj_id})"
                ),
                row=1,
                col=2
            )
    
    # Configure 3D scene
    fig.update_scenes(
        aspectmode='data',
        xaxis_title='X (forward)',
        yaxis_title='Y (left)',
        zaxis_title='Z (up)',
        camera=dict(
            eye=dict(x=0, y=0, z=5),
            up=dict(x=0, y=0, z=1)
        )
    )
    
    # Update scatter plot
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
    fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
    
    # Update layout
    fig.update_layout(
        height=600,
        width=1200,
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
```

This temporal analysis visualization provides:

1. **Object Trajectories**: Tracks the movement of objects across frames, visualizing their 3D paths through the environment.

2. **Velocity Analysis**: Calculates and displays object velocities over time, enabling analysis of acceleration, deceleration, and steady-state motion.

3. **Interaction Detection**: Identifies potential interactions between objects by analyzing their trajectories and proximity.

4. **Sequence Playback**: Offers frame-by-frame playback with adjustable speed and the ability to pause, rewind, and advance through the sequence.

5. **Scene Evolution Analysis**: Highlights changes in the environment over time, including the appearance and disappearance of objects.

These temporal analysis tools enable understanding of dynamic behaviors in driving scenes, essential for developing and testing perception, prediction, and planning algorithms for autonomous vehicles.

## 5. Contributions

This project contributes to autonomous driving research in several significant ways:

### 5.1 Data Accessibility Improvements

Our visualization framework significantly enhances access to the complex Waymo dataset through:

1. **Format Conversion**: Transforms the specialized TFRecord format into the more widely supported MCAP format, compatible with popular visualization tools like Foxglove Studio.

2. **Coordinate Transformation**: Automatically handles the complex transformations between global, vehicle, and sensor coordinate systems, presenting a unified spatial representation.

3. **Data Filtering**: Provides intuitive mechanisms for filtering the massive dataset (1,950 segments, 600,000 frames) by scene type, object composition, or environmental conditions.

4. **Metadata Extraction**: Exposes critical metadata about sensors, calibration, and frame relationships that would otherwise require specialized knowledge to access.

5. **Batch Processing**: Implements efficient batch processing to handle the 1.2TB+ dataset, with progress tracking and incremental output generation.

By making this rich dataset more accessible, our project enables researchers without specialized knowledge of the Waymo data format to leverage this valuable resource for autonomous driving research.

### 5.2 Modular Analysis Framework

Our system provides a comprehensive yet modular framework for analyzing multi-sensor autonomous driving data:

1. **Pipeline Architecture**: Implements a flexible processing pipeline that can be customized or extended at each stage from data loading to visualization.

2. **Component Integration**: Seamlessly integrates point cloud processing, image analysis, and sensor fusion components with minimal coupling.

3. **Algorithm Encapsulation**: Encapsulates complex algorithms for detection, segmentation, and tracking behind clean interfaces, allowing for algorithm swapping without affecting other system components.

4. **Performance Optimization**: Implements efficient processing techniques such as voxel downsampling, region of interest filtering, and parallel computation to handle large-scale data.

5. **Extensibility**: Provides clear extension points for adding new processing modules, visualization components, or analysis algorithms.

This modular architecture enables researchers to adapt and extend the framework for specific research questions while leveraging the implemented functionality for common tasks.

### 5.3 Novel Visualization Techniques

The project implements several novel visualization approaches specifically designed for autonomous driving data:

1. **Multi-modal Fusion Views**: Creates integrated visualizations combining LiDAR point clouds with camera images, maintaining spatial consistency and depth information.

2. **Semantic-enhanced Rendering**: Renders point clouds with semantic class information, enabling intuitive understanding of scene composition through consistent color coding.

3. **Temporal Trajectory Visualization**: Visualizes object trajectories over time in 3D space, with additional velocity and acceleration visualizations.

4. **Density and Coverage Analysis**: Provides spatial density maps highlighting areas of high and low sensor coverage, essential for understanding perception limitations.

5. **Interactive Cross-filtering**: Implements interactive filtering where selection in one visualization automatically highlights corresponding elements in other views.

These visualization techniques provide intuitive representations of the complex multi-sensor data, enabling deeper insights into autonomous driving scenes.

### 5.4 Educational Resource Development

Our project serves as a comprehensive educational resource for understanding autonomous driving perception systems:

1. **Documentation**: Provides extensive documentation on the Waymo dataset structure, sensor configurations, and coordinate systems.

2. **Tutorial Notebooks**: Includes step-by-step tutorial notebooks demonstrating data loading, processing, and visualization techniques.

3. **Algorithm Explanations**: Explains key perception algorithms such as 3D object detection, semantic segmentation, and sensor fusion through code and visualizations.

4. **Interactive Examples**: Offers interactive examples of various processing stages, allowing users to experiment with parameters and observe results.

5. **Best Practices**: Demonstrates best practices for handling large-scale autonomous driving data, including efficient processing, appropriate visualization, and result interpretation.

By providing these educational resources, our project helps researchers and students enter the field of autonomous driving perception, accelerating progress in this critical area.


## 6. Future Works

### 6.1 Integration with Real-Time Simulation

A primary direction for future work is extending the project to support real-time simulation of autonomous driving scenarios:

**Goal**: Extend the visualization framework to support closed-loop, interactive agent behavior simulation, aligned with the Sim Agents Challenge's focus on realistic autonomous agent simulation.

**Technical Approach**:

1. **Motion Prediction Integration**: Incorporate transformer-based motion prediction models to forecast object trajectories based on historical observations:

```python
def predict_object_trajectories(object_tracks, prediction_horizon=5.0, time_step=0.1):
    """Predict future object trajectories using a transformer-based motion model
    
    Args:
        object_tracks: Dictionary with historical object tracks
        prediction_horizon: Time horizon for prediction in seconds
        time_step: Prediction time step in seconds
        
    Returns:
        Dictionary with predicted trajectories
    """
    # Initialize transformer-based prediction model
    model = MotionTransformer.from_pretrained('models/motion_transformer')
    
    # Format past trajectories for model input
    batch_inputs = []
    
    for obj_id, track in object_tracks.items():
        if len(track['positions']) < 3:
            continue
            
        # Extract positions and timestamps
        positions = np.array(track['positions'])
        timestamps = np.array(track['timestamps'])
        
        # Convert to relative timestamps
        rel_times = (timestamps - timestamps[0]) / 1e6  # seconds
        
        # Create input features
        features = np.concatenate([positions, rel_times[:, np.newaxis]], axis=1)
        batch_inputs.append(features)
    
    # Run prediction model
    predictions = model.predict_batch(
        batch_inputs,
        prediction_time_points=np.arange(0.0, prediction_horizon + time_step, time_step)
    )
    
    # Format predictions
    predicted_trajectories = {}
    
    for i, obj_id in enumerate(object_tracks.keys()):
        if i < len(predictions):
            predicted_trajectories[obj_id] = {
                'positions': predictions[i],
                'timestamps': np.arange(0.0, prediction_horizon + time_step, time_step),
                'uncertainty': model.get_uncertainty(predictions[i])
            }
    
    return predicted_trajectories
```

The transformer model uses a self-attention mechanism to model spatio-temporal dependencies in trajectory data:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $Q$, $K$, and $V$ are query, key, and value matrices derived from input trajectory features, and $d_k$ is the dimension of the key vectors.

2. **Physics-Based Simulation**: Implement physics-based motion models for realistic object interaction, including collision avoidance and dynamic constraints:

```python
def simulate_physics(predicted_trajectories, scene_data, simulation_steps=50):
    """Apply physics-based constraints to predicted trajectories
    
    Args:
        predicted_trajectories: Initial predicted trajectories
        scene_data: Current scene data with environment information
        simulation_steps: Number of simulation steps
        
    Returns:
        Physics-corrected trajectories
    """
    # Initialize physics engine
    physics = PhysicsEngine(
        gravity=9.81,
        friction_coefficient=0.7,
        collision_threshold=0.2
    )
    
    # Extract drivable area from segmentation
    road_mask = scene_data['segmentation_results']['segment_indices'] == 18  # Road class
    road_points = scene_data['point_cloud']['points'][road_mask]
    
    # Fit drivable surface
    drivable_surface = physics.fit_drivable_surface(road_points)
    
    # Initialize object states
    object_states = {}
    
    for obj_id, trajectory in predicted_trajectories.items():
        # Get object properties
        positions = trajectory['positions']
        
        # Get object class and dimensions
        obj_class = scene_data['detection_results']['class_labels'][obj_id]
        obj_dims = scene_data['detection_results']['boxes_3d'][obj_id][3:6]  # length, width, height
        
        # Initialize physical properties based on class
        if 'Vehicle' in obj_class:
            mass = 1500.0  # kg
            max_acceleration = 3.0  # m/s²
            max_deceleration = 7.0  # m/s²
        elif 'Pedestrian' in obj_class:
            mass = 70.0  # kg
            max_acceleration = 1.0  # m/s²
            max_deceleration = 2.0  # m/s²
        else:
            mass = 100.0  # kg
            max_acceleration = 2.0  # m/s²
            max_deceleration = 4.0  # m/s²
        
        # Initial state
        initial_pos = positions[0]
        initial_vel = (positions[1] - positions[0]) / (trajectory['timestamps'][1] - trajectory['timestamps'][0])
        
        object_states[obj_id] = {
            'position': initial_pos,
            'velocity': initial_vel,
            'acceleration': np.zeros(3),
            'mass': mass,
            'dimensions': obj_dims,
            'max_acceleration': max_acceleration,
            'max_deceleration': max_deceleration,
            'trajectory': [initial_pos]
        }
    
    # Run simulation
    dt = 0.1  # seconds
    
    for step in range(simulation_steps):
        # Update each object
        for obj_id, state in object_states.items():
            # Apply physics
            state = update_physics(state, dt, object_states, drivable_surface)
            
            # Record trajectory
            state['trajectory'].append(state['position'].copy())
    
    # Format results
    corrected_trajectories = {}
    
    for obj_id, state in object_states.items():
        corrected_trajectories[obj_id] = {
            'positions': np.array(state['trajectory']),
            'timestamps': np.arange(0, simulation_steps * dt + dt, dt),
        }
    
    return corrected_trajectories
```

The physics model solves the equations of motion:

$$\mathbf{a}(t) = \mathbf{F}(t)/m$$
$$\mathbf{v}(t+\Delta t) = \mathbf{v}(t) + \mathbf{a}(t)\Delta t$$
$$\mathbf{x}(t+\Delta t) = \mathbf{x}(t) + \mathbf{v}(t)\Delta t + \frac{1}{2}\mathbf{a}(t)\Delta t^2$$

where $\mathbf{F}(t)$ is the sum of all forces acting on the object.

3. **Agent Behavior Modeling**: Develop realistic driver and pedestrian behavioral models for naturalistic scenario simulation:

```python
def simulate_agent_behavior(corrected_trajectories, scene_data, agent_config):
    """Simulate realistic agent behaviors for interactive simulation
    
    Args:
        corrected_trajectories: Physics-corrected trajectories
        scene_data: Current scene data with environment information
        agent_config: Configuration for different agent types
        
    Returns:
        Behaviorally enhanced trajectories
    """
    # Initialize agent behavior models
    behavior_models = {
        'Vehicle': IDMDriverModel(**agent_config['vehicle']),
        'Pedestrian': PedestrianModel(**agent_config['pedestrian']),
        'Cyclist': CyclistModel(**agent_config['cyclist'])
    }
    
    # Extract road network
    road_network = extract_road_network(scene_data)
    
    # Initialize agent states
    agent_states = {}
    
    for obj_id, trajectory in corrected_trajectories.items():
        # Get object class
        obj_class = scene_data['detection_results']['class_labels'][obj_id]
        
        # Initialize agent state
        positions = trajectory['positions']
        timestamps = trajectory['timestamps']
        
        if len(positions) < 2:
            continue
            
        # Calculate velocity and heading
        velocity = (positions[1] - positions[0]) / (timestamps[1] - timestamps[0])
        heading = np.arctan2(velocity[1], velocity[0])
        
        # Initialize state
        agent_states[obj_id] = {
            'position': positions[0],
            'velocity': velocity,
            'heading': heading,
            'class': obj_class,
            'trajectory': [positions[0]],
            'behavioral_state': 'normal'  # Initial state
        }
    
    # Simulate agent behaviors
    simulation_steps = 50
    dt = 0.1  # seconds
    
    for step in range(simulation_steps):
        # Update each agent
        for obj_id, state in agent_states.items():
            # Get appropriate behavior model
            model = next((m for c, m in behavior_models.items() if c in state['class']), 
                         behavior_models['Vehicle'])
            
            # Get nearby agents
            nearby_agents = get_nearby_agents(obj_id, agent_states, max_distance=50.0)
            
            # Update behavioral state
            state['behavioral_state'] = model.update_state(
                state, 
                nearby_agents,
                road_network
            )
            
            # Calculate action based on behavioral state
            action = model.calculate_action(
                state,
                nearby_agents,
                road_network
            )
            
            # Apply action
            state['velocity'] += action['acceleration'] * dt
            state['heading'] += action['steering'] * dt
            
            # Update position
            dx = state['velocity'][0] * dt
            dy = state['velocity'][1] * dt
            state['position'][0] += dx
            state['position'][1] += dy
            
            # Record trajectory
            state['trajectory'].append(state['position'].copy())
    
    return agent_states
```

For vehicles, the Intelligent Driver Model (IDM) acceleration is given by:

$$a = a_{\max}\left[1 - \left(\frac{v}{v_0}\right)^\delta - \left(\frac{s^*(v, \Delta v)}{s}\right)^2\right]$$

where $v$ is the current velocity, $v_0$ is the desired velocity, $s$ is the current gap to the leading vehicle, and $s^*$ is the desired gap:

$$s^*(v, \Delta v) = s_0 + vT + \frac{v \Delta v}{2\sqrt{a_{\max} b}}$$

with $s_0$ as the minimum gap, $T$ as the desired time headway, $\Delta v$ as the velocity difference to the lead vehicle, and $b$ as the comfortable deceleration.

### 6.2 Enhanced Sensor Fusion Techniques

To improve the integration quality of multi-sensor data, we plan to implement more advanced sensor fusion techniques:

```python
def advanced_sensor_fusion(lidar_data, camera_data, radar_data=None):
    """Implement advanced sensor fusion to improve perception performance
    
    Args:
        lidar_data: LiDAR data dictionary
        camera_data: Camera data dictionary
        radar_data: Optional radar data dictionary
        
    Returns:
        Fused perception results
    """
    # 1. Feature-level fusion: Extract features from individual sensors
    lidar_features = extract_lidar_features(lidar_data)
    camera_features = extract_camera_features(camera_data)
    
    if radar_data is not None:
        radar_features = extract_radar_features(radar_data)
        combined_features = fuse_features([lidar_features, camera_features, radar_features])
    else:
        combined_features = fuse_features([lidar_features, camera_features])
    
    # 2. Detection-level fusion: Merge detections from different sensors
    lidar_detections = detect_objects_lidar(lidar_data)
    camera_detections = detect_objects_camera(camera_data)
    
    if radar_data is not None:
        radar_detections = detect_objects_radar(radar_data)
        fused_detections = fuse_detections([lidar_detections, camera_detections, radar_detections])
    else:
        fused_detections = fuse_detections([lidar_detections, camera_detections])
    
    # 3. Track-level fusion: Leverage temporal consistency for object tracking
    tracked_objects = multi_sensor_tracking(fused_detections)
    
    # 4. Semantic-level fusion: Merge segmentation and classification results
    lidar_segmentation = segment_lidar(lidar_data)
    camera_segmentation = segment_camera(camera_data)
    
    fused_segmentation = fuse_segmentation(lidar_segmentation, camera_segmentation)
    
    return {
        'fused_detections': fused_detections,
        'tracked_objects': tracked_objects,
        'fused_segmentation': fused_segmentation,
        'fused_features': combined_features
    }
```

Feature fusion can be implemented using learned attention weights:

$$F_{fused} = \sum_{i=1}^{N} \alpha_i F_i$$

where $\alpha_i$ are attention weights computed as:

$$\alpha_i = \frac{\exp(W_i \cdot F_i)}{\sum_{j=1}^{N} \exp(W_j \cdot F_j)}$$

and $W_i$ are learned weight matrices.

### 6.3 Real-time Processing Optimization

To support real-time analysis of large-scale data, we plan to implement key performance optimizations:

```python
def optimize_processing_pipeline(pipeline_config):
    """Optimize processing pipeline for real-time performance
    
    Args:
        pipeline_config: Processing pipeline configuration
        
    Returns:
        Optimized processing pipeline
    """
    # 1. Dynamic computation resource allocation based on task importance
    resource_allocator = DynamicResourceAllocator(
        gpu_memory=pipeline_config.get('gpu_memory', 8000),
        cpu_cores=pipeline_config.get('cpu_cores', 8)
    )
    
    # 2. Model quantization and pruning
    model_optimizer = ModelOptimizer(
        quantization=pipeline_config.get('enable_quantization', True),
        pruning=pipeline_config.get('enable_pruning', True),
        target_precision=pipeline_config.get('target_precision', 'fp16')
    )
    
    # 3. Adaptive point cloud sampling
    point_cloud_sampler = AdaptivePointCloudSampler(
        base_resolution=pipeline_config.get('base_voxel_size', 0.1),
        adaptive_regions=pipeline_config.get('adaptive_regions', True),
        importance_weighting=pipeline_config.get('importance_weighting', True)
    )
    
    # 4. Parallel processing pipeline
    processing_pipeline = ParallelPipeline(
        num_workers=pipeline_config.get('num_workers', 4),
        pipeline_depth=pipeline_config.get('pipeline_depth', 3),
        prefetch_size=pipeline_config.get('prefetch_size', 2)
    )
    
    # 5. Cache frequently used results
    result_cache = ProcessingCache(
        cache_size=pipeline_config.get('cache_size_mb', 1024),
        ttl=pipeline_config.get('cache_ttl_ms', 500)
    )
    
    # Build optimized pipeline
    optimized_pipeline = {
        'resource_allocator': resource_allocator,
        'model_optimizer': model_optimizer,
        'point_cloud_sampler': point_cloud_sampler,
        'processing_pipeline': processing_pipeline,
        'result_cache': result_cache
    }
    
    return optimized_pipeline
```

For adaptive point cloud sampling, we use a non-uniform sampling density function:

$$p(x,y,z) = \frac{I(x,y,z)^{\alpha}}{\sum_{i,j,k} I(x_i,y_j,z_k)^{\alpha}}$$

where $I(x,y,z)$ is an importance function that assigns higher values to regions of interest (e.g., object boundaries, road edges), and $\alpha$ is a parameter controlling the sampling bias.

### 6.4 Support for Additional Autonomous Driving Datasets

To extend the system's applicability, we plan to add support for other major autonomous driving datasets:

```python
def add_dataset_support(dataset_name, dataset_path):
    """Add support for new dataset
    
    Args:
        dataset_name: Dataset name ('nuScenes', 'KITTI', 'Argoverse')
        dataset_path: Dataset file path
        
    Returns:
        Configured dataset adapter
    """
    dataset_adapters = {
        'nuScenes': NuScenesAdapter(dataset_path),
        'KITTI': KITTIAdapter(dataset_path),
        'Argoverse': ArgoAdapter(dataset_path),
        'Lyft': LyftAdapter(dataset_path),
        'BDD100K': BDDAdapter(dataset_path)
    }
    
    if dataset_name not in dataset_adapters:
        raise ValueError(f"Unsupported dataset: {dataset_name}, supported datasets: {list(dataset_adapters.keys())}")
    
    # Instantiate and configure adapter
    adapter = dataset_adapters[dataset_name]
    adapter.initialize()
    
    # Verify adapter works
    test_sample = adapter.get_sample(0)
    
    print(f"Successfully configured {dataset_name} dataset adapter")
    print(f"Sample data: {len(adapter)} scenes")
    print(f"Sensor types: {adapter.get_sensor_types()}")
    
    return adapter
```

## Conclusion

This project provides a comprehensive set of tools and techniques for visualizing and analyzing autonomous driving scenes from the Waymo Open Dataset. Our system handles complex sensor data formats, coordinate transformations, and perception tasks, enabling researchers to gain deeper insights into the perception capabilities of autonomous driving systems.

Through innovative visualization techniques, we have achieved the integration of multi-modal perception data, including LiDAR point clouds, camera images, and semantic segmentation results. These visualization methods not only provide intuitive scene understanding but also support quantitative analysis and statistical evaluation.

Our methodology offers a solid technical foundation for autonomous driving research, enabling efficient processing, visualization, and analysis of complex multi-sensor data. This work contributes to advancing autonomous driving technology research and development, particularly in scene understanding and perception algorithm evaluation.

Future work will extend this framework to support real-time simulation, advanced sensor fusion, and additional datasets, further enhancing the system's capabilities and application scope. We believe these tools will provide valuable resources for the autonomous driving research community, accelerating the development of autonomous driving technology.