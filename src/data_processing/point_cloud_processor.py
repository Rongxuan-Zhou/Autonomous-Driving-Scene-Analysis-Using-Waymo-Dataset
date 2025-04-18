import numpy as np
import open3d as o3d
from sklearn.linear_model import RANSACRegressor
from .coordinate_transforms import spherical_to_cartesian, transform_point_cloud

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

def process_camera_image(image_data, camera_name, calibration):
    """Process camera image data with intrinsic correction and annotation
    
    Args:
        image_data: Raw image bytes from Waymo dataset
        camera_name: Camera identifier (e.g., 'FRONT', 'FRONT_LEFT')
        calibration: Camera calibration parameters
        
    Returns:
        Dictionary with processed image and metadata
    """
    import cv2
    from PIL import Image
    import io
    
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
    import cv2
    
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

def project_lidar_to_camera(point_cloud, camera_data, calibration):
    """Project LiDAR points onto camera image
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        camera_data: Dictionary containing processed camera data
        calibration: Calibration data for transformations
        
    Returns:
        Visualization with projected points and image with depth information
    """
    import cv2
    import numpy as np
    from .coordinate_transforms import transform_point_cloud
    
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

def extract_frame(mcap_file, frame_index=0):
    """Extract a single frame from MCAP file
    
    Args:
        mcap_file: Path to MCAP file
        frame_index: Index of frame to extract
        
    Returns:
        Dictionary with frame data
    """
    from mcap.reader import make_reader
    import json
    
    # Initialize frame data
    frame_data = {
        'timestamp': None,
        'range_image': None,
        'images': {},
        'calibration': None,
        'topic_counter': {},
        'message_sizes': [],
        'timestamps': [],
        'message_counts': []
    }
    
    # Open MCAP file
    with open(mcap_file, 'rb') as f:
        reader = make_reader(f)
        
        # Collect frame timestamps
        frame_timestamps = []
        
        for schema, channel, message in reader.iter_messages():
            if channel.topic == '/tf':
                frame_timestamps.append(message.log_time)
        
        if frame_index >= len(frame_timestamps):
            raise ValueError(f"Frame index {frame_index} out of range (0-{len(frame_timestamps)-1})")
        
        # Get target timestamp
        target_timestamp = frame_timestamps[frame_index]
        frame_data['timestamp'] = target_timestamp
        
        # Reset reader
        f.seek(0)
        reader = make_reader(f)
        
        # Collect messages for this frame
        topics = []
        message_sizes = []
        
        for schema, channel, message in reader.iter_messages():
            # Only process messages within a small time window of the target timestamp
            time_diff = abs(message.log_time - target_timestamp)
            if time_diff > 1_000_000_000:  # 1 second in nanoseconds
                continue
            
            topics.append(channel.topic)
            message_sizes.append(len(message.data))
            
            # Process specific message types
            if channel.topic == '/lidar/points' and time_diff < 50_000_000:  # 50ms
                try:
                    point_cloud_msg = json.loads(message.data)
                    # In a real implementation, this would parse the point cloud data
                    # For now, we'll create a dummy range image
                    frame_data['range_image'] = np.random.rand(64, 2650, 4)  # H, W, C
                except Exception as e:
                    print(f"Error parsing point cloud: {e}")
            
            elif channel.topic.startswith('/camera/') and time_diff < 50_000_000:  # 50ms
                try:
                    camera_name = channel.topic.split('/')[-1].upper()
                    image_msg = json.loads(message.data)
                    # In a real implementation, this would decode the image data
                    # For now, we'll create a dummy image
                    frame_data['images'][camera_name] = np.random.randint(0, 255, (900, 1600, 3), dtype=np.uint8)
                except Exception as e:
                    print(f"Error parsing image: {e}")
        
        # Count topics
        from collections import Counter
        topic_counter = Counter(topics)
        frame_data['topic_counter'] = dict(topic_counter)
        frame_data['message_sizes'] = message_sizes
        
        # Create dummy calibration data
        class DummyCalibration:
            def __init__(self):
                self.vehicle_to_global = np.eye(4)
                self.lidar_extrinsics = [np.eye(4) for _ in range(5)]
                self.camera_extrinsics = [np.eye(4) for _ in range(5)]
                self.beam_inclinations = np.linspace(-np.pi/6, np.pi/6, 64)
                self.camera_intrinsics = [
                    [800, 800, 800, 450],  # fx, fy, cx, cy
                    [800, 800, 800, 450],
                    [800, 800, 800, 450],
                    [800, 800, 800, 450],
                    [800, 800, 800, 450]
                ]
                self.camera_distortion = [
                    np.zeros(5),
                    np.zeros(5),
                    np.zeros(5),
                    np.zeros(5),
                    np.zeros(5)
                ]
        
        frame_data['calibration'] = DummyCalibration()
    
    return frame_data
