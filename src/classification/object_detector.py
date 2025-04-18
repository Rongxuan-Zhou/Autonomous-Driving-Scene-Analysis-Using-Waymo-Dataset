import numpy as np
import tensorflow as tf
from scipy.spatial import ConvexHull
from ..data_processing.point_cloud_processor import voxelize_point_cloud

def detect_objects_3d(point_cloud, model_path=None):
    """Detect 3D objects in LiDAR point cloud
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        model_path: Path to pre-trained 3D object detection model
        
    Returns:
        Dictionary with detected objects and their properties
    """
    # If model_path is None, use a dummy detection approach
    if model_path is None:
        return dummy_object_detection(point_cloud)
    
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
    
    try:
        # Load model (assuming TensorFlow SavedModel format)
        model = tf.saved_model.load(model_path)
        detect_fn = model.signatures['serving_default']
        
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
    except Exception as e:
        print(f"Error running object detection model: {e}")
        return dummy_object_detection(point_cloud)
    
    # Convert boxes to corners for visualization
    corners = []
    convex_hulls = []
    
    for box in boxes_3d:
        # Extract box parameters
        x, y, z, length, width, height, heading = box
        
        # Generate corners in object frame
        corner_points = get_3d_box_corners([length, width, height], [x, y, z], heading)
        
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
    class_labels = [class_names.get(int(cls), 'Unknown') for cls in classes]
    
    # Count instances of each class
    class_counts = {}
    for label in class_labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    return {
        'boxes_3d': boxes_3d,
        'corners': np.array(corners),
        'convex_hulls': convex_hulls,
        'scores': scores,
        'classes': classes,
        'class_labels': class_labels,
        'class_counts': class_counts
    }

def dummy_object_detection(point_cloud):
    """Generate dummy object detections for demonstration purposes
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        
    Returns:
        Dictionary with dummy detected objects
    """
    # Create some dummy detections
    num_objects = 10
    
    # Random positions within reasonable range
    positions = np.random.uniform(-20, 20, (num_objects, 3))
    positions[:, 2] = np.random.uniform(0, 2, num_objects)  # z-coordinate (height)
    
    # Random dimensions
    dimensions = np.zeros((num_objects, 3))
    
    # Assign different classes
    classes = np.random.choice([1, 2, 4], num_objects)  # Vehicle, Pedestrian, Cyclist
    
    for i, cls in enumerate(classes):
        if cls == 1:  # Vehicle
            dimensions[i] = [np.random.uniform(3.5, 5.0), np.random.uniform(1.5, 2.0), np.random.uniform(1.5, 2.0)]
        elif cls == 2:  # Pedestrian
            dimensions[i] = [np.random.uniform(0.5, 0.8), np.random.uniform(0.5, 0.8), np.random.uniform(1.5, 1.8)]
        else:  # Cyclist
            dimensions[i] = [np.random.uniform(1.5, 2.0), np.random.uniform(0.5, 0.8), np.random.uniform(1.5, 1.8)]
    
    # Random headings
    headings = np.random.uniform(-np.pi, np.pi, num_objects)
    
    # Create 3D boxes
    boxes_3d = np.column_stack([
        positions,
        dimensions,
        headings
    ])
    
    # Random scores
    scores = np.random.uniform(0.5, 0.95, num_objects)
    
    # Convert boxes to corners for visualization
    corners = []
    convex_hulls = []
    
    for i, box in enumerate(boxes_3d):
        # Extract box parameters
        x, y, z, length, width, height, heading = box
        
        # Generate corners in object frame
        corner_points = get_3d_box_corners([length, width, height], [x, y, z], heading)
        
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
    class_labels = [class_names.get(int(cls), 'Unknown') for cls in classes]
    
    # Count instances of each class
    class_counts = {}
    for label in class_labels:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1
    
    return {
        'boxes_3d': boxes_3d,
        'corners': np.array(corners),
        'convex_hulls': convex_hulls,
        'scores': scores,
        'classes': classes,
        'class_labels': class_labels,
        'class_counts': class_counts
    }

def get_3d_box_corners(dimensions, position, heading):
    """Generate corners of 3D bounding box
    
    Args:
        dimensions: [length, width, height] of the box
        position: [x, y, z] center position of the box
        heading: rotation around z-axis
        
    Returns:
        8x3 array of corner coordinates
    """
    # Extract dimensions and position
    l, w, h = dimensions
    x, y, z = position
    
    # Create corners in object frame (centered at origin, aligned with axes)
    corners = np.array([
        [-l/2, -w/2, -h/2],  # bottom left back
        [l/2, -w/2, -h/2],   # bottom right back
        [l/2, w/2, -h/2],    # bottom right front
        [-l/2, w/2, -h/2],   # bottom left front
        [-l/2, -w/2, h/2],   # top left back
        [l/2, -w/2, h/2],    # top right back
        [l/2, w/2, h/2],     # top right front
        [-l/2, w/2, h/2]     # top left front
    ])
    
    # Create rotation matrix around z-axis
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    rotation_matrix = np.array([
        [cos_h, -sin_h, 0],
        [sin_h, cos_h, 0],
        [0, 0, 1]
    ])
    
    # Rotate corners
    corners = np.dot(corners, rotation_matrix.T)
    
    # Translate corners to box position
    corners += np.array([x, y, z])
    
    return corners

def segment_point_cloud(point_cloud, model_path=None):
    """Perform semantic segmentation on LiDAR point cloud
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        model_path: Path to pre-trained semantic segmentation model
        
    Returns:
        Dictionary with segmentation results
    """
    # If model_path is None, use a dummy segmentation approach
    if model_path is None:
        return dummy_segmentation(point_cloud)
    
    # Extract points
    points = point_cloud['points']
    
    try:
        # Load model
        model = tf.saved_model.load(model_path)
        segment_fn = model.signatures['serving_default']
        
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
    except Exception as e:
        print(f"Error running segmentation model: {e}")
        return dummy_segmentation(point_cloud)
    
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

def dummy_segmentation(point_cloud):
    """Generate dummy segmentation for demonstration purposes
    
    Args:
        point_cloud: Dictionary containing processed point cloud data
        
    Returns:
        Dictionary with dummy segmentation results
    """
    # Extract points
    points = point_cloud['points']
    num_points = len(points)
    
    # Create dummy segmentation
    # Assign classes based on simple rules
    segment_indices = np.zeros(num_points, dtype=np.int32)
    
    # Points close to ground (low z) are road
    road_mask = points[:, 2] < 0.2
    segment_indices[road_mask] = 18  # Road
    
    # Points slightly above ground are sidewalk
    sidewalk_mask = (points[:, 2] >= 0.2) & (points[:, 2] < 0.4)
    segment_indices[sidewalk_mask] = 21  # Sidewalk
    
    # Points far from origin in xy plane might be buildings
    dist_xy = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    building_mask = (dist_xy > 10) & (points[:, 2] > 0.5)
    segment_indices[building_mask] = 14  # Building
    
    # Some random vegetation
    veg_mask = (points[:, 2] > 0.5) & (points[:, 2] < 3.0) & (~building_mask)
    veg_mask = veg_mask & (np.random.rand(num_points) < 0.3)
    segment_indices[veg_mask] = 15  # Vegetation
    
    # Some random poles
    pole_mask = (dist_xy < 8) & (points[:, 2] > 0.5) & (points[:, 2] < 3.0)
    pole_mask = pole_mask & (np.random.rand(num_points) < 0.05)
    segment_indices[pole_mask] = 10  # Pole
    
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
    colors = np.zeros((num_points, 3), dtype=np.uint8)
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
