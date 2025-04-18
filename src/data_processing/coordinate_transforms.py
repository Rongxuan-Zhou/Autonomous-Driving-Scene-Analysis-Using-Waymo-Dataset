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

def cartesian_to_spherical(points):
    """Convert Cartesian coordinates to spherical coordinates
    
    Args:
        points: Nx3 array of Cartesian coordinates (x, y, z)
        
    Returns:
        Nx3 array of spherical coordinates (range, azimuth, inclination)
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Calculate spherical coordinates
    range_values = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    inclination = np.arcsin(z / range_values)
    
    # Stack coordinates
    spherical_points = np.stack([range_values, azimuth, inclination], axis=-1)
    return spherical_points

def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion to rotation matrix
    
    Args:
        quaternion: 4-element array (w, x, y, z)
        
    Returns:
        3x3 rotation matrix
    """
    return Rotation.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]]).as_matrix()

def euler_to_rotation_matrix(euler_angles):
    """Convert Euler angles to rotation matrix
    
    Args:
        euler_angles: 3-element array (roll, pitch, yaw) in radians
        
    Returns:
        3x3 rotation matrix
    """
    return Rotation.from_euler('xyz', euler_angles).as_matrix()

def heading_to_quaternion(heading):
    """Convert heading angle to quaternion representing rotation around z-axis
    
    Args:
        heading: Heading angle in radians
        
    Returns:
        4-element array (w, x, y, z)
    """
    return np.array([np.cos(heading/2), 0, 0, np.sin(heading/2)])

def create_transformation_matrix(translation, rotation):
    """Create 4x4 transformation matrix from translation and rotation
    
    Args:
        translation: 3-element array (x, y, z)
        rotation: 3x3 rotation matrix or 4-element quaternion (w, x, y, z)
        
    Returns:
        4x4 transformation matrix
    """
    # Create transformation matrix
    transform = np.eye(4)
    
    # Set translation
    transform[:3, 3] = translation
    
    # Set rotation
    if rotation.shape == (4,):
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(rotation)
    else:
        rotation_matrix = rotation
    
    transform[:3, :3] = rotation_matrix
    
    return transform
