import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.coordinate_transforms import (
    spherical_to_cartesian, 
    cartesian_to_spherical,
    quaternion_to_rotation_matrix,
    euler_to_rotation_matrix,
    heading_to_quaternion,
    create_transformation_matrix
)

class TestCoordinateTransforms(unittest.TestCase):
    """Test coordinate transformation functions"""
    
    def test_spherical_to_cartesian(self):
        """Test conversion from spherical to Cartesian coordinates"""
        # Create a simple range image (1x1) with a single point at distance 10
        range_image = np.array([[[10.0]]])
        
        # Define inclination and azimuth angles (0, 0) -> point along x-axis
        inclinations = np.array([0.0])
        azimuths = np.array([0.0])
        
        # Convert to Cartesian
        points = spherical_to_cartesian(range_image, inclinations, azimuths)
        
        # Expected result: [10, 0, 0] (point at distance 10 along x-axis)
        expected = np.array([[[10.0, 0.0, 0.0]]])
        
        # Check result
        np.testing.assert_allclose(points, expected, rtol=1e-5)
    
    def test_cartesian_to_spherical(self):
        """Test conversion from Cartesian to spherical coordinates"""
        # Create a point along x-axis at distance 10
        points = np.array([[10.0, 0.0, 0.0]])
        
        # Convert to spherical
        spherical = cartesian_to_spherical(points)
        
        # Expected result: [10, 0, 0] (range=10, azimuth=0, inclination=0)
        expected = np.array([[10.0, 0.0, 0.0]])
        
        # Check result
        np.testing.assert_allclose(spherical, expected, rtol=1e-5)
    
    def test_quaternion_to_rotation_matrix(self):
        """Test conversion from quaternion to rotation matrix"""
        # Identity quaternion (no rotation)
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        
        # Convert to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)
        
        # Expected result: identity matrix
        expected = np.eye(3)
        
        # Check result
        np.testing.assert_allclose(rotation_matrix, expected, rtol=1e-5)
    
    def test_euler_to_rotation_matrix(self):
        """Test conversion from Euler angles to rotation matrix"""
        # Zero Euler angles (no rotation)
        euler_angles = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        
        # Convert to rotation matrix
        rotation_matrix = euler_to_rotation_matrix(euler_angles)
        
        # Expected result: identity matrix
        expected = np.eye(3)
        
        # Check result
        np.testing.assert_allclose(rotation_matrix, expected, rtol=1e-5)
    
    def test_heading_to_quaternion(self):
        """Test conversion from heading angle to quaternion"""
        # Zero heading (no rotation)
        heading = 0.0
        
        # Convert to quaternion
        quaternion = heading_to_quaternion(heading)
        
        # Expected result: identity quaternion (w=1, x=y=z=0)
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Check result
        np.testing.assert_allclose(quaternion, expected, rtol=1e-5)
    
    def test_create_transformation_matrix(self):
        """Test creation of transformation matrix"""
        # No translation, no rotation
        translation = np.array([0.0, 0.0, 0.0])
        rotation = np.eye(3)
        
        # Create transformation matrix
        transform = create_transformation_matrix(translation, rotation)
        
        # Expected result: identity matrix
        expected = np.eye(4)
        
        # Check result
        np.testing.assert_allclose(transform, expected, rtol=1e-5)
        
        # Test with translation
        translation = np.array([1.0, 2.0, 3.0])
        rotation = np.eye(3)
        
        # Create transformation matrix
        transform = create_transformation_matrix(translation, rotation)
        
        # Expected result: identity rotation with translation
        expected = np.eye(4)
        expected[:3, 3] = translation
        
        # Check result
        np.testing.assert_allclose(transform, expected, rtol=1e-5)

class MockCalibration:
    """Mock calibration class for testing"""
    
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

if __name__ == '__main__':
    unittest.main()
