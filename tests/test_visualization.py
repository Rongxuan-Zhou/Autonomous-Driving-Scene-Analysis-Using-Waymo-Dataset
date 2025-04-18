import unittest
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.visualizer import WaymoVisualizer

class TestWaymoVisualizer(unittest.TestCase):
    """Test visualization functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create a visualizer instance
        self.visualizer = WaymoVisualizer(theme='dark')
        
        # Create a sample topic counter
        self.topic_counter = {
            '/lidar/points': 100,
            '/camera/front': 80,
            '/tf': 200,
            '/objects': 50,
            '/camera/front_left': 80,
            '/camera/front_right': 80,
            '/camera/side_left': 80,
            '/camera/side_right': 80
        }
        
        # Create a sample point cloud
        self.point_cloud = np.random.rand(100, 3)
        
        # Create sample colors
        self.colors = np.random.rand(100, 3)
        
        # Create sample segment indices
        self.segment_indices = np.random.randint(0, 23, 100)
        
        # Create sample class names
        self.class_names = {
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
            22: 'Ground'
        }
        
        # Create sample objects
        self.objects = {
            'boxes_3d': np.array([
                [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],  # x, y, z, l, w, h, heading
                [10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0]
            ]),
            'class_labels': ['Vehicle', 'Pedestrian'],
            'scores': [0.9, 0.8]
        }
        
        # Create sample dashboard data
        self.dashboard_data = {
            'topic_counter': self.topic_counter,
            'message_sizes': np.random.randint(1000, 1000000, 100),
            'timestamps': pd.date_range(start='2023-01-01', periods=100, freq='100ms'),
            'message_counts': np.random.randint(1, 10, 100),
            'class_counts': {'Vehicle': 10, 'Pedestrian': 5, 'Cyclist': 2}
        }
    
    def test_plot_topic_distribution(self):
        """Test topic distribution plotting"""
        # Test matplotlib version
        fig = self.visualizer.plot_topic_distribution(
            self.topic_counter,
            title="Test Topic Distribution",
            interactive=False
        )
        
        # Check that figure was created
        self.assertIsNotNone(fig)
        
        # Close figure to avoid memory leaks
        plt.close(fig)
        
        # Test interactive version
        if hasattr(self.visualizer, 'plot_topic_distribution'):
            fig = self.visualizer.plot_topic_distribution(
                self.topic_counter,
                title="Test Topic Distribution",
                interactive=True
            )
            
            # Check that figure was created
            self.assertIsNotNone(fig)
    
    def test_create_interactive_dashboard(self):
        """Test interactive dashboard creation"""
        # Create dashboard
        dashboard = self.visualizer.create_interactive_dashboard(self.dashboard_data)
        
        # Check that dashboard was created
        self.assertIsNotNone(dashboard)
        
        # Check that dashboard has expected traces
        self.assertTrue(len(dashboard.data) > 0)
    
    def test_segmentation_palette(self):
        """Test segmentation color palette"""
        # Check that palette was created
        self.assertTrue(hasattr(self.visualizer, 'segmentation_palette'))
        
        # Check that palette has correct shape
        self.assertEqual(self.visualizer.segmentation_palette.shape[0], 23)
        self.assertEqual(self.visualizer.segmentation_palette.shape[1], 3)
        
        # Check that palette values are in range [0, 1]
        self.assertTrue(np.all(self.visualizer.segmentation_palette >= 0))
        self.assertTrue(np.all(self.visualizer.segmentation_palette <= 1))
    
    def test_theme_colors(self):
        """Test theme color configuration"""
        # Check dark theme
        dark_visualizer = WaymoVisualizer(theme='dark')
        self.assertEqual(dark_visualizer.theme, 'dark')
        self.assertTrue('primary' in dark_visualizer.colors)
        self.assertTrue('secondary' in dark_visualizer.colors)
        self.assertTrue('accent' in dark_visualizer.colors)
        
        # Check light theme
        light_visualizer = WaymoVisualizer(theme='light')
        self.assertEqual(light_visualizer.theme, 'light')
        self.assertTrue('primary' in light_visualizer.colors)
        self.assertTrue('secondary' in light_visualizer.colors)
        self.assertTrue('accent' in light_visualizer.colors)
        
        # Check that themes have different colors
        self.assertNotEqual(dark_visualizer.colors['background'], light_visualizer.colors['background'])

class TestMockScene(unittest.TestCase):
    """Test with mock scene data"""
    
    def setUp(self):
        """Set up mock scene data"""
        # Create a visualizer instance
        self.visualizer = WaymoVisualizer(theme='dark')
        
        # Create mock point cloud
        self.point_cloud = {
            'points': np.random.rand(100, 3),
            'intensities': np.random.rand(100),
            'ground_mask': np.random.choice([True, False], 100)
        }
        
        # Create mock camera data
        self.camera_data = {
            'FRONT': {
                'visualization': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'camera_name': 'FRONT'
            },
            'FRONT_LEFT': {
                'visualization': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'camera_name': 'FRONT_LEFT'
            },
            'FRONT_RIGHT': {
                'visualization': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                'camera_name': 'FRONT_RIGHT'
            }
        }
        
        # Create mock detection results
        self.detection_results = {
            'boxes_3d': np.array([
                [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],  # x, y, z, l, w, h, heading
                [10.0, 5.0, 0.0, 4.0, 2.0, 1.5, 0.0]
            ]),
            'corners': np.array([
                np.random.rand(8, 3),
                np.random.rand(8, 3)
            ]),
            'class_labels': ['Vehicle', 'Pedestrian'],
            'scores': [0.9, 0.8],
            'class_counts': {'Vehicle': 1, 'Pedestrian': 1}
        }
        
        # Create mock segmentation results
        self.segmentation_results = {
            'segment_indices': np.random.randint(0, 23, 100),
            'colors': np.random.randint(0, 255, (100, 3), dtype=np.uint8),
            'class_counts': {'Road': 50, 'Vehicle': 20, 'Vegetation': 30},
            'class_distributions': {'Road': 0.5, 'Vehicle': 0.2, 'Vegetation': 0.3}
        }
        
        # Create mock scene data
        self.scene_data = {
            'point_cloud': self.point_cloud,
            'camera_data': self.camera_data,
            'detection_results': self.detection_results,
            'segmentation_results': self.segmentation_results
        }
    
    def test_create_sensor_fusion_visualization(self):
        """Test sensor fusion visualization"""
        # Skip if method not available (for compatibility with different versions)
        if not hasattr(self.visualizer, 'create_sensor_fusion_visualization'):
            self.skipTest("create_sensor_fusion_visualization not available")
        
        # Create visualization
        fusion_vis = self.visualizer.create_sensor_fusion_visualization(self.scene_data)
        
        # Check that visualization was created
        self.assertIsNotNone(fusion_vis)
        
        # Check that visualization has expected traces
        self.assertTrue(len(fusion_vis.data) > 0)
    
    def test_create_analysis_dashboard(self):
        """Test analysis dashboard creation"""
        # Skip if method not available (for compatibility with different versions)
        if not hasattr(self.visualizer, 'create_analysis_dashboard'):
            self.skipTest("create_analysis_dashboard not available")
        
        # Create dashboard
        dashboard = self.visualizer.create_analysis_dashboard(self.scene_data)
        
        # Check that dashboard was created
        self.assertIsNotNone(dashboard)
        
        # Check that dashboard has expected traces
        self.assertTrue(len(dashboard.data) > 0)

if __name__ == '__main__':
    unittest.main()
