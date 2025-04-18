import unittest
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.classification.scene_classifier import classify_scene, analyze_scene_composition, detect_scene_anomalies
from src.classification.object_detector import get_3d_box_corners, dummy_object_detection, dummy_segmentation

class TestSceneClassifier(unittest.TestCase):
    """Test scene classification functions"""
    
    def setUp(self):
        """Set up test data"""
        # Create a sample DataFrame with message features
        self.features_df = pd.DataFrame({
            'topic': [
                '/lidar/points',
                '/camera/front',
                '/tf',
                '/objects',
                '/camera/front_left',
                '/lidar/points',
                '/tf',
                '/objects'
            ],
            'message_size': [
                1000000,  # Large message (LiDAR)
                500000,   # Medium message (Camera)
                1000,     # Small message (TF)
                5000,     # Small message (Objects)
                500000,   # Medium message (Camera)
                1000000,  # Large message (LiDAR)
                1000,     # Small message (TF)
                5000      # Small message (Objects)
            ],
            'timestamp': [
                1000000000,  # 1 second
                1000000100,  # 1.0001 second
                1000000200,  # 1.0002 second
                1000000300,  # 1.0003 second
                2000000000,  # 2 seconds
                2000000100,  # 2.0001 second
                2000000200,  # 2.0002 second
                2000000300   # 2.0003 second
            ]
        })
    
    def test_classify_scene(self):
        """Test scene classification"""
        # Classify scene
        classified_df, scene_stats = classify_scene(self.features_df)
        
        # Check that all rows have a scene_category
        self.assertTrue('scene_category' in classified_df.columns)
        self.assertEqual(len(classified_df), len(self.features_df))
        
        # Check that primary category is identified
        self.assertTrue('primary_category' in scene_stats)
        
        # Check that category counts are calculated
        self.assertTrue('category_counts' in scene_stats)
        self.assertTrue('category_percentages' in scene_stats)
        
        # Check specific classifications
        lidar_rows = classified_df[classified_df['topic'] == '/lidar/points']
        self.assertTrue(all(lidar_rows['scene_category'] == 'lidar_perception'))
        
        camera_rows = classified_df[classified_df['topic'].str.contains('/camera/')]
        self.assertTrue(all(camera_rows['scene_category'] == 'visual_perception'))
    
    def test_analyze_scene_composition(self):
        """Test scene composition analysis"""
        # Analyze scene composition
        composition = analyze_scene_composition(self.features_df)
        
        # Check that analysis contains expected keys
        self.assertTrue('topic_frequency' in composition)
        self.assertTrue('total_messages' in composition)
        self.assertTrue('unique_topics' in composition)
        
        # Check specific values
        self.assertEqual(composition['total_messages'], len(self.features_df))
        self.assertEqual(composition['unique_topics'], len(self.features_df['topic'].unique()))
        
        # Check topic frequency
        self.assertEqual(composition['topic_frequency']['/lidar/points'], 2)
        self.assertEqual(composition['topic_frequency']['/camera/front'], 1)
    
    def test_detect_scene_anomalies(self):
        """Test scene anomaly detection"""
        # First classify the scene
        classified_df, _ = classify_scene(self.features_df)
        
        # Detect anomalies
        anomalies = detect_scene_anomalies(classified_df)
        
        # Check that anomalies structure is correct
        self.assertTrue('anomalies' in anomalies)
        self.assertTrue('anomaly_count' in anomalies)
        self.assertTrue('has_critical_anomalies' in anomalies)
        
        # Create a modified DataFrame with missing topics
        modified_df = self.features_df.copy()
        modified_df = modified_df[modified_df['topic'] != '/lidar/points']
        
        # Classify and detect anomalies
        modified_classified_df, _ = classify_scene(modified_df)
        modified_anomalies = detect_scene_anomalies(modified_classified_df)
        
        # Should detect missing topics
        self.assertTrue(modified_anomalies['anomaly_count'] > 0)
        
        # At least one anomaly should be about missing topics
        missing_topics_anomaly = False
        for anomaly in modified_anomalies['anomalies']:
            if anomaly['type'] == 'missing_topics':
                missing_topics_anomaly = True
                break
        
        self.assertTrue(missing_topics_anomaly)

class TestObjectDetector(unittest.TestCase):
    """Test object detection functions"""
    
    def test_get_3d_box_corners(self):
        """Test 3D box corner calculation"""
        # Define a box at the origin with unit dimensions
        dimensions = [1.0, 1.0, 1.0]  # length, width, height
        position = [0.0, 0.0, 0.0]    # x, y, z
        heading = 0.0                 # No rotation
        
        # Calculate corners
        corners = get_3d_box_corners(dimensions, position, heading)
        
        # Expected corners for a 1x1x1 box at origin
        expected_corners = np.array([
            [-0.5, -0.5, -0.5],  # bottom left back
            [0.5, -0.5, -0.5],   # bottom right back
            [0.5, 0.5, -0.5],    # bottom right front
            [-0.5, 0.5, -0.5],   # bottom left front
            [-0.5, -0.5, 0.5],   # top left back
            [0.5, -0.5, 0.5],    # top right back
            [0.5, 0.5, 0.5],     # top right front
            [-0.5, 0.5, 0.5]     # top left front
        ])
        
        # Check result
        np.testing.assert_allclose(corners, expected_corners, rtol=1e-5)
        
        # Test with translation
        position = [1.0, 2.0, 3.0]
        corners = get_3d_box_corners(dimensions, position, heading)
        
        # Expected corners for a 1x1x1 box at (1,2,3)
        expected_corners = expected_corners + np.array([1.0, 2.0, 3.0])
        
        # Check result
        np.testing.assert_allclose(corners, expected_corners, rtol=1e-5)
    
    def test_dummy_object_detection(self):
        """Test dummy object detection"""
        # Create a simple point cloud dictionary
        point_cloud = {
            'points': np.random.rand(100, 3),
            'intensities': np.random.rand(100)
        }
        
        # Run dummy detection
        detection_results = dummy_object_detection(point_cloud)
        
        # Check that results contain expected keys
        self.assertTrue('boxes_3d' in detection_results)
        self.assertTrue('corners' in detection_results)
        self.assertTrue('convex_hulls' in detection_results)
        self.assertTrue('scores' in detection_results)
        self.assertTrue('classes' in detection_results)
        self.assertTrue('class_labels' in detection_results)
        self.assertTrue('class_counts' in detection_results)
        
        # Check that boxes have correct shape (N, 7) - x, y, z, l, w, h, heading
        self.assertEqual(detection_results['boxes_3d'].shape[1], 7)
        
        # Check that class labels match classes
        self.assertEqual(len(detection_results['class_labels']), len(detection_results['classes']))
    
    def test_dummy_segmentation(self):
        """Test dummy segmentation"""
        # Create a simple point cloud dictionary
        point_cloud = {
            'points': np.random.rand(100, 3),
            'intensities': np.random.rand(100)
        }
        
        # Run dummy segmentation
        segmentation_results = dummy_segmentation(point_cloud)
        
        # Check that results contain expected keys
        self.assertTrue('segment_indices' in segmentation_results)
        self.assertTrue('colors' in segmentation_results)
        self.assertTrue('class_counts' in segmentation_results)
        self.assertTrue('class_distributions' in segmentation_results)
        self.assertTrue('class_names' in segmentation_results)
        
        # Check that segment indices and colors have correct shape
        self.assertEqual(len(segmentation_results['segment_indices']), len(point_cloud['points']))
        self.assertEqual(segmentation_results['colors'].shape, (len(point_cloud['points']), 3))
        
        # Check that class distributions sum to approximately 1
        total_percentage = sum(segmentation_results['class_distributions'].values())
        self.assertAlmostEqual(total_percentage, 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
