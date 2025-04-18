import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.mcap_processor import extract_features
from src.data_processing.point_cloud_processor import extract_frame, process_point_cloud, process_camera_image, project_lidar_to_camera
from src.classification.scene_classifier import classify_scene, analyze_scene_composition
from src.classification.object_detector import detect_objects_3d, segment_point_cloud
from src.visualization.visualizer import WaymoVisualizer

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Waymo Dataset Visualization Demo')
    parser.add_argument('--mcap_file', type=str, default='data/foxglove_samples/waymo-scene.mcap',
                        help='Path to MCAP file')
    parser.add_argument('--frame_index', type=int, default=0,
                        help='Frame index to visualize')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save visualizations')
    parser.add_argument('--theme', type=str, default='dark', choices=['dark', 'light'],
                        help='Visualization theme')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive visualizations')
    parser.add_argument('--detection_model', type=str, default=None,
                        help='Path to 3D object detection model')
    parser.add_argument('--segmentation_model', type=str, default=None,
                        help='Path to semantic segmentation model')
    
    return parser.parse_args()

def visualize_scene(mcap_file, frame_index=0, output_dir='output', theme='dark', 
                   interactive=False, detection_model=None, segmentation_model=None):
    """Create comprehensive visualization of a single frame
    
    Args:
        mcap_file: Path to MCAP file
        frame_index: Index of frame to visualize
        output_dir: Directory to save visualizations
        theme: Visualization theme ('dark' or 'light')
        interactive: If True, use interactive visualizations
        detection_model: Path to 3D object detection model
        segmentation_model: Path to semantic segmentation model
        
    Returns:
        Dictionary with visualization components
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = WaymoVisualizer(theme=theme)
    
    # Extract frame data
    print(f"Extracting frame {frame_index} from {mcap_file}")
    frame_data = extract_frame(mcap_file, frame_index)
    
    # Process point cloud
    print("Processing point cloud")
    point_cloud = process_point_cloud(
        frame_data['range_image'],
        frame_data['calibration']
    )
    
    # Process camera images
    print("Processing camera images")
    camera_data = {}
    for camera_name, image_data in frame_data['images'].items():
        camera_data[camera_name] = process_camera_image(
            image_data,
            camera_name,
            frame_data['calibration']
        )
    
    # Perform 3D object detection
    print("Performing 3D object detection")
    detection_results = detect_objects_3d(
        point_cloud,
        model_path=detection_model
    )
    
    # Perform semantic segmentation
    print("Performing semantic segmentation")
    segmentation_results = segment_point_cloud(
        point_cloud,
        model_path=segmentation_model
    )
    
    # Project LiDAR points onto front camera
    print("Projecting LiDAR points onto camera images")
    camera_projections = {}
    for camera_name, data in camera_data.items():
        camera_projections[camera_name] = project_lidar_to_camera(
            point_cloud,
            data,
            frame_data['calibration']
        )
    
    # Extract features for scene classification
    print("Extracting features for scene classification")
    features_df = extract_features(mcap_file)
    
    # Classify scene
    print("Classifying scene")
    classified_df, scene_stats = classify_scene(features_df)
    
    # Analyze scene composition
    print("Analyzing scene composition")
    scene_composition = analyze_scene_composition(classified_df)
    
    # Create visualizations
    print("Creating visualizations")
    
    # 1. Topic distribution visualization
    topic_vis = visualizer.plot_topic_distribution(
        frame_data['topic_counter'],
        title="Message Topic Distribution",
        save_path=os.path.join(output_dir, "topic_distribution.png"),
        interactive=interactive
    )
    
    # 2. Create sensor fusion visualization
    scene_data = {
        'point_cloud': point_cloud,
        'camera_data': camera_data,
        'detection_results': detection_results,
        'segmentation_results': segmentation_results
    }
    
    if interactive:
        fusion_vis = visualizer.create_sensor_fusion_visualization(scene_data)
        fusion_vis.write_html(os.path.join(output_dir, "sensor_fusion.html"))
    
    # 3. Create analysis dashboard
    if interactive:
        dashboard_data = {
            'topic_counter': frame_data['topic_counter'],
            'message_sizes': frame_data['message_sizes'],
            'timestamps': frame_data.get('timestamps', []),
            'message_counts': frame_data.get('message_counts', []),
            'class_counts': detection_results['class_counts']
        }
        
        dashboard = visualizer.create_interactive_dashboard(dashboard_data)
        dashboard.write_html(os.path.join(output_dir, "analysis_dashboard.html"))
        
        # Create statistical analysis dashboard
        analysis_dashboard = visualizer.create_analysis_dashboard(scene_data)
        analysis_dashboard.write_html(os.path.join(output_dir, "statistical_analysis.html"))
    
    # 4. Save camera projections
    for camera_name, projection in camera_projections.items():
        plt.figure(figsize=(12, 8))
        plt.imshow(projection['visualization'])
        plt.title(f"LiDAR Projection on {camera_name} Camera")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"projection_{camera_name}.png"))
        plt.close()
    
    # 5. Save segmentation visualization
    plt.figure(figsize=(12, 8))
    # Create a 2D visualization of segmentation
    segment_vis = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Project points to top-down view
    points = point_cloud['points']
    segment_indices = segmentation_results['segment_indices']
    colors = segmentation_results['colors']
    
    # Scale points to image coordinates
    scale = 10.0  # meters per 100 pixels
    offset_x, offset_y = 256, 256  # center of image
    
    for i, (x, y, z) in enumerate(points):
        # Convert to image coordinates (top-down view)
        ix = int(offset_x + x * scale)
        iy = int(offset_y - y * scale)  # Flip y-axis
        
        # Check if within image bounds
        if 0 <= ix < 512 and 0 <= iy < 512:
            segment_vis[iy, ix] = colors[i]
    
    plt.imshow(segment_vis)
    plt.title("Semantic Segmentation (Top-Down View)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "segmentation.png"))
    plt.close()
    
    # 6. Save object detection visualization
    plt.figure(figsize=(12, 8))
    # Create a 2D visualization of object detection
    detection_vis = np.zeros((512, 512, 3), dtype=np.uint8)
    detection_vis[:, :, :] = 50  # Dark gray background
    
    # Draw ground grid
    for i in range(0, 512, 50):
        detection_vis[i, :, :] = 80  # Horizontal lines
        detection_vis[:, i, :] = 80  # Vertical lines
    
    # Draw coordinate axes
    detection_vis[256, 256:, 2] = 255  # X-axis (blue)
    detection_vis[:256, 256, 0] = 255  # Y-axis (red)
    
    # Draw detected objects
    boxes_3d = detection_results['boxes_3d']
    class_labels = detection_results['class_labels']
    
    for i, box in enumerate(boxes_3d):
        x, y, z, length, width, height, heading = box
        
        # Convert to image coordinates
        ix = int(offset_x + x * scale)
        iy = int(offset_y - y * scale)  # Flip y-axis
        
        # Determine color based on class
        if 'Vehicle' in class_labels[i]:
            color = [0, 0, 255]  # Blue
        elif 'Pedestrian' in class_labels[i]:
            color = [255, 0, 0]  # Red
        elif 'Cyclist' in class_labels[i]:
            color = [255, 165, 0]  # Orange
        else:
            color = [180, 180, 180]  # Gray
        
        # Draw rotated rectangle
        l_scaled = int(length * scale)
        w_scaled = int(width * scale)
        
        # Create rotation matrix
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        
        # Calculate corner points
        corners = np.array([
            [-l_scaled/2, -w_scaled/2],
            [l_scaled/2, -w_scaled/2],
            [l_scaled/2, w_scaled/2],
            [-l_scaled/2, w_scaled/2]
        ])
        
        # Rotate corners
        R = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])
        
        rotated_corners = np.dot(corners, R.T)
        
        # Translate corners to object position
        corners_img = rotated_corners + np.array([ix, iy])
        
        # Draw polygon
        corners_img = corners_img.astype(np.int32)
        cv2.polylines(detection_vis, [corners_img], True, color, 2)
        
        # Add class label
        cv2.putText(
            detection_vis,
            class_labels[i],
            (ix, iy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
    
    plt.imshow(detection_vis)
    plt.title("3D Object Detection (Top-Down View)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "object_detection.png"))
    plt.close()
    
    # 7. Save scene classification results
    with open(os.path.join(output_dir, "scene_classification.txt"), 'w') as f:
        f.write("Scene Classification Results\n")
        f.write("===========================\n\n")
        f.write(f"Primary Category: {scene_stats['primary_category']}\n")
        if 'scenario_class' in scene_stats:
            f.write(f"Scenario Class: {scene_stats['scenario_class']}\n")
        f.write("\nCategory Distribution:\n")
        for category, percentage in scene_stats['category_percentages'].items():
            f.write(f"  {category}: {percentage:.2f}%\n")
        
        f.write("\nScene Composition Analysis:\n")
        f.write(f"  Total Messages: {scene_composition['total_messages']}\n")
        f.write(f"  Unique Topics: {scene_composition['unique_topics']}\n")
        
        f.write("\nObject Detection Results:\n")
        for class_name, count in detection_results['class_counts'].items():
            f.write(f"  {class_name}: {count}\n")
        
        f.write("\nSegmentation Results:\n")
        for class_name, percentage in list(segmentation_results['class_distributions'].items())[:10]:
            f.write(f"  {class_name}: {percentage*100:.2f}%\n")
    
    print(f"Visualizations saved to {output_dir}")
    
    return {
        'point_cloud': point_cloud,
        'camera_data': camera_data,
        'detection_results': detection_results,
        'segmentation_results': segmentation_results,
        'camera_projections': camera_projections,
        'scene_stats': scene_stats,
        'scene_composition': scene_composition
    }

def main():
    """Main function"""
    args = parse_args()
    
    visualize_scene(
        args.mcap_file,
        args.frame_index,
        args.output_dir,
        args.theme,
        args.interactive,
        args.detection_model,
        args.segmentation_model
    )

if __name__ == "__main__":
    main()
