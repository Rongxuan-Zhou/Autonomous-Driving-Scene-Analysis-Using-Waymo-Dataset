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
        
        print(f"Initialized WaymoDataProcessor with data_dir={data_dir}, output_dir={output_dir}")
    
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
        
        print(f"Found {len(tfrecord_files)} TFRecord files to process")
        
        # Process each file
        for i, tfrecord_path in enumerate(tfrecord_files):
            segment_id = os.path.basename(tfrecord_path).split('.')[0]
            mcap_path = os.path.join(self.output_dir, f"{segment_id}.mcap")
            
            if os.path.exists(mcap_path) and not force_reprocess:
                print(f"Skipping already processed segment {segment_id} ({i+1}/{len(tfrecord_files)})")
                
                # Update stats from existing file
                segment_stats = self._analyze_mcap_file(mcap_path)
                self._update_stats(segment_stats)
                continue
            
            print(f"Processing segment {segment_id} ({i+1}/{len(tfrecord_files)})")
            
            try:
                # Convert TFRecord to MCAP
                convert_tfrecord_to_mcap(tfrecord_path, mcap_path)
                
                # Analyze and update stats
                segment_stats = self._analyze_mcap_file(mcap_path)
                self._update_stats(segment_stats)
                
                print(f"Successfully processed segment {segment_id}")
            except Exception as e:
                print(f"Error processing segment {segment_id}: {e}")
        
        # Calculate total processing time
        self.stats['processing_time'] = time.time() - start_time
        
        # Save final stats
        stats_path = os.path.join(self.output_dir, 'processing_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"Completed dataset processing. Stats saved to {stats_path}")
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
            print(f"Error analyzing MCAP file {mcap_path}: {e}")
        
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

def parse_tfrecord_files(file_paths, output_dir):
    """Parse multiple TFRecord files with error handling and logging
    
    Args:
        file_paths: List of TFRecord file paths
        output_dir: Directory to save parsed frames
        
    Returns:
        Number of frames processed
    """
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

def convert_tfrecord_to_mcap(tfrecord_path, mcap_path):
    """Convert Waymo TFRecord to MCAP format with rich metadata
    
    Args:
        tfrecord_path: Path to TFRecord file
        mcap_path: Output path for MCAP file
        
    Returns:
        None
    """
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

def convert_points_to_pointcloud2(points, frame_id, timestamp):
    """Convert point cloud to ROS PointCloud2 format
    
    Args:
        points: Nx3 array of points
        frame_id: Frame identifier
        timestamp: Timestamp in nanoseconds
        
    Returns:
        PointCloud2 message in dictionary format
    """
    # Create PointCloud2 message
    msg = {
        "header": {
            "frame_id": frame_id,
            "stamp": {
                "sec": timestamp // 1_000_000_000,
                "nanosec": timestamp % 1_000_000_000
            }
        },
        "height": 1,
        "width": len(points),
        "fields": [
            {"name": "x", "offset": 0, "datatype": 7, "count": 1},
            {"name": "y", "offset": 4, "datatype": 7, "count": 1},
            {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            {"name": "intensity", "offset": 12, "datatype": 7, "count": 1}
        ],
        "is_bigendian": False,
        "point_step": 16,
        "row_step": 16 * len(points),
        "is_dense": True,
        "data": points.tolist()  # In a real implementation, this would be binary data
    }
    
    return msg

def convert_image_to_ros(image_data, camera_name, timestamp):
    """Convert image data to ROS Image format
    
    Args:
        image_data: Raw image bytes
        camera_name: Camera identifier
        timestamp: Timestamp in nanoseconds
        
    Returns:
        Image message in dictionary format
    """
    # Create Image message
    msg = {
        "header": {
            "frame_id": camera_name,
            "stamp": {
                "sec": timestamp // 1_000_000_000,
                "nanosec": timestamp % 1_000_000_000
            }
        },
        "height": 0,  # Would be set from actual image
        "width": 0,   # Would be set from actual image
        "encoding": "rgb8",
        "is_bigendian": False,
        "step": 0,    # Would be set from actual image
        "data": []    # In a real implementation, this would be binary data
    }
    
    return msg

def create_tf_message(frame, timestamp):
    """Create TF message from frame data
    
    Args:
        frame: Waymo frame data
        timestamp: Timestamp in nanoseconds
        
    Returns:
        TF message in dictionary format
    """
    # Create TF message
    transforms = []
    
    # Add vehicle to global transform
    if hasattr(frame, 'pose'):
        transforms.append({
            "header": {
                "frame_id": "global",
                "stamp": {
                    "sec": timestamp // 1_000_000_000,
                    "nanosec": timestamp % 1_000_000_000
                }
            },
            "child_frame_id": "vehicle",
            "transform": {
                "translation": {
                    "x": frame.pose.position.x,
                    "y": frame.pose.position.y,
                    "z": frame.pose.position.z
                },
                "rotation": {
                    "x": frame.pose.orientation.x,
                    "y": frame.pose.orientation.y,
                    "z": frame.pose.orientation.z,
                    "w": frame.pose.orientation.w
                }
            }
        })
    
    # Add sensor transforms
    # In a real implementation, these would be extracted from calibration data
    
    msg = {
        "transforms": transforms
    }
    
    return msg

def create_marker_array_from_labels(labels, timestamp):
    """Create visualization markers from object labels
    
    Args:
        labels: Object detection labels
        timestamp: Timestamp in nanoseconds
        
    Returns:
        MarkerArray message in dictionary format
    """
    markers = []
    
    for i, label in enumerate(labels):
        # Create marker for each label
        marker = {
            "header": {
                "frame_id": "vehicle",
                "stamp": {
                    "sec": timestamp // 1_000_000_000,
                    "nanosec": timestamp % 1_000_000_000
                }
            },
            "ns": "objects",
            "id": i,
            "type": 1,  # CUBE
            "action": 0,  # ADD
            "pose": {
                "position": {
                    "x": label.box.center_x,
                    "y": label.box.center_y,
                    "z": label.box.center_z
                },
                "orientation": {
                    "x": 0,
                    "y": 0,
                    "z": np.sin(label.box.heading / 2),
                    "w": np.cos(label.box.heading / 2)
                }
            },
            "scale": {
                "x": label.box.length,
                "y": label.box.width,
                "z": label.box.height
            },
            "color": {
                "r": 1.0 if label.type == 1 else 0.0,  # Red for vehicles
                "g": 1.0 if label.type == 2 else 0.0,  # Green for pedestrians
                "b": 1.0 if label.type == 4 else 0.0,  # Blue for cyclists
                "a": 0.7
            },
            "lifetime": {
                "sec": 0,
                "nanosec": 0
            },
            "frame_locked": False,
            "text": f"{label.type}_{label.id}"
        }
        
        markers.append(marker)
    
    msg = {
        "markers": markers
    }
    
    return msg

def extract_features(mcap_path):
    """Extract features from MCAP file
    
    Args:
        mcap_path: Path to MCAP file
        
    Returns:
        DataFrame with extracted features
    """
    features = []
    
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        
        for schema, channel, message in tqdm(reader.iter_messages()):
            feature = {
                'timestamp': message.log_time,
                'topic': channel.topic,
                'message_size': len(message.data)
            }
            features.append(feature)
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    # Add datetime column
    if 'timestamp' in features_df.columns:
        features_df['datetime'] = pd.to_datetime(features_df['timestamp'], unit='ns')
    
    return features_df
