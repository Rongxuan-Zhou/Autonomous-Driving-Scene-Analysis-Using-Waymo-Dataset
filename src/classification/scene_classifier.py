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
        elif 'annotation' in topic or 'label' in topic or 'object' in topic:
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
            features_df['datetime'] = pd.to_datetime(features_df['timestamp'], unit='ns')
        
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

def analyze_scene_composition(features_df):
    """Analyze scene composition based on message topics and sizes
    
    Args:
        features_df: DataFrame containing message features and topics
        
    Returns:
        Dictionary with scene composition analysis
    """
    # Ensure scene_category is present
    if 'scene_category' not in features_df.columns:
        features_df, _ = classify_scene(features_df)
    
    # Calculate message size distribution by category
    size_by_category = {}
    if 'message_size' in features_df.columns:
        for category in features_df['scene_category'].unique():
            category_mask = features_df['scene_category'] == category
            category_sizes = features_df.loc[category_mask, 'message_size']
            
            size_by_category[category] = {
                'mean': category_sizes.mean(),
                'median': category_sizes.median(),
                'min': category_sizes.min(),
                'max': category_sizes.max(),
                'total': category_sizes.sum()
            }
    
    # Calculate temporal distribution
    temporal_distribution = {}
    if 'datetime' in features_df.columns:
        # Group by second
        time_groups = features_df.groupby(features_df['datetime'].dt.floor('1S'))
        
        # Count messages per second
        messages_per_second = time_groups.size()
        
        # Count categories per second
        categories_per_second = time_groups['scene_category'].apply(
            lambda x: x.value_counts().to_dict()
        )
        
        temporal_distribution = {
            'messages_per_second': messages_per_second.to_dict(),
            'categories_per_second': {
                str(ts): cats for ts, cats in categories_per_second.items()
            }
        }
    
    # Calculate topic frequency
    topic_frequency = features_df['topic'].value_counts().to_dict()
    
    # Return comprehensive analysis
    return {
        'size_by_category': size_by_category,
        'temporal_distribution': temporal_distribution,
        'topic_frequency': topic_frequency,
        'total_messages': len(features_df),
        'unique_topics': len(topic_frequency)
    }

def detect_scene_anomalies(features_df, baseline_stats=None):
    """Detect anomalies in scene data compared to baseline or expected patterns
    
    Args:
        features_df: DataFrame containing message features and topics
        baseline_stats: Optional baseline statistics for comparison
        
    Returns:
        Dictionary with detected anomalies
    """
    anomalies = []
    
    # Ensure scene_category is present
    if 'scene_category' not in features_df.columns:
        features_df, _ = classify_scene(features_df)
    
    # Check for missing expected topics
    expected_topics = [
        '/lidar/points',
        '/camera/front',
        '/tf',
        '/objects'
    ]
    
    missing_topics = [topic for topic in expected_topics if topic not in features_df['topic'].values]
    if missing_topics:
        anomalies.append({
            'type': 'missing_topics',
            'description': f"Missing expected topics: {', '.join(missing_topics)}",
            'severity': 'high' if len(missing_topics) > 2 else 'medium'
        })
    
    # Check for unusual message size distribution
    if 'message_size' in features_df.columns:
        size_stats = features_df['message_size'].describe()
        
        # Check for unusually large messages
        max_size = size_stats['max']
        mean_size = size_stats['mean']
        
        if max_size > mean_size * 10:
            large_messages = features_df[features_df['message_size'] > mean_size * 10]
            anomalies.append({
                'type': 'large_messages',
                'description': f"Found {len(large_messages)} unusually large messages",
                'topics': large_messages['topic'].value_counts().to_dict(),
                'severity': 'medium'
            })
    
    # Check for temporal gaps
    if 'datetime' in features_df.columns:
        # Sort by time
        sorted_df = features_df.sort_values('datetime')
        
        # Calculate time differences
        time_diffs = sorted_df['datetime'].diff().dt.total_seconds()
        
        # Find gaps larger than 0.5 seconds
        gaps = time_diffs[time_diffs > 0.5]
        
        if not gaps.empty:
            anomalies.append({
                'type': 'temporal_gaps',
                'description': f"Found {len(gaps)} temporal gaps > 0.5s",
                'max_gap': gaps.max(),
                'gap_locations': sorted_df.iloc[gaps.index]['datetime'].tolist(),
                'severity': 'high' if gaps.max() > 2.0 else 'medium'
            })
    
    # Compare with baseline if provided
    if baseline_stats is not None:
        # Compare category distributions
        if 'category_counts' in baseline_stats:
            baseline_categories = set(baseline_stats['category_counts'].keys())
            current_categories = set(features_df['scene_category'].unique())
            
            # Check for missing categories
            missing_categories = baseline_categories - current_categories
            if missing_categories:
                anomalies.append({
                    'type': 'missing_categories',
                    'description': f"Missing categories compared to baseline: {', '.join(missing_categories)}",
                    'severity': 'medium'
                })
            
            # Check for category distribution differences
            current_counts = features_df['scene_category'].value_counts()
            for category, baseline_count in baseline_stats['category_counts'].items():
                if category in current_counts:
                    current_count = current_counts[category]
                    # If category count differs by more than 50%
                    if abs(current_count - baseline_count) / baseline_count > 0.5:
                        anomalies.append({
                            'type': 'category_distribution_change',
                            'description': f"Category '{category}' count differs significantly from baseline",
                            'baseline': baseline_count,
                            'current': current_count,
                            'change_percent': (current_count - baseline_count) / baseline_count * 100,
                            'severity': 'low'
                        })
    
    return {
        'anomalies': anomalies,
        'anomaly_count': len(anomalies),
        'has_critical_anomalies': any(a['severity'] == 'high' for a in anomalies)
    }
