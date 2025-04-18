#!/usr/bin/env python3
"""
Autonomous Driving Scene Analysis Using Waymo Dataset
Main entry point for the project
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.mcap_processor import WaymoDataProcessor, extract_features
from src.visualization.visualization_demo import visualize_scene

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Autonomous Driving Scene Analysis Using Waymo Dataset')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Process dataset command
    process_parser = subparsers.add_parser('process', help='Process Waymo dataset')
    process_parser.add_argument('--data_dir', type=str, required=True,
                              help='Directory containing Waymo TFRecord files')
    process_parser.add_argument('--output_dir', type=str, default='data/processed',
                              help='Directory to save processed data')
    process_parser.add_argument('--limit', type=int, default=None,
                              help='Limit number of segments to process')
    process_parser.add_argument('--force', action='store_true',
                              help='Force reprocessing of already processed files')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Visualize Waymo dataset')
    visualize_parser.add_argument('--mcap_file', type=str, default='data/foxglove_samples/waymo-scene.mcap',
                                help='Path to MCAP file')
    visualize_parser.add_argument('--frame_index', type=int, default=0,
                                help='Frame index to visualize')
    visualize_parser.add_argument('--output_dir', type=str, default='output',
                                help='Directory to save visualizations')
    visualize_parser.add_argument('--theme', type=str, default='dark', choices=['dark', 'light'],
                                help='Visualization theme')
    visualize_parser.add_argument('--interactive', action='store_true',
                                help='Use interactive visualizations')
    visualize_parser.add_argument('--detection_model', type=str, default=None,
                                help='Path to 3D object detection model')
    visualize_parser.add_argument('--segmentation_model', type=str, default=None,
                                help='Path to semantic segmentation model')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze Waymo dataset')
    analyze_parser.add_argument('--mcap_file', type=str, required=True,
                              help='Path to MCAP file')
    analyze_parser.add_argument('--output_dir', type=str, default='analysis',
                              help='Directory to save analysis results')
    
    return parser.parse_args()

def process_dataset(args):
    """Process Waymo dataset"""
    processor = WaymoDataProcessor(args.data_dir, args.output_dir)
    stats = processor.process_dataset(limit=args.limit, force_reprocess=args.force)
    
    print("\nProcessing Statistics:")
    print(f"Total segments: {stats['total_segments']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total objects: {stats['total_objects']}")
    print(f"Processing time: {stats['processing_time']:.2f} seconds")

def analyze_dataset(args):
    """Analyze Waymo dataset"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract features
    print(f"Extracting features from {args.mcap_file}")
    features_df = extract_features(args.mcap_file)
    
    # Save features to CSV
    features_path = os.path.join(args.output_dir, 'features.csv')
    features_df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(f"Total messages: {len(features_df)}")
    print(f"Unique topics: {features_df['topic'].nunique()}")
    
    if 'message_size' in features_df.columns:
        print(f"Total data size: {features_df['message_size'].sum() / (1024*1024):.2f} MB")
        print(f"Average message size: {features_df['message_size'].mean():.2f} bytes")
    
    if 'datetime' in features_df.columns:
        time_range = features_df['datetime'].max() - features_df['datetime'].min()
        print(f"Time range: {time_range}")
        print(f"Messages per second: {len(features_df) / time_range.total_seconds():.2f}")

def main():
    """Main function"""
    args = parse_args()
    
    if args.command == 'process':
        process_dataset(args)
    elif args.command == 'visualize':
        visualize_scene(
            args.mcap_file,
            args.frame_index,
            args.output_dir,
            args.theme,
            args.interactive,
            args.detection_model,
            args.segmentation_model
        )
    elif args.command == 'analyze':
        analyze_dataset(args)
    else:
        print("Please specify a command: process, visualize, or analyze")
        sys.exit(1)

if __name__ == "__main__":
    main()
