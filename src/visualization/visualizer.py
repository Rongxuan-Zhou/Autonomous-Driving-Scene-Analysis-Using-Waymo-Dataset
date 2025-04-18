import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class WaymoVisualizer:
    """Visualization toolkit for Waymo dataset analysis"""
    
    def __init__(self, theme='dark', figure_size=(12, 8), dpi=100):
        """Initialize visualizer with style settings
        
        Args:
            theme: Visualization theme ('dark' or 'light')
            figure_size: Default figure size for matplotlib
            dpi: Resolution for matplotlib figures
        """
        self.theme = theme
        self.figure_size = figure_size
        self.dpi = dpi
        
        # Configure matplotlib style
        if theme == 'dark':
            plt.style.use('dark_background')
            self.colors = {
                'primary': '#00a8e1',    # Blue
                'secondary': '#ff9e1b',  # Orange
                'accent': '#00cf91',     # Green
                'highlight': '#ff5c8d',  # Pink
                'background': '#1e1e1e', # Dark gray
                'text': '#ffffff'        # White
            }
        else:
            plt.style.use('default')
            self.colors = {
                'primary': '#0072b2',    # Blue
                'secondary': '#e69f00',  # Orange
                'accent': '#009e73',     # Green
                'highlight': '#cc79a7',  # Pink
                'background': '#ffffff', # White
                'text': '#000000'        # Black
            }
        
        # Configure seaborn
        sns.set(style="ticks", font_scale=1.1)
        
        # Create color palette for segmentation
        self._create_segmentation_palette()
    
    def _create_segmentation_palette(self):
        """Create color palette for semantic segmentation visualization"""
        # Palette for 23 semantic classes in Waymo dataset
        colors = [
            [0, 0, 0],       # Undefined - Black
            [0, 0, 255],     # Car - Blue
            [0, 0, 180],     # Truck - Dark Blue
            [0, 0, 120],     # Bus - Darker Blue
            [0, 0, 80],      # Other Vehicle - Very Dark Blue
            [255, 0, 0],     # Pedestrian - Red
            [255, 80, 0],    # Cyclist - Orange
            [255, 120, 0],   # Motorcyclist - Darker Orange
            [255, 255, 0],   # Sign - Yellow
            [255, 255, 100], # Traffic Light - Light Yellow
            [80, 80, 80],    # Pole - Gray
            [255, 100, 255], # Construction Cone - Pink
            [0, 255, 200],   # Bicycle - Cyan
            [0, 200, 255],   # Motorcycle - Light Blue
            [120, 80, 0],    # Building - Brown
            [0, 155, 0],     # Vegetation - Green
            [0, 100, 0],     # Tree Trunk - Dark Green
            [155, 155, 155], # Curb - Light Gray
            [100, 100, 100], # Road - Medium Gray
            [255, 255, 255], # Lane Marker - White
            [155, 100, 0],   # Walkable - Light Brown
            [155, 155, 0],   # Sidewalk - Olive
            [50, 50, 50]     # Ground - Dark Gray
        ]
        
        # Normalize to 0-1 range for matplotlib
        self.segmentation_palette = np.array(colors) / 255.0
    
    def plot_topic_distribution(self, topic_counter, top_n=10, title="Topic Distribution", 
                              save_path=None, interactive=False):
        """Visualize distribution of message topics
        
        Args:
            topic_counter: Counter object or dictionary with topic counts
            top_n: Number of top topics to display
            title: Plot title
            save_path: Optional path to save visualization
            interactive: If True, use plotly for interactive visualization
            
        Returns:
            Figure object
        """
        # Get top topics
        top_topics = dict(sorted(topic_counter.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        if interactive:
            # Create interactive plotly figure
            fig = px.bar(
                x=list(top_topics.values()),
                y=list(top_topics.keys()),
                orientation='h',
                title=title,
                labels={'x': 'Message Count', 'y': 'Topic'},
                color=list(top_topics.values()),
                color_continuous_scale='Viridis'
            )
            
            # Update layout
            fig.update_layout(
                height=max(500, 50 * len(top_topics)),
                width=800,
                template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=50, b=10)
            )
            
            if save_path:
                fig.write_html(save_path)
            
            return fig
        else:
            # Create matplotlib figure
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot horizontal bar chart with custom colors
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['accent'], self.colors['highlight']] * 10
            bars = plt.barh(
                range(len(top_topics)), 
                list(top_topics.values()),
                color=colors[:len(top_topics)]
            )
            
            # Customize appearance
            plt.yticks(range(len(top_topics)), list(top_topics.keys()))
            plt.xlabel('Message Count')
            plt.title(title)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(
                    width + 0.5, 
                    bar.get_y() + bar.get_height()/2,
                    f'{int(width):,}',
                    ha='left', 
                    va='center',
                    color=self.colors['text'],
                    fontweight='bold'
                )
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
            
            return plt.gcf()
    
    def visualize_point_cloud(self, point_cloud, colors=None, custom_draw=None, 
                            background_color=None, window_name="Point Cloud Visualization"):
        """Visualize 3D point cloud using Open3D
        
        Args:
            point_cloud: Nx3 array of point coordinates or Open3D PointCloud object
            colors: Optional Nx3 array of RGB colors for each point
            custom_draw: Optional function for custom rendering logic
            background_color: Optional background color override
            window_name: Title for visualization window
            
        Returns:
            Open3D Visualizer object
        """
        # Create point cloud object if not already one
        if isinstance(point_cloud, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud)
            
            if colors is not None:
                # Normalize colors if in range 0-255
                if np.max(colors) > 1:
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd = point_cloud
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        
        # Add point cloud
        vis.add_geometry(pcd)
        
        # Configure visualization settings
        opt = vis.get_render_option()
        
        # Set background color
        if background_color is None:
            bg_color = [0.1, 0.1, 0.1] if self.theme == 'dark' else [1, 1, 1]
        else:
            bg_color = background_color
        opt.background_color = bg_color
        
        # Configure other rendering options
        opt.point_size = 2.0
        opt.show_coordinate_frame = True
        
        # Set default camera viewpoint (top-down with slight angle)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        # Apply custom rendering logic if provided
        if custom_draw is not None:
            custom_draw(vis)
        
        return vis
    
    def visualize_segmentation(self, points, segment_indices, class_names=None,
                             window_name="Semantic Segmentation"):
        """Visualize semantically segmented point cloud
        
        Args:
            points: Nx3 array of point coordinates
            segment_indices: N array of semantic class indices
            class_names: Optional dictionary mapping indices to class names
            window_name: Title for visualization window
            
        Returns:
            Open3D Visualizer object
        """
        # Set colors based on segmentation classes
        colors = self.segmentation_palette[segment_indices % len(self.segmentation_palette)]
        
        # Create visualization
        vis = self.visualize_point_cloud(
            points, 
            colors=colors,
            window_name=window_name
        )
        
        # Add legend if class names are provided
        if class_names is not None:
            # Count instances of each class
            unique_indices, counts = np.unique(segment_indices, return_counts=True)
            
            # Sort by frequency
            sort_idx = np.argsort(counts)[::-1]
            sorted_indices = unique_indices[sort_idx]
            sorted_counts = counts[sort_idx]
            
            # Create custom draw function to add text
            def add_legend(vis):
                # Get render window size
                renderer = vis.get_render_option()
                window_width = vis.get_window_size()[0]
                window_height = vis.get_window_size()[1]
                
                # Add class information as text
                for i, class_idx in enumerate(sorted_indices[:10]):  # Show top 10 classes
                    if class_idx in class_names:
                        class_name = class_names[class_idx]
                        count = sorted_counts[i]
                        percentage = 100 * count / len(segment_indices)
                        
                        color = self.segmentation_palette[class_idx % len(self.segmentation_palette)]
                        color_str = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
                        
                        text = f"{class_name}: {count:,} points ({percentage:.1f}%)"
                        # Add text using renderer
                        # Note: Actual text rendering requires custom implementation
                        # as Open3D visualization lacks text support
                        # This would typically be done using a custom GUI framework
            
            # We would apply the legend in a real implementation
            # Currently not applicable due to Open3D limitations
        
        return vis
    
    def visualize_objects_3d(self, point_cloud, objects, window_name="3D Object Detection"):
        """Visualize 3D object detections in point cloud
        
        Args:
            point_cloud: Nx3 array of point coordinates
            objects: Dictionary with detection results including boxes_3d
            window_name: Title for visualization window
            
        Returns:
            Open3D Visualizer object
        """
        # Extract detection boxes
        boxes_3d = objects.get('boxes_3d', [])
        class_labels = objects.get('class_labels', ['Unknown'] * len(boxes_3d))
        scores = objects.get('scores', [1.0] * len(boxes_3d))
        
        # Define colors for different classes
        class_colors = {
            'Vehicle': [0, 0, 1],      # Blue
            'Pedestrian': [1, 0, 0],   # Red
            'Cyclist': [1, 0.5, 0],    # Orange
            'Sign': [1, 1, 0],         # Yellow
            'Unknown': [0.5, 0.5, 0.5] # Gray
        }
        
        # Create custom drawing function for visualizer
        def draw_3d_boxes(vis):
            # Get renderer
            render_option = vis.get_render_option()
            
            # For each detection box
            for i, box in enumerate(boxes_3d):
                # Extract box parameters
                x, y, z, length, width, height, heading = box
                
                # Get class and color
                class_label = class_labels[i]
                score = scores[i]
                
                color = class_colors.get(class_label, [0.5, 0.5, 0.5])
                
                # Create box geometry
                box_points = o3d.utility.Vector3dVector([
                    [x - length/2, y - width/2, z - height/2],
                    [x + length/2, y - width/2, z - height/2],
                    [x + length/2, y + width/2, z - height/2],
                    [x - length/2, y + width/2, z - height/2],
                    [x - length/2, y - width/2, z + height/2],
                    [x + length/2, y - width/2, z + height/2],
                    [x + length/2, y + width/2, z + height/2],
                    [x - length/2, y + width/2, z + height/2]
                ])
                
                # Create line indices for box edges
                lines = o3d.utility.Vector2iVector([
                    [0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]
                ])
                
                # Create line set and set properties
                line_set = o3d.geometry.LineSet(box_points, lines)
                line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
                
                # Apply rotation for heading
                R = np.array([
                    [np.cos(heading), -np.sin(heading), 0],
                    [np.sin(heading), np.cos(heading), 0],
                    [0, 0, 1]
                ])
                
                # Apply rotation around z-axis at box center
                center = np.array([x, y, z])
                points_centered = np.asarray(box_points) - center
                points_rotated = np.dot(points_centered, R.T) + center
                line_set.points = o3d.utility.Vector3dVector(points_rotated)
                
                # Add to visualization
                vis.add_geometry(line_set)
                
                # Add text for class and score (not directly supported in Open3D)
                # This would require a custom GUI framework integration
        
        # Create point cloud visualization with custom drawing
        return self.visualize_point_cloud(
            point_cloud,
            custom_draw=draw_3d_boxes,
            window_name=window_name
        )
    
    def create_interactive_dashboard(self, data, config=None):
        """Create interactive dashboard for dataset analysis
        
        Args:
            data: Dictionary containing analysis results
            config: Optional configuration for dashboard layout
            
        Returns:
            Plotly dashboard figure
        """
        # Use provided config or default layout
        if config is None:
            config = {
                'layout': [
                    ['topic_distribution', 'message_size_distribution'],
                    ['time_series', 'class_distribution']
                ],
                'height': 900,
                'width': 1200
            }
        
        # Create subplot grid based on layout
        rows = len(config['layout'])
        cols = max(len(row) for row in config['layout'])
        
        # Define subplot titles
        subplot_titles = []
        for row in config['layout']:
            for panel in row:
                if panel in data:
                    subplot_titles.append(panel.replace('_', ' ').title())
                else:
                    subplot_titles.append('')
        
        # Create subplot figure
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Add content to each subplot
        for r, row in enumerate(config['layout']):
            for c, panel in enumerate(row):
                row_idx = r + 1
                col_idx = c + 1
                
                if panel == 'topic_distribution' and 'topic_counter' in data:
                    # Add topic distribution bar chart
                    topics = list(data['topic_counter'].keys())[:10]
                    counts = list(data['topic_counter'].values())[:10]
                    
                    fig.add_trace(
                        go.Bar(
                            y=topics,
                            x=counts,
                            orientation='h',
                            marker=dict(
                                color=counts,
                                colorscale='Viridis'
                            ),
                            name='Topics'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                
                elif panel == 'message_size_distribution' and 'message_sizes' in data:
                    # Add message size histogram
                    fig.add_trace(
                        go.Histogram(
                            x=data['message_sizes'],
                            nbinsx=50,
                            marker=dict(
                                color=self.colors['primary'],
                                line=dict(color='white', width=0.5)
                            ),
                            name='Message Sizes'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                
                elif panel == 'time_series' and 'timestamps' in data:
                    # Add time series plot
                    fig.add_trace(
                        go.Scatter(
                            x=data['timestamps'],
                            y=data['message_counts'],
                            mode='lines',
                            line=dict(
                                color=self.colors['accent'],
                                width=2
                            ),
                            name='Messages Over Time'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
                
                elif panel == 'class_distribution' and 'class_counts' in data:
                    # Add class distribution pie chart
                    labels = list(data['class_counts'].keys())
                    values = list(data['class_counts'].values())
                    
                    fig.add_trace(
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.4,
                            marker=dict(
                                colors=px.colors.qualitative.Pastel
                            ),
                            name='Object Classes'
                        ),
                        row=row_idx,
                        col=col_idx
                    )
        
        # Update layout
        fig.update_layout(
            height=config['height'],
            width=config['width'],
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig

    def create_sensor_fusion_visualization(self, scene_data):
        """Create multi-sensor fusion visualization
        
        Args:
            scene_data: Dictionary with processed scene data
            
        Returns:
            Interactive visualization dashboard
        """
        # Create multi-panel visualization layout
        fig = make_subplots(
            rows=2, 
            cols=3,
            specs=[
                [{"type": "image"}, {"type": "image"}, {"type": "image"}],
                [{"type": "scene", "colspan": 3}, None, None]
            ],
            subplot_titles=(
                "Front Camera", "Front Left Camera", "Front Right Camera", 
                "LiDAR Point Cloud with Camera Fusion"
            )
        )
        
        # Add camera images with projected LiDAR points
        camera_positions = {
            'FRONT': (1, 1),
            'FRONT_LEFT': (1, 2),
            'FRONT_RIGHT': (1, 3)
        }
        
        for camera_name, (row, col) in camera_positions.items():
            if camera_name in scene_data['camera_data']:
                # Get camera visualization with projected points
                camera_vis = scene_data['camera_data'][camera_name]['visualization']
                
                # Add to subplot
                fig.add_trace(
                    go.Image(z=camera_vis),
                    row=row,
                    col=col
                )
        
        # Add 3D point cloud with detections
        point_cloud = scene_data['point_cloud']['points']
        colors = scene_data['segmentation_results']['colors'] * 255
        
        # Create scatter3d trace for points
        fig.add_trace(
            go.Scatter3d(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                z=point_cloud[:, 2],
                mode='markers',
                marker=dict(
                    size=1,
                    color=['rgb({},{},{})'.format(r, g, b) for r, g, b in colors.astype(int)],
                    opacity=0.7
                ),
                hoverinfo='none'
            ),
            row=2,
            col=1
        )
        
        # Add 3D boxes for detections
        boxes_3d = scene_data['detection_results']['boxes_3d']
        corners = scene_data['detection_results']['corners']
        class_labels = scene_data['detection_results']['class_labels']
        
        for i, box in enumerate(boxes_3d):
            x, y, z, length, width, height, heading = box
            
            # Get class for color
            class_name = class_labels[i]
            
            # Define color based on class
            if 'Vehicle' in class_name:
                color = 'rgb(0, 0, 255)'  # Blue
            elif 'Pedestrian' in class_name:
                color = 'rgb(255, 0, 0)'  # Red
            elif 'Cyclist' in class_name:
                color = 'rgb(255, 165, 0)'  # Orange
            else:
                color = 'rgb(180, 180, 180)'  # Gray
            
            # Add box as mesh3d
            corner_points = corners[i]
            
            # Define vertices (corners)
            x_corners = corner_points[:, 0]
            y_corners = corner_points[:, 1]
            z_corners = corner_points[:, 2]
            
            # Define faces for a cuboid
            i0, i1, i2, i3, i4, i5, i6, i7 = range(8)
            faces = np.array([
                [i0, i1, i2], [i0, i2, i3],  # Bottom face
                [i4, i5, i6], [i4, i6, i7],  # Top face
                [i0, i1, i5], [i0, i5, i4],  # Side face
                [i1, i2, i6], [i1, i6, i5],  # Side face
                [i2, i3, i7], [i2, i7, i6],  # Side face
                [i3, i0, i4], [i3, i4, i7]   # Side face
            ])
            
            # Add box as mesh3d
            fig.add_trace(
                go.Mesh3d(
                    x=x_corners,
                    y=y_corners,
                    z=z_corners,
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=0.7,
                    hoverinfo='text',
                    hovertext=f"{class_name}<br>Confidence: {scene_data['detection_results']['scores'][i]:.2f}"
                ),
                row=2,
                col=1
            )
        
        # Configure 3D scene
        fig.update_scenes(
            aspectmode='data',
            xaxis_title='X (forward)',
            yaxis_title='Y (left)',
            zaxis_title='Z (up)',
            camera=dict(
                eye=dict(x=0, y=0, z=3),
                up=dict(x=0, y=0, z=1)
            )
        )
        
        # Update layout
        fig.update_layout(
            height=900,
            width=1200,
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False
        )
        
        return fig

    def create_analysis_dashboard(self, scene_data, historical_data=None):
        """Create statistical analysis dashboard
        
        Args:
            scene_data: Dictionary with current scene data
            historical_data: Optional dictionary with historical data for comparison
            
        Returns:
            Interactive statistical dashboard
        """
        # Create subplot layout
        fig = make_subplots(
            rows=2, 
            cols=2,
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ],
            subplot_titles=(
                "Object Distribution by Class", 
                "Road vs. Non-Road Points",
                "Spatial Density Map",
                "Object Distance Distribution"
            )
        )
        
        # 1. Object distribution by class
        object_classes = scene_data['detection_results']['class_labels']
        class_counts = {}
        for cls in object_classes:
            if cls in class_counts:
                class_counts[cls] += 1
            else:
                class_counts[cls] = 1
        
        # Sort by count
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_classes)
        
        # Add to subplot
        fig.add_trace(
            go.Bar(
                x=classes,
                y=counts,
                marker_color=['blue' if 'Vehicle' in c else 
                             'red' if 'Pedestrian' in c else 
                             'orange' if 'Cyclist' in c else 
                             'gray' for c in classes]
            ),
            row=1,
            col=1
        )
        
        # 2. Road vs. Non-Road points
        # Identify road-related classes from segmentation
        road_classes = [18, 19]  # Road, Lane Marker
        segment_indices = scene_data['segmentation_results']['segment_indices']
        
        road_mask = np.isin(segment_indices, road_classes)
        road_points = np.sum(road_mask)
        non_road_points = len(segment_indices) - road_points
        
        # Add to subplot
        fig.add_trace(
            go.Pie(
                labels=['Road', 'Non-Road'],
                values=[road_points, non_road_points],
                marker_colors=['gray', 'green']
            ),
            row=1,
            col=2
        )
        
        # 3. Spatial density map
        points = scene_data['point_cloud']['points']
        
        # Create 2D histogram (top-down view)
        x_range = (-50, 50)
        y_range = (-50, 50)
        bin_size = 1.0
        
        x_bins = np.arange(x_range[0], x_range[1] + bin_size, bin_size)
        y_bins = np.arange(y_range[0], y_range[1] + bin_size, bin_size)
        
        # Compute 2D histogram
        H, x_edges, y_edges = np.histogram2d(
            points[:, 0],  # x coordinates (forward)
            points[:, 1],  # y coordinates (left)
            bins=[x_bins, y_bins]
        )
        
        # Add to subplot
        fig.add_trace(
            go.Heatmap(
                z=H.T,  # Transpose for correct orientation
                x=x_edges[:-1],
                y=y_edges[:-1],
                colorscale='Viridis',
                showscale=False
            ),
            row=2,
            col=1
        )
        
        # 4. Object distance distribution
        object_distances = []
        object_types = []
        
        # Calculate distance from ego vehicle to each object
        for i, box in enumerate(scene_data['detection_results']['boxes_3d']):
            x, y, z, length, width, height, heading = box
            distance = np.sqrt(x**2 + y**2)  # 2D distance in xy-plane
            object_distances.append(distance)
            object_types.append(scene_data['detection_results']['class_labels'][i])
        
        # Add to subplot
        for cls in set(object_types):
            # Filter distances for this class
            cls_distances = [d for d, t in zip(object_distances, object_types) if t == cls]
            
            # Choose color based on class
            color = 'blue' if 'Vehicle' in cls else \
                    'red' if 'Pedestrian' in cls else \
                    'orange' if 'Cyclist' in cls else 'gray'
            
            # Add trace
            fig.add_trace(
                go.Scatter(
                    x=cls_distances,
                    y=[cls] * len(cls_distances),
                    mode='markers',
                    marker=dict(
                        color=color,
                        size=10,
                        opacity=0.7
                    ),
                    name=cls
                ),
                row=2,
                col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            template='plotly_dark' if self.theme == 'dark' else 'plotly_white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Object Class", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        
        fig.update_xaxes(title_text="X (forward) [m]", row=2, col=1)
        fig.update_yaxes(title_text="Y (left) [m]", row=2, col=1)
        
        fig.update_xaxes(title_text="Distance from Ego Vehicle [m]", row=2, col=2)
        fig.update_yaxes(title_text="Object Class", row=2, col=2)
        
        return fig
