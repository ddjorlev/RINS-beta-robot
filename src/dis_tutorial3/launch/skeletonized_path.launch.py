from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dis_tutorial3',
            executable='skeletonized_path.py',
            name='skeletonized_path',
            output='screen',
            parameters=[
                {'map_image_path': '/home/dimitar/colcon_ws/src/dis_tutorial3/maps/map.pgm'},
                {'path_topic': '/global_path'},
                {'dilation_pixels': 6},
                {'resolution': 0.05} 
            ]
        )
    ])