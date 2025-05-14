#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation, label, find_objects
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy
import cv2

class SkeletonizedPath(Node):
    def __init__(self):
        super().__init__('skeletonized_path')

        # Declare parameters
        self.declare_parameter('map_image_path', '/home/beta/RINS-beta-robot/map/map.pgm')
        self.declare_parameter('path_topic', '/global_path')
        self.declare_parameter('dilation_pixels', 5)
        self.declare_parameter('resolution', 0.05)  # Map resolution in meters/pixel
        self.declare_parameter('map_origin_x', -3.31)  # Example value, adjust based on your map
        self.declare_parameter('map_origin_y', -0.411)  # Example value, adjust based on your map

        # Get parameters
        self.map_image_path = self.get_parameter('map_image_path').get_parameter_value().string_value
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.dilation_pixels = self.get_parameter('dilation_pixels').get_parameter_value().integer_value
        self.resolution = self.get_parameter('resolution').get_parameter_value().double_value

        # Publisher
        self.path_pub = self.create_publisher(
            Path,
            self.path_topic,
            QoSProfile(
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1
            )
        )
        
        # Process the map and publish the path
        self.process_map_and_publish_path()

    def process_map_and_publish_path(self):
        """Process the map image, generate skeletonized path, and publish it."""
        try:
            # Read and preprocess the map
            self.get_logger().info(f"Reading map from {self.map_image_path}")
            map_image = self.read_pgm(self.map_image_path)
            
            # Generate skeletonized path
            self.get_logger().info("Generating skeletonized path...")
            path_points = self.generate_skeleton_path(map_image)
            # 
            # Publish the path
            self.publish_path(path_points)
            self.get_logger().info(f"Published global path with {len(path_points)} waypoints to {self.path_topic}")
            
        except Exception as e:
            self.get_logger().error(f"Error processing map: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def read_pgm(self, filename):
        """Read a PGM file."""
        with open(filename, 'rb') as f:
            line = f.readline().decode('ascii').strip()
            if line != 'P5':
                raise ValueError("Not a PGM image (P5 format)")
            
            # Skip comments
            while True:
                line = f.readline().decode('ascii').strip()
                if not line.startswith('#'):
                    break
            
            width, height = map(int, line.split())
            maxval = int(f.readline().decode('ascii').strip())
            
            # Read image data
            image = np.frombuffer(f.read(), dtype=np.uint8 if maxval < 256 else np.uint16)
            image = image.reshape((height, width))
            
        return image


    def clean_image(self, image):
        """Clean the image by removing small objects and dilating obstacles."""
        # Thresholding to create a binary image
        binary_map = (image > 2).astype(np.uint8)
        
        # Label connected components
        labeled_map, num_features = label(binary_map)
        
        # Find objects and remove small ones
        objects = find_objects(labeled_map)
        for obj in objects:
            if obj is not None:
                if obj[0].stop - obj[0].start < 10 or obj[1].stop - obj[1].start < 10:
                    labeled_map[obj] = 0
        
        # Create a binary image of obstacles
        cleaned_image = labeled_map > 0
        
        # Dilate the obstacles to make them appear larger using an elliptical structuring element
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.dilation_pixels * 2 + 1, self.dilation_pixels * 2 + 1))
        dilated_image = binary_dilation(~cleaned_image, structure=structuring_element)
        
        # Invert the dilated image to get the cleaned free space
        cleaned_image = ~dilated_image
        
        return cleaned_image
    
    def generate_skeleton_path(self, image):
        """Generate a skeletonized path from the map image."""
        # Convert image to binary (free space = 1, obstacles = 0)
        # Assuming that obstacles are black (value < 50)
        # binary_map = (image > 245).astype(np.uint8)
        # self.get_logger().info(f"Binary map shape: {binary_map.shape}, unique values: {np.unique(binary_map)}")
        
        # # Dilate obstacles for safety
        # SE = np.ones((self.dilation_pixels, self.dilation_pixels), dtype=bool)
        # dilated_obstacles = binary_dilation(~binary_map, SE)
        # safe_space = ~dilated_obstacles
        
        # Apply skeletonization
        new_space = self.clean_image(image)
        # Save the cleaned image for visualization
        cleaned_image_path = self.map_image_path.replace('.pgm', '_cleaned_real.pgm')
        with open(cleaned_image_path, 'wb') as f:
            f.write(b'P5\n')
            f.write(f'# Cleaned image generated by SkeletonizedPath\n'.encode('ascii'))
            f.write(f'{new_space.shape[1]} {new_space.shape[0]}\n255\n'.encode('ascii'))
            f.write((new_space * 255).astype(np.uint8).tobytes())

        self.get_logger().info(f"Cleaned image saved to {cleaned_image_path}")

        skeleton = skeletonize(new_space)
        self.get_logger().info(f"Skeleton shape: {skeleton.shape}, unique values: {np.unique(skeleton)}")

        # Extract path points from skeleton
        y_indices, x_indices = np.where(skeleton)
        path_points = np.column_stack((x_indices, y_indices))
        
        # Save the skeleton on the map image for visualization
        output_image = (new_space).astype(np.uint8) * 255  # Convert dilated space to grayscale
        output_image[skeleton] = 0  # Mark skeleton with black
        
        # Save the modified image to a new file
        output_path = self.map_image_path.replace('.pgm', '_skeleton_proba_real.pgm')
        with open(output_path, 'wb') as f:
            f.write(b'P5\n')
            f.write(f'# Generated by SkeletonizedPath\n'.encode('ascii'))
            f.write(f'{image.shape[1]} {image.shape[0]}\n255\n'.encode('ascii'))
            f.write(output_image.astype(np.uint8).tobytes())
        
        self.get_logger().info(f"Skeleton path saved to {output_path}")
        
        # Thin out the path points to reduce computational load
        # Take every nth point (adjust as needed)
        # Order the path points by x-coordinate first
        path_points = path_points[np.argsort(path_points[:, 0])]
        sparse_path = path_points[::25]
        if len(sparse_path) < 2:  # If too few points after thinning, use all
            sparse_path = path_points
            

         # Save the published path points on the map image for visualization
        output_image_with_path = (image > 2).astype(np.uint8) * 255  # Convert map to grayscale
        for point in sparse_path:
            x_pixel = int(point[0])
            y_pixel = int(point[1])
            output_image_with_path[y_pixel, x_pixel] = 0  # Mark path points with black

        # Save the modified image to a new file
        output_path_with_points = self.map_image_path.replace('.pgm', '_path_points_real.pgm')
        with open(output_path_with_points, 'wb') as f:
            f.write(b'P5\n')
            f.write(f'# Generated by SkeletonizedPath\n'.encode('ascii'))
            f.write(f'{image.shape[1]} {image.shape[0]}\n255\n'.encode('ascii'))
            f.write(output_image_with_path.astype(np.uint8).tobytes())

        self.get_logger().info(f"Path points saved to {output_path_with_points}")
    


        return sparse_path

    def order_path_points(self, points):
        """Order path points to form a continuous path."""
        if len(points) < 2:
            return points
            
        # Start with the first point
        ordered = [points[0]]
        remaining = list(points[1:])
        
        # Greedy nearest neighbor
        while remaining:
            current = ordered[-1]
            # Find index of closest point
            distances = np.sqrt(np.sum((np.array(remaining) - current)**2, axis=1))
            idx = np.argmin(distances)
            # Add closest point to ordered list
            ordered.append(remaining.pop(idx))
            
        return np.array(ordered)

    def publish_path(self, path_points):
        """Publish the path as a nav_msgs/Path message with correct coordinate transformation."""
        # Order points to create a coherent path
        ordered_points = self.order_path_points(path_points)
        
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Get the image dimensions for Y-flipping
        height = 0
        try:
            with open(self.map_image_path, 'rb') as f:
                line = f.readline()  # P5
                line = f.readline()  # Skip comment line
                while line.startswith(b'#'):
                    line = f.readline()
                width, height = map(int, line.decode().strip().split())
        except Exception as e:
            self.get_logger().warn(f"Could not read map dimensions: {e}. Using default Y-flip.")
        
        # Define map origin and orientation 
        # These should be read from your map.yaml file
        # Adding parameters for map origin
                
        map_origin_x = self.get_parameter('map_origin_x').get_parameter_value().double_value
        map_origin_y = self.get_parameter('map_origin_y').get_parameter_value().double_value
        
        self.get_logger().info(f"Using map origin: ({map_origin_x}, {map_origin_y})")
        
        # Apply transformations to each point
        for point in ordered_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            
            # Convert from pixels to meters with correct orientation and origin
            pixel_x = point[0]
            pixel_y = point[1]
            
            # Apply resolution scaling and origin offset
            # First convert from pixel to meters
            if height > 0:
                # Flip Y coordinate (image coordinates have y=0 at top, map has y=0 at bottom)
                world_y = (height - pixel_y) * self.resolution
            else:
                world_y = pixel_y * self.resolution
                
            world_x = pixel_x * self.resolution
            
            # Then add origin offset
            pose.pose.position.x = world_x + map_origin_x
            pose.pose.position.y = world_y + map_origin_y
            
            # Set default orientation
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f"Published path with {len(path_msg.poses)} poses")
        
        # Log first and last few points with corrected coordinates
        for i in range(min(5, len(path_msg.poses))):
            self.get_logger().info(f"Transformed path point {i}: "
                                f"({path_msg.poses[i].pose.position.x:.2f}, "
                                f"{path_msg.poses[i].pose.position.y:.2f})")
        if len(path_msg.poses) > 10:
            self.get_logger().info("...")
            for i in range(len(path_msg.poses) - 5, len(path_msg.poses)):
                self.get_logger().info(f"Transformed path point {i}: "
                                    f"({path_msg.poses[i].pose.position.x:.2f}, "
                                    f"{path_msg.poses[i].pose.position.y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    
    # Import scikit-image at runtime to avoid startup delays
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        import sys
        print("Error: This node requires scikit-image. Please install it with:")
        print("pip install scikit-image")
        sys.exit(1)
        
    node = SkeletonizedPath()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
    