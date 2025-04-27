#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer

from tf2_ros.transform_listener import TransformListener

from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Point
from std_msgs.msg import Header
from rclpy.time import Time


import numpy as np
import time

import math

import os
import datetime

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)


def write_stamped_points_to_txt(stamped_points, filename):
    """
    Write an array of StampedPoints to a txt file.

    Args:
        stamped_points (list): List of StampedPoint objects to be written.
        filename (str): The path to the text file where points will be stored.
    """
    try:
        if not stamped_points:
            print("No points to write!")
            return

        string_to_write = ""
        for point in stamped_points:
            # Assuming each point has a timestamp and a point (x, y, z)
            point_data = point.point
            
            # Write timestamp and point data (x, y, z)
            string_to_write += (f"{point_data.x} {point_data.y} {point_data.z}\n")

        print("Writing to file:")
        print(string_to_write)  # Debugging output to check what you're writing to the file

        # Write to the file
        with open(filename, 'w') as file:
            file.write(string_to_write)

        print(f"Data successfully written to {filename}")
    
    except Exception as e:
        print(f"Error writing to file: {e}")





class TranformPoints(Node):
    """Demonstrating some convertions and loading the map as an image"""
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 1
        timer_period = 1/timer_frequency

        # Functionality variables
        self.marker_id = 0

        # For listening and loading the 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # For publishing the markers
        self.marker_pub = self.create_publisher(Marker, "/breadcrumbs", QoSReliabilityPolicy.BEST_EFFORT)
	
        # Create a timer, to do the main work.
        #self.timer = self.create_timer(timer_period, self.timer_callback)

        
        self.points_arr = []
        self.point_visit_arr = []        
        
        
        
        """new start"""
        
        self.subscriber_people = self.create_subscription(Marker,"/people_marker", self.nova_funkcija, 10)
    

	
    def nova_funkcija(self, msg):

        def difference_between_stamped(p1, p2):
            p1 = np.array([p1.point.x, p1.point.y, p1.point.z])
            p2 = np.array([p2.point.x, p2.point.y, p2.point.z])

            diff = p1 - p2
            
            return diff

        def distance_between_stamped(p1, p2):
            
            diff = difference_between_stamped(p1, p2)

            distance = math.sqrt(np.dot(diff.T, diff))

            return distance
        
        def add_marker(point, id):
            marker_in_map_frame = self.create_marker(point, id)
            self.marker_pub.publish(marker_in_map_frame)
        
        def adjusted_point(point, robot_position):
            adjusted_point = PointStamped()
            adjusted_point.header.frame_id = "/map"
            adjusted_point.header.stamp = self.get_clock().now().to_msg()

            
            diff = difference_between_stamped(point, robot_position)

            offset = diff / np.linalg.norm(diff) * 0.9
            adjusted_point.point.x = point.point.x - offset[0]
            adjusted_point.point.y = point.point.y - offset[1]
            adjusted_point.point.z = point.point.z - offset[2]

            return adjusted_point



        point_in_robot_frame = PointStamped()
        point_in_robot_frame.header.frame_id = "/base_link"
        point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

        point_in_robot_frame.point.x = msg.pose.position.x
        point_in_robot_frame.point.y = msg.pose.position.y
        point_in_robot_frame.point.z = msg.pose.position.z


        robot_position_in_robot_frame = PointStamped()
        robot_position_in_robot_frame.header.frame_id = "/base_link"
        robot_position_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

        robot_position_in_robot_frame.point.x = 0.
        robot_position_in_robot_frame.point.y = 0.
        robot_position_in_robot_frame.point.z = 0.

        # Now we look up the transform between the base_link and the map frames
        # and then we apply it to our PointStamped
        time_now = rclpy.time.Time()
        timeout = Duration(seconds=0.1)
        try:
            # An example of how you can get a transform from /base_link frame to the /map frame
            # as it is at time_now, wait for timeout for it to become available
            trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
            
            # self.get_logger().info(f"Looks like the transform is available.")

            # Now we apply the transform to transform the point_in_robot_frame to the map frame
            # The header in the result will be copied from the Header of the transform
            point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
            robot_position_in_map_frame = tfg.do_transform_point(robot_position_in_robot_frame, trans)
            
            lock = False
            for point in self.points_arr:
                if distance_between_stamped(point_in_map_frame, point) < 0.5:
                    lock = True
                    break
            if lock == False:

                adjusted = adjusted_point(point_in_map_frame, robot_position_in_map_frame)
                add_marker(adjusted, len(self.point_visit_arr))

                self.points_arr.append(point_in_map_frame)
                self.point_visit_arr.append(adjusted)

                write_stamped_points_to_txt(self.points_arr, "points.txt")
                write_stamped_points_to_txt(self.point_visit_arr, "points_front.txt")


            print("Points count: ", len(self.points_arr))
            print("Last point: ", self.points_arr[-1])
            print("--------------------------------")
        except TransformException as te:
            self.get_logger().info(f"Cound not get the transform: {te}")



    def create_marker(self, point_stamped, marker_id, lifetime=30.0):
        """You can see the description of the Marker message here: https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html"""
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = marker_id

        # Set the scale of the marker
        scale = 0.1
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y
        marker.pose.position.z = point_stamped.point.z

        marker.lifetime = Duration(seconds=lifetime).to_msg()

        return marker

def main():

    rclpy.init(args=None)
    node = TranformPoints()
    
    rclpy.spin(node)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()
