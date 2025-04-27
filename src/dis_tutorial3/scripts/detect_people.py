#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker

from cv_bridge import CvBridge
import cv2
import numpy as np
import csv
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

from ultralytics import YOLO

class detect_faces(Node):
    def __init__(self):
        super().__init__('detect_faces')

        self.declare_parameters('', [('device', '')])
        self.device = self.get_parameter('device').get_parameter_value().string_value

        self.bridge = CvBridge()
        self.faces = []
        self.csv_file = "faces_pos.csv"
        
        self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)
        
        self.marker_pub = self.create_publisher(Marker, "/people_marker", QoSReliabilityPolicy.BEST_EFFORT)
        self.model = YOLO("yolov8n.pt")

        self.get_logger().info("Node initialized, detecting faces...")

    def rgb_callback(self, data):
        self.faces = []
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
            
            for x in res:
                bbox = x.boxes.xyxy
                if bbox.nelement() == 0:
                    continue
                
                bbox = bbox[0]
                cx, cy = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
                self.faces.append((cx, cy))
        
        except Exception as e:
            self.get_logger().error(f"Error processing RGB image: {e}")

    def pointcloud_callback(self, data):
        height, width = data.height, data.width
        if not self.faces:
            return
        
        a = pc2.read_points_numpy(data, field_names=("x", "y", "z")).reshape((height, width, 3))
        face_positions = [a[y, x, :] for x, y in self.faces if 0 <= x < width and 0 <= y < height]
        
        if len(face_positions) >= 10:
            face_positions = np.array(face_positions[:10])
            normal_pca, normal_ransac = self.estimate_plane(face_positions)
            
            # Compute angle between normals
            dot_product = np.dot(normal_pca, normal_ransac)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
            
            if angle < 10:  # If PCA and RANSAC normals agree within 10 degrees
                self.publish_marker(face_positions.mean(axis=0))
                self.save_face_position(face_positions.mean(axis=0))
    
    def estimate_plane(self, points):
        pca = PCA(n_components=3)
        pca.fit(points)
        normal_pca = pca.components_[-1]
        
        ransac = RANSACRegressor()
        ransac.fit(points[:, :2], points[:, 2])
        normal_ransac = np.cross([1, 0, ransac.estimator_.coef_[0]], [0, 1, ransac.estimator_.coef_[1]])
        normal_ransac /= np.linalg.norm(normal_ransac)

        self.get_logger().info(f"PCA Normal: {normal_pca}")
        self.get_logger().info(f"RANSAC Normal: {normal_ransac}")
        return normal_pca, normal_ransac

    def publish_marker(self, position):
        marker = Marker()
        marker.header.frame_id = "/base_link"
        marker.type = Marker.SPHERE
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        self.marker_pub.publish(marker)
    
    def save_face_position(self, position):
        try:
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([position[0], position[1], position[2]])
            self.get_logger().info(f"Face position saved: {position}")
        except Exception as e:
            self.get_logger().error(f"Failed to save face position: {e}")

def main():
    rclpy.init()
    node = detect_faces()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
