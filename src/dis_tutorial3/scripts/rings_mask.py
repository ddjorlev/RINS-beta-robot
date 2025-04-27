#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1
)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        timer_frequency = 2
        timer_period = 1 / timer_frequency

        self.bridge = CvBridge()
        self.marker_id = 0

        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)

        self.marker_pub = self.create_publisher(Marker, "/ring_markers", qos_profile)

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("3D Detected Rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

    def image_callback(self, data):
        self.get_logger().info(f"New image received. Detecting rings...")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        self.last_rgb_image = cv_image

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        masks = {
            "red": cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (160, 100, 100), (179, 255, 255)),
            "green": cv2.inRange(hsv, (35, 100, 100), (85, 255, 255)),
            "blue": cv2.inRange(hsv, (100, 100, 100), (130, 255, 255)),
            "black": cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        }

        detected_candidates = []

        for color, mask in masks.items():
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            elps = []
            for cnt in contours:
                if cnt.shape[0] >= 20:
                    try:
                        ellipse = cv2.fitEllipse(cnt)
                        elps.append(ellipse)
                    except:
                        continue

            for n in range(len(elps)):
                for m in range(n + 1, len(elps)):
                    e1, e2 = elps[n], elps[m]
                    dist = np.linalg.norm(np.array(e1[0]) - np.array(e2[0]))
                    angle_diff = abs(e1[2] - e2[2])

                    if dist > 5 or angle_diff > 4:
                        continue

                    e1_major, e1_minor = e1[1]
                    e2_major, e2_minor = e2[1]

                    if e1_major >= e2_major and e1_minor >= e2_minor:
                        le, se = e1, e2
                    elif e2_major >= e1_major and e2_minor >= e1_minor:
                        le, se = e2, e1
                    else:
                        continue

                    center_x = int((le[0][0] + se[0][0]) / 2)
                    center_y = int((le[0][1] + se[0][1]) / 2)
                    detected_candidates.append((le, se, color, (center_x, center_y)))

        self.detected_candidates = detected_candidates
        self.show_candidates(cv_image, detected_candidates)

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        if not hasattr(self, 'detected_candidates'):
            return

        image_copy = self.last_rgb_image.copy()

        for e1, e2, color, (cx, cy) in self.detected_candidates:
            if 0 <= cy < depth_image.shape[0] and 0 <= cx < depth_image.shape[1]:
                depth = depth_image[cy, cx]
                if np.isfinite(depth) and depth > 0:
                    print(f"3D {color} ring detected at ({cx},{cy}) with depth {depth:.2f}")

                    # Show in 3D ring image
                    cv2.ellipse(image_copy, e1, (0, 255, 0), 2)
                    cv2.ellipse(image_copy, e2, (0, 255, 0), 2)
                    center = (int(e1[0][0]), int(e1[0][1]))
                    cv2.putText(image_copy, color, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # Publish marker
                    self.publish_marker(cx, cy, depth, color)

        cv2.imshow("3D Detected Rings", image_copy)

        # Optional: visualize depth
        vis_image = np.nan_to_num(depth_image, nan=0.0)
        vis_image = (vis_image / np.max(vis_image) * 255).astype(np.uint8)
        cv2.imshow("Depth window", vis_image)
        cv2.waitKey(1)

    def show_candidates(self, image, candidates):
        for e1, e2, color, _ in candidates:
            cv2.ellipse(image, e1, (0, 255, 0), 2)
            cv2.ellipse(image, e2, (0, 255, 0), 2)
            center = (int(e1[0][0]), int(e1[0][1]))
            cv2.putText(image, color, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Detected rings", image)
        cv2.waitKey(1)

    def publish_marker(self, x, y, z, color_name):
        marker = Marker()
        marker.header.frame_id = "oakd_rgb_camera_optical_frame"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(x) / 100.0
        marker.pose.position.y = float(y) / 100.0
        marker.pose.position.z = float(z)
        marker.pose.orientation.w = 1.0
        marker.scale = Vector3(x=0.05, y=0.05, z=0.05)

        color_map = {
            "red": ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),
            "green": ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            "blue": ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0),
            "black": ColorRGBA(r=0.1, g=0.1, b=0.1, a=1.0),
        }

        marker.color = color_map.get(color_name, ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))

        self.marker_pub.publish(marker)


def main():
    rclpy.init(args=None)
    rd_node = RingDetector()
    rclpy.spin(rd_node)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
