#! /usr/bin/env python3
# Modified from Samsung Research America
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import time
import subprocess
import os
import numpy as np

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from nav_msgs.msg import Path
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler
from visualization_msgs.msg import MarkerArray

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data


class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        
        # Person detection tracking
        self.detected_person_markers = None
        self.greeting_distance_threshold = 1.5  # meters
        self.greeted_faces = set()  # Keep track of which faces we've already greeted
        self.interrupt_for_greeting = False
        self.current_waypoint_idx = 0
        self.tts_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           "speak.py")  # Use direct path from current script
        self.greeting_text = "Hello there!"
        self.current_task = None  # None, 'waypoint', or 'greeting'
 
        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)
        
        # Subscribe to person markers
        self.people_marker_sub = self.create_subscription(
            MarkerArray,
            '/people_marker',
            self._peopleMarkerCallback,
            10
        )
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose',
                                                      10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        self.get_logger().info(f"Robot commander has been initialized!")
        
    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        self.debug(f"goToPose: Goal accepted, result_future created: {self.result_future}")
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        self.debug(f"spin: Goal accepted, result_future created: {self.result_future}")
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task in cancelTask.')
        if self.result_future:
            self.info("cancelTask: result_future exists, attempting to cancel goal.")
            try:
                self.info("cancelTask: Calling cancel_goal_async...")
                future = self.goal_handle.cancel_goal_async()
                
                # Add timeout to prevent hanging
                self.info("cancelTask: Waiting for cancel_goal_async to complete (with timeout)...")
                timeout_sec = 2.0
                # Just check if succeeded instead of checking for TimeoutException
                success = rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
                
                if not success:
                    self.error(f"cancelTask: cancel_goal_async timed out after {timeout_sec} seconds!")
                else:
                    try:
                        cancel_result = future.result()
                        self.info(f"cancelTask: cancel_goal_async completed with result: {cancel_result}")
                    except Exception as e:
                        self.error(f"cancelTask: Error getting cancel result: {e}")
                        
            except Exception as e:
                self.error(f"cancelTask: Exception occurred during cancellation: {e}")
                # Continue even if there was an error
                
            # Reset the variables regardless of success or failure
            self.info("cancelTask: Resetting result_future and goal_handle")
            self.result_future = None
            self.goal_handle = None
            self.info("cancelTask: result_future and goal_handle reset to None.")
        else:
            self.info("cancelTask: No result_future, nothing to cancel.")
        
        # Force a callback processing opportunity to flush events
        rclpy.spin_once(self, timeout_sec=0.1)
        self.info("cancelTask: Task cancellation procedure completed.")
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            self.debug("isTaskComplete: No result_future, task is complete.")
            return True
        
        try:
            rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
            
            # Make sure result_future is still valid (might have been reset by another callback)
            if not self.result_future:
                self.debug("isTaskComplete: result_future was reset during spin, task is complete.")
                return True
                
            if self.result_future.done():
                result = self.result_future.result()
                if result:
                    self.status = result.status
                    if self.status != GoalStatus.STATUS_SUCCEEDED:
                        self.debug(f'Task failed with status code: {self.status}')
                        return True
                    else:
                        self.debug('Task succeeded!')
                        return True
            else:
                # Timed out, still processing, not complete yet
                self.debug("isTaskComplete: result_future timed out, task not complete.")
                return False
        except Exception as e:
            self.error(f"isTaskComplete: Exception occurred: {e}")
            # If there's an error, consider the task complete to avoid getting stuck
            return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose.pose
        
        # Check if we need to interrupt waypoint navigation to greet someone
        if self.current_task == 'waypoint' and self.detected_person_markers:
            self._check_for_nearby_people()
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def _peopleMarkerCallback(self, msg: MarkerArray):
        """Handle incoming people markers"""
        self.detected_person_markers = msg
        self.info(f"Received people markers: {len(msg.markers)}")
        
        # Log the details of each marker
        for marker in msg.markers:
            self.debug(f"_peopleMarkerCallback: Received marker - NS: {marker.ns}, ID: {marker.id}, Pose: {marker.pose}")
        
        # Check if we should interrupt current navigation to greet someone
        if self.current_task == 'waypoint' and self.current_pose:
            self._check_for_nearby_people()
            
    def _check_for_nearby_people(self):
        """Check if there are people nearby that we should greet"""
        if not self.detected_person_markers or not self.current_pose:
            self.debug("_check_for_nearby_people: No person markers or current pose, returning.")
            return
        
        # Get robot position
        robot_pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y
        ])
        
        # Find goal positions markers (ns="goal_positions")
        goal_markers = []
        person_markers = {}
        for marker in self.detected_person_markers.markers:
            self.debug(f"_check_for_nearby_people: Processing marker - NS: {marker.ns}, ID: {marker.id}") # Added debug log
            if marker.ns == "goal_positions":
                face_id = marker.id
                if face_id not in self.greeted_faces:  # Only consider faces we haven't greeted yet
                    goal_pos = np.array([marker.pose.position.x, marker.pose.position.y])
                    distance = np.linalg.norm(robot_pos - goal_pos)
                    goal_markers.append((face_id, goal_pos, distance))
                    self.debug(f"_check_for_nearby_people: Goal marker found - Face ID: {face_id}, Distance: {distance:.2f}") # Added debug log
                else:
                    self.debug(f"_check_for_nearby_people: Face ID {face_id} already greeted, skipping.") # Added debug log
            elif marker.ns == "person_positions":
                person_markers[marker.id] = np.array([marker.pose.position.x, marker.pose.position.y])
        
        if goal_markers:
            self.debug("_check_for_nearby_people: Goal markers found, processing...") # Added debug log
            # Sort by distance
            goal_markers.sort(key=lambda x: x[2])
            
            # Find closest ungreeted person
            closest_face_id, goal_pos, distance = goal_markers[0]
            
            self.info(f"Closest person (ID: {closest_face_id}) is {distance:.2f} m away")
            
            # If person is close enough and we're not already greeting them
            if distance < self.greeting_distance_threshold and self.current_task != 'greeting':
                self.info(f"Person detected within greeting range! Interrupting navigation to greet.")
                
                # Interrupt current navigation
                if self.result_future:
                    self.debug("_check_for_nearby_people: Cancelling current task because of check for nearby people.")
                    
                    # Save values before cancellation for debugging
                    self.debug(f"_check_for_nearby_people: Before cancellation - result_future: {self.result_future}, goal_handle: {self.goal_handle}")
                    
                    self.cancelTask()
                    self.debug("_check_for_nearby_people: cancelTask() called.")
                    
                    # Check values after cancellation for debugging
                    self.debug(f"_check_for_nearby_people: After cancellation - result_future: {self.result_future}, goal_handle: {self.goal_handle}")
                else:
                    self.debug("_check_for_nearby_people: No result_future, cannot cancel task.")
                
                # Set flag to greet this person
                self.interrupt_for_greeting = True
                self.debug(f"_check_for_nearby_people: interrupt_for_greeting set to True")
                self.person_to_greet = {
                    'face_id': closest_face_id,
                    'goal_pos': goal_pos,
                    'person_pos': person_markers.get(closest_face_id, None)
                }
                self.debug(f"_check_for_nearby_people: interrupt_for_greeting set to True, person_to_greet: {self.person_to_greet}")
            else:
                self.debug(f"_check_for_nearby_people: Person not within greeting range or already greeting. Distance: {distance:.2f}, current_task: {self.current_task}, threshold: {self.greeting_distance_threshold}")
        else:
            self.debug("_check_for_nearby_people: No goal markers found.") # Added debug log

    def sayGreeting(self, text=None):
        """Use text-to-speech to say greeting"""
        if text is None:
            text = self.greeting_text
            
        try:
            # Print to terminal
            self.info(f"Speaking: {text}")
            
            # Check if file exists before running
            if os.path.exists(self.tts_script_path):
                self.info(f"Running TTS script: {self.tts_script_path}")
                subprocess.Popen(["python3", self.tts_script_path, text])
            else:
                self.error(f"TTS script not found at: {self.tts_script_path}")
                # Just print the text as fallback
                print(f"ROBOT SAYS: {text}")
        except Exception as e:
            self.error(f"Error running TTS script: {e}")

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    
    def navigate_to_person(self, goal_pos):
        """Navigate to a person's goal position"""
        self.current_task = 'greeting'
        
        # Create the goal pose
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        
        goal_pose.pose.position.x = goal_pos[0]
        goal_pose.pose.position.y = goal_pos[1]
        goal_pose.pose.position.z = 0.0
        
        # Calculate orientation to face the person
        if self.person_to_greet['person_pos'] is not None:
            # Calculate direction vector from goal position to person position
            direction = self.person_to_greet['person_pos'] - goal_pos
            # Calculate yaw angle
            yaw = np.arctan2(direction[1], direction[0])
            goal_pose.pose.orientation = self.YawToQuaternion(yaw)
        else:
            # Default orientation if we don't have person position
            goal_pose.pose.orientation.w = 1.0
        
        # Navigate to the goal position
        self.info(f"Navigating to greet person (face ID: {self.person_to_greet['face_id']})")
        return self.goToPose(goal_pose)
    

def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    rc.waitUntilNav2Active()

    # while rc.is_docked is None:
    #     rclpy.spin_once(rc, timeout_sec=0.5)

    # if rc.is_docked:
    #     rc.undock()
    
    # Read waypoints from the /global_path topic
    #     waypoints = [
    #     # (-1.0, 0.61, 1.57),  # Starting position         
    #     (-1.74, 0.99, 1.57),          
    #     (-0.913, 2.44, 1.57),       
    #     (-1.99, 2.99, 1.57)          
    # ]
    waypoints = []

    def global_path_callback(msg):
        """Callback to process the global path and extract waypoints."""
        nonlocal waypoints
        waypoints = [(pose.pose.position.x, pose.pose.position.y, 1.57) for pose in msg.poses]
        rc.info(f"Received {len(waypoints)} waypoints from /global_path")

    # Subscribe to the /global_path topic
    global_path_sub = rc.create_subscription(
        Path,
        '/global_path',
        global_path_callback,
        QoSProfile(
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
    )

    # Wait until waypoints are received
    rc.info("Waiting for waypoints from /global_path...")
    while not waypoints:
        rclpy.spin_once(rc, timeout_sec=0.5)
    
    rc.info("Starting navigation through waypoints...")
    
    # Navigate to each waypoint, but check for people between waypoints
    rc.current_waypoint_idx = 0
    
    while rc.current_waypoint_idx < len(waypoints):
        # Process any pending callbacks
        rclpy.spin_once(rc, timeout_sec=0.1)
        
        # If we're interrupting to greet a person
        if rc.interrupt_for_greeting:
            rc.info("Interrupting waypoint navigation to greet a person")
            rc.debug(f"main: interrupt_for_greeting is True, current_task: {rc.current_task}")
            
            # Navigate to the person's goal position
            rc.debug("main: About to call navigate_to_person")
            nav_result = rc.navigate_to_person(rc.person_to_greet['goal_pos'])
            rc.debug(f"main: navigate_to_person returned: {nav_result}")
            
            if nav_result:
                # Wait for navigation to complete
                while not rc.isTaskComplete():
                    rc.info("Moving to greet person...")
                    rclpy.spin_once(rc, timeout_sec=0.5)
                    time.sleep(0.5)
                
                # Say greeting
                rc.info("Reached person! Saying greeting...")
                rc.sayGreeting()

                
                # Mark this face as greeted
                rc.greeted_faces.add(rc.person_to_greet['face_id'])
                
                # Optional: Wait a moment after greeting
                time.sleep(2.0)
                
                rc.info("Greeting complete, resuming waypoint navigation")
            else:
                rc.warn("Failed to start navigation to person.")
                rc.debug(f"main: Failed to navigate to person, state: result_future={rc.result_future}, goal_handle={rc.goal_handle}")
            
            # Reset interrupt flag
            rc.interrupt_for_greeting = False
            rc.current_task = None
            rc.debug("main: interrupt_for_greeting reset to False, current_task reset to None.")
            
        # Otherwise, continue with waypoint navigation
        else:
            # Get current waypoint
            x, y, yaw = waypoints[rc.current_waypoint_idx]
            
            rc.info(f"Navigating to waypoint {rc.current_waypoint_idx + 1}/{len(waypoints)}: ({x}, {y})")
            
            # Create the goal pose
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = 'map'
            goal_pose.header.stamp = rc.get_clock().now().to_msg()
            
            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            goal_pose.pose.orientation = rc.YawToQuaternion(yaw)
            
            # Set current task
            rc.current_task = 'waypoint'
            
            # Send navigation command
            rc.goToPose(goal_pose)
            
            # Wait for completion or interruption
            while not rc.isTaskComplete() and not rc.interrupt_for_greeting:
                rc.info(f"Moving to waypoint {rc.current_waypoint_idx + 1}...")
                rclpy.spin_once(rc, timeout_sec=0.5)
                time.sleep(0.5)
            
            # If we completed the waypoint (not interrupted)
            if not rc.interrupt_for_greeting:
                rc.info(f"Reached waypoint {rc.current_waypoint_idx + 1}!")
                
                # Optional: spin at each waypoint to look around
                if rc.current_waypoint_idx < len(waypoints) - 1:  # Don't spin at the last waypoint
                    rc.info("Spinning to look around...")
                    rc.spin(6.28)  # Spin 360 degrees
                    while not rc.isTaskComplete() and not rc.interrupt_for_greeting:
                        rclpy.spin_once(rc, timeout_sec=0.5)
                        time.sleep(0.5)
                
                # Move to next waypoint if we weren't interrupted
                if not rc.interrupt_for_greeting:
                    rc.current_waypoint_idx += 1
                    rc.current_task = None
    
    rc.info("Completed all waypoints!")
    
    # Optional: return to starting position
    rc.info("Navigation sequence completed successfully.")
    
    rc.destroyNode()

# And a simple example
if __name__=="__main__":
    main()