import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

class PersonFollower(Node):
    def __init__(self):
        super().__init__('person_follower')
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/stretch/cmd_vel', 10)
        self.bridge = CvBridge()
        self.target_x = None
        self.target_y = None
        self.max_linear_speed = 3.0  # Adjust as needed
        self.max_angular_speed = 3.0  # Adjust as needed
        
        # Action client for navigating to a pose
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav_client.wait_for_server()

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process_image(cv_image)

    def process_image(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect person using a pre-trained model (HOG + SVM)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))

        # Reset target coordinates
        self.target_x = None
        self.target_y = None

        # Draw bounding boxes and find centroid of the first detected person
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            self.target_x = x + w // 2
            self.target_y = y + h // 2
            break  # Only consider the first detected person

        # Calculate movement based on target coordinates
        if self.target_x is not None and self.target_y is not None:
            self.move_towards_person(frame.shape[1], frame.shape[0])

        # Display the resulting frame (Commented out for headless mode)
        #cv2.imshow('Person Follower', frame)
        #cv2.waitKey(1)

    def move_towards_person(self, image_width, image_height):
        # Calculate linear and angular velocities
        linear_speed = self.max_linear_speed
        angular_speed = self.max_angular_speed

        # Calculate direction towards the person
        center_x = image_width // 2
        center_y = image_height // 2

        if self.target_x < center_x - 20:
            # Person is to the left
            self.move_robot(linear_speed, angular_speed)
        elif self.target_x > center_x + 20:
            # Person is to the right
            self.move_robot(linear_speed, -angular_speed)
        else:
            # Person is in the center (adjust as needed)
            self.move_robot(linear_speed, 0.0)

    def move_robot(self, linear_speed, angular_speed):
        # Create Twist message and publish
        twist_msg = Twist()
        twist_msg.linear.x = linear_speed
        twist_msg.angular.z = angular_speed
        self.publisher.publish(twist_msg)

    def navigate_to(self, x, y, theta=0.0):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = np.sin(theta / 2.0)
        goal_msg.pose.pose.orientation.w = np.cos(theta / 2.0)

        self.nav_client.send_goal_async(goal_msg).add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal was rejected.')
            return

        self.get_logger().info('Goal accepted, waiting for result...')
        goal_handle.result().add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result:
            self.get_logger().info('Goal succeeded!')
        else:
            self.get_logger().info('Goal failed.')

def main(args=None):
    rclpy.init(args=args)
    person_follower = PersonFollower()
    
    # Example: Navigate to a specific coordinate (2.0, 1.0)
    person_follower.navigate_to(2.0, 1.0)
    
    rclpy.spin(person_follower)

    person_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
