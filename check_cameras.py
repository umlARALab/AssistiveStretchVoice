import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class PersonDetectionNode(Node):
    def __init__(self):
        super().__init__('person_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  
            self.image_callback,
            10
        )
        self.subscription  
        self.person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        self.bridge = CvBridge()

    def image_callback(self, msg):
        frame = cv2.rotate(self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8'),cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        persons = self.person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(300, 300))
        for (x, y, w, h) in persons:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Camera Feed", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
