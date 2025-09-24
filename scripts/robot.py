import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class ImageConverter:
    def __init__(self):
        # ROS publishers and subscribers
        self.image_pub = rospy.Publisher("/rrbot/camera1/processed", Image, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image_sub = rospy.Subscriber("/rrbot/camera1/image_raw", Image, self.callback)
        self.bridge = CvBridge()

        # Line-following parameters
        self.width = 640
        self.height = 480
        self.threshold = 100
        self.subtract_constant = 50
        self.bottom_height = 100
        self.radius = 10
        self.last_center_x = self.width // 2
        self.last_center_y = self.height - self.bottom_height // 2

        # Control parameters
        self.linear_speed = 0.2  # m/s
        self.angular_gain = 0.01  # Proportional gain for steering

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Image processing for line following
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Color filtering
        red_rgb = frame_rgb.copy()
        red_rgb[:, :, 1] = 0
        red_rgb[:, :, 2] = 0

        blue_rgb = frame_rgb.copy()
        blue_rgb[:, :, 0] = 0
        blue_rgb[:, :, 1] = 0

        green_rgb = frame_rgb.copy()
        green_rgb[:, :, 0] = 0
        green_rgb[:, :, 2] = 0

        filtered = (red_rgb.astype(np.float32) + green_rgb.astype(np.float32)) / 2 - blue_rgb.astype(np.float32)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        # Grayscale and thresholding
        frame_gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        blurred_gray = cv2.blur(frame_gray, (5, 5))
        processed_gray = blurred_gray.copy()
        processed_gray[processed_gray > self.threshold] = 255
        processed_gray[processed_gray <= self.threshold] = np.clip(
            processed_gray[processed_gray <= self.threshold] - self.subtract_constant, 0, 255)

        # Line detection in bottom subimage
        subimage = processed_gray[-self.bottom_height:, :]
        y_rel, x = np.where(subimage < 255)
        if len(x) > 0:
            pixel_values = subimage[y_rel, x]
            weights = 255 - pixel_values
            self.last_center_x = np.average(x, weights=weights)
            self.last_center_y = self.height - self.bottom_height + np.average(y_rel, weights=weights)

        # Draw circle on output frame
        output_frame = frame.copy()
        cv2.circle(output_frame, (int(self.last_center_x), int(self.last_center_y)), 
                   max(1, int(self.radius)), (0, 0, 255), -1)

        # Display image
        cv2.imshow("Image window", output_frame)
        cv2.waitKey(3)

        # Publish processed image
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_frame, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        # Compute and publish velocity command
        error_x = self.last_center_x - self.width // 2  # Deviation from image center
        angular_z = -self.angular_gain * error_x  # Proportional control
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

    def shutdown(self):
        # Stop robot
        self.cmd_vel_pub.publish(Twist())
        cv2.destroyAllWindows()

def main():
    rospy.init_node('image_converter', anonymous=True)
    ic = ImageConverter()
    rospy.on_shutdown(ic.shutdown)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")

if __name__ == '__main__':
    main()
