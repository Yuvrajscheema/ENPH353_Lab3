#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError

class ImageConverter:
    def __init__(self):
        # ROS publishers and subscribers
        self.image_pub = rospy.Publisher("/enph353/camera/processed", Image, queue_size=10)
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self.image_sub = rospy.Subscriber("/enph353/camera/image_raw", Image, self.callback)
        self.bridge = CvBridge()
        self.move = Twist()

        # Line-following parameters
        self.width = 640
        self.height = 480
        self.threshold = 100
        self.subtract_constant = 50
        self.bottom_height = 100
        self.radius = 10
        self.last_center_x = self.width // 2
        self.linear_speed = 0.5  # BASE_SPEED
        self.angular_speed = 0
        self.Kp = 0.04  # Proportional gain

    def callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Image processing with user's filtering
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
        frame_gray = cv2.cvtColor(filtered, cv2.COLOR_RGB2GRAY)
        blurred_gray = cv2.blur(frame_gray, (5, 5))
        processed_gray = blurred_gray.copy()
        processed_gray[processed_gray > self.threshold] = 255
        processed_gray[processed_gray <= self.threshold] = np.clip(
            processed_gray[processed_gray <= self.threshold] - self.subtract_constant, 0, 255)
        subimage = processed_gray[-self.bottom_height:, :]
        y_rel, x = np.where(subimage < 255)
        
        # Compute centroid
        if len(x) > 0:
            pixel_values = subimage[y_rel, x]
            weights = 255 - pixel_values
            self.last_center_x = np.average(x, weights=weights)
            rospy.loginfo(f"Line center x={self.last_center_x}")
            error = (self.width / 2) - self.last_center_x
            self.angular_speed = self.Kp * error
            self.linear_speed = 0.5  # BASE_SPEED
        else:
            rospy.logwarn("Line center not found, stopping robot")
            self.angular_speed = 0
            self.linear_speed = 0

        # Draw circle on output frame
        output_frame = frame.copy()
        cv2.circle(output_frame, (int(self.last_center_x), int(self.height - self.bottom_height // 2)), 
                   max(1, int(self.radius)), (0, 0, 255), -1)

        # Display image
        cv2.imshow("Image window", output_frame)
        cv2.waitKey(3)

        # Publish processed image
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(output_frame, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        # Publish velocity command
        self.move.linear.x = self.linear_speed
        self.move.angular.z = self.angular_speed
        rospy.loginfo(f"Published linear speed: {self.linear_speed}, angular: {self.angular_speed}")
        self.cmd_vel_pub.publish(self.move)

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
