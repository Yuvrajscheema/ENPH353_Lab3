#!/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

## Portion of the image (from the top) to ignore when detecting the line
START_ROW = 0.80 #: The amount of the upper half that is ignored (%)

## Proportional control gain for angular correction
Kp = 0.4

## Base linear speed of the robot
BASE_SPEED = 0.5

threshold = 70
subtract_constant = 20
##
# @class LineFollower
# @brief A ROS node class for line following using camera input.
#
# This class subscribes to the robot's camera feed, processes images to detect
# the line position, computes control commands using a proportional controller,
# and publishes velocity commands to the robot.
#
class LineFollower:
    ##
    # @brief Constructor: initializes subscribers, publishers, and state.
    #
    # Subscribes to the camera topic, publishes velocity commands, and sets
    # up the CvBridge for converting ROS image messages to OpenCV images.
    #
    def __init__(self):
        rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.callback)
        self.pub = rospy.Publisher('/cmd_vel', Twist,queue_size=1)
        self.bridge = CvBridge()
        self.move = Twist()
        self.linear_speed = BASE_SPEED
        self.angular_speed = 0
        

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        center = self.find_center(cv_image)
        _, w = cv_image.shape[:2]
        if center is None:
            self.angular_speed = 1
            self.linear_speed = 0
        else:
            error = (w / 2) - center
            correction = Kp * error
            self.angular_speed = correction
            self.linear_speed = BASE_SPEED

        self.move.angular.z = self.angular_speed
        self.move.linear.x = self.linear_speed
        self.pub.publish(self.move)
 



    def find_center(self, frame):
        height, width = frame.shape[:2]
        bottom_height = int(0.2 * height)

        # Convert to RGB and separate channels
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        red_rgb = frame_rgb.copy()
        red_rgb[:, :, 1] = 0
        red_rgb[:, :, 2] = 0

        blue_rgb = frame_rgb.copy()
        blue_rgb[:, :, 0] = 0
        blue_rgb[:, :, 1] = 0

        green_rgb = frame_rgb.copy()
        green_rgb[:, :, 0] = 0
        green_rgb[:, :, 2] = 0

        # Custom filter: emphasize yellow, suppress blue
        filtered = (red_rgb.astype(np.float32) + green_rgb.astype(np.float32)) / 2 - blue_rgb.astype(np.float32)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        # Grayscale + blur
        frame_gray = cv.cvtColor(filtered, cv.COLOR_RGB2GRAY)
        blurred_gray = cv.blur(frame_gray, (5, 5))

        # Thresholding logic
        processed_gray = blurred_gray.copy()
        processed_gray[processed_gray > threshold] = 255
        processed_gray[processed_gray <= threshold] = np.clip(
            processed_gray[processed_gray <= threshold] - subtract_constant, 0, 255
        )

        # Focus only on bottom region
        subimage = processed_gray[-bottom_height:, :]

        # Weighted centroid
        y_rel, x = np.where(subimage < 255)
        if len(x) > 0:
            pixel_values = subimage[y_rel, x]
            weights = 255 - pixel_values
            cx = np.average(x, weights=weights)
            cy = height - bottom_height + np.average(y_rel, weights=weights)
            return int(cx), int(cy)
        else:
            rospy.loginfo("Couldnt find line")
            return None

def main():
    rospy.init_node('robot_controller')
    lf = LineFollower()
    rospy.spin()

if __name__ == '__main__':
    main()
