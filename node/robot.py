#!/bin/env python3
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

## Portion of the image (from the top) to ignore when detecting the line
START_ROW = 0.80 #: The amount of the upper half that is ignored (%)

## Proportional control gain for angular correction
KP = 0.04

## Base linear speed of the robot
BASE_SPEED = 0.5

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
        

    ##
    # @brief Callback function for processing camera images.
    #
    # Converts ROS Image messages to OpenCV images, finds the line center,
    # computes proportional control corrections, and publishes Twist commands.
    #
    # @param data The incoming ROS Image message.
    #
    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        center = self.find_center(cv_image)
        _, w = cv_image.shape[:2]
        if center is None:
            rospy.logwarn("Line center not found, stopping robot")
            self.angular_speed = 1
            self.linear_speed = 0
        else:
            error = (w / 2) - center
            correction = KP * error
            self.angular_speed = correction
            self.linear_speed = BASE_SPEED

        self.move.angular.z = self.angular_speed
        self.move.linear.x = self.linear_speed
        rospy.loginfo(f"The published linear speed {self.linear_speed} and angular {self.angular_speed}")
        # rospy.loginfo(f"error {error} and image center {w/2}")
        self.pub.publish(self.move)
    
    ##
    # @brief Finds the x-coordinate of the line center in the image.
    #
    # Processes the bottom portion of the image, thresholds it to detect the
    # line, and computes the centroid using image moments.
    #
    # @param frame The OpenCV image (BGR format) from the robot's camera.
    # @return The x-coordinate of the line center, or None if no line is detected.
    #
    def find_center(self, frame):
        h, _ = frame.shape[:2]
        start_row = int(h * START_ROW)
        bottom_region = frame[start_row:, :]

        gray = cv.cvtColor(bottom_region, cv.COLOR_BGR2GRAY)
        # Binary threshold + invert to make line = white
        _, img_bin = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
        img_inv = cv.bitwise_not(img_bin)

        # Compute centroid
        M = cv.moments(img_inv)
        
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            rospy.loginfo(f"Line center x={cx}")
            return cx
        else:
            rospy.logwarn("No line detected!")
            return

##
# @brief Main entry point of the program.
#
# Initializes the ROS node, creates a LineFollower object, and enters
# the ROS spin loop.
#
def main():
    rospy.init_node('robot_controller')
    lf = LineFollower()
    rospy.spin()

##
# @brief Script entry point.
#
if __name__ == '__main__':
    main()
