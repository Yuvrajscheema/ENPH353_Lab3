#!/bin/env python3
"""!
 * @file line_follower.py
 * @brief A ROS node for a robot to follow a line using camera feedback.
 *
 * This node subscribes to a camera's image topic, processes the image to find the center of a line,
 * and publishes twist messages to control the robot's movement. It implements a simple proportional
 * controller to correct the robot's heading based on the line's position in the image.
 * The core logic focuses on processing the bottom portion of the image to efficiently detect
 * the line and calculate the error for the controller.
 *
 * @author Your Name/Team Name (Replace with actual name)
 * @date YYYY-MM-DD (Replace with actual date)
 * @version 1.0.0
 * @see http://wiki.ros.org/rospy
 * @see http://wiki.ros.org/cv_bridge
 * @see http://wiki.ros.org/sensor_msgs
 * @see http://wiki.ros.org/geometry_msgs
 """
import rospy
from cv_bridge import CvBridge, CvBridgeError
import cv2 as cv
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import numpy as np

# Global constants for the line follower logic.
# These values can be tuned for better performance in different environments.

Kp = 0.011


#: The base linear speed of the robot when following the line.
BASE_SPEED = 1.5

#: The threshold value used to binarize the image and find the line.
threshold = 70

#: A constant subtracted from pixel values below the threshold.
#: This helps to suppress noise and emphasize the line.
subtract_constant = 20

class LineFollower:
    """!
    @brief ROS node class for a line following robot.

    This class encapsulates the entire logic for the line follower, including
    subscribing to image data, processing it, and publishing motor commands.
    """
    def __init__(self):
        """!
        @brief Constructor for the LineFollower class.

        Initializes ROS subscribers and publishers, as well as class variables.
        """
        #: The ROS subscriber for the camera image topic.
        rospy.Subscriber('/rrbot/camera1/image_raw', Image, self.callback)
        #: The ROS publisher for the robot's velocity commands.
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        #: The CvBridge object for converting ROS images to OpenCV images.
        self.bridge = CvBridge()
        #: The Twist message object used to control the robot's movement.
        self.move = Twist()
        #: The current linear speed of the robot.
        self.linear_speed = BASE_SPEED
        #: The current angular speed (steering correction) of the robot.
        self.angular_speed = 1
        

    def callback(self, data):
        """!
        @brief Callback function for the image subscriber.

        This function is called whenever a new image message is received.
        It converts the image, finds the line's center, calculates the error,
        and publishes the appropriate twist message.

        @param data The incoming ROS Image message.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        center = self.find_center(cv_image)
        _, w = cv_image.shape[:2]
        if center is None:
            # If the line is not found, reduce speed and spin to find it.
            self.linear_speed = BASE_SPEED * 0.7

        else:
            # Calculate the error based on the line's center relative to the image's center.
            error = (w / 2) - center
            correction = Kp * error 
            self.angular_speed = correction
            self.linear_speed = BASE_SPEED

        # Set the linear and angular velocities in the Twist message.
        self.move.angular.z = self.angular_speed
        self.move.linear.x = self.linear_speed
        # Publish the twist message to control the robot.
        self.pub.publish(self.move)
    


    def find_center(self, frame):
        """!
        @brief Finds the center of the line in the provided image frame.

        This function processes the image by filtering, blurring, and thresholding
        to isolate the line. It then calculates the weighted centroid of the line
        in the bottom section of the image to determine its horizontal position.

        @param frame The OpenCV image frame to process.
        @return The horizontal coordinate of the line's center, or None if the line is not found.
        """
        height, width = frame.shape[:2]
        bottom_height = int(0.3 * height)

        # Convert to RGB and separate channels to apply a custom filter.
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

        # Custom filter: emphasize yellow (red+green), suppress blue.
        filtered = (red_rgb.astype(np.float32) + green_rgb.astype(np.float32)) / 2 - blue_rgb.astype(np.float32)
        filtered = np.clip(filtered, 0, 255).astype(np.uint8)

        # Convert to grayscale and apply blur.
        frame_gray = cv.cvtColor(filtered, cv.COLOR_RGB2GRAY)
        blurred_gray = cv.blur(frame_gray, (5, 5))

        # Thresholding logic to isolate the line.
        processed_gray = blurred_gray.copy()
        processed_gray[processed_gray > threshold] = 255
        processed_gray[processed_gray <= threshold] = np.clip(
            processed_gray[processed_gray <= threshold] - subtract_constant, 0, 255
        )

        # Focus on the bottom region of the image where the line is expected.
        subimage = processed_gray[-bottom_height:, :]

        # Calculate the weighted centroid of the line.
        y_rel, x = np.where(subimage < 255)
        if len(x) > 0:
            pixel_values = subimage[y_rel, x]
            weights = 255 - pixel_values
            cx = np.average(x, weights=weights)
            return int(cx)
        else:
            rospy.loginfo("Couldnt find line")
            return None

def main():
    """!
    @brief The main function of the ROS node.

    Initializes the ROS node and starts the LineFollower object.
    """
    rospy.init_node('robot_controller')
    lf = LineFollower()
    rospy.spin()

if __name__ == '__main__':
    main()

