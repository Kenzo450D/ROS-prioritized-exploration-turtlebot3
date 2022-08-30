#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
import copy
import math

class TurtlebotAverageSpeed:
    def __init__(self):\
        # -- initialize node
        rospy.init_node('turtlebot_average_speed', anonymous=True)
        
        # -- subscribers
        rospy.Subscriber("/odom", Odometry,  self.odom_cb)
        
        # -- initialize parameter
        self.calculate_speed = False
        self.init_speed = 0
        self.flag_calculate_speed = False
        self.dist_moved = 0.0
        self.time_elapsed = 0.0
        self.cur_time_elapsed = 0.0
        self.pub = rospy.Publisher('average_speed', Float64, queue_size=10)
        rospy.spin()


    def odom_cb(self, msg):
        # x = msg.pose.pose.position.x
        # y = msg.pose.pose.position.y
        # th = msg.pose.pose.orientation
        self.robot_pose = msg.pose.pose
        # Use a parameter to start calculating when the robot is moving about
        # If the robot is moving, turning, then start calculating the speed
        update_speed_flag_rosparam = rospy.get_param('robot_moving')
        if update_speed_flag_rosparam: # update speed value
            if self.flag_calculate_speed == False:
                #self.init_odom = self.robot_pose
                self.init_time = rospy.get_time()
                self.cur_time_elapsed = 0
                self.prev_odom = self.robot_pose
                self.flag_calculate_speed = True
                self.default_publish_speed()
            else:
                speed = self.get_average_speed()
                self.pub.publish(speed)
                self.prev_odom = self.robot_pose
        else: # when updated speed flag (rosparam) is False
            self.flag_calculate_speed = False # flag to ensure speed is not recalculated
            self.time_elapsed += self.cur_time_elapsed # update time elapsed
            self.cur_time_elapsed = 0.0 # so that the total time is not incorrectly updated
            self.default_publish_speed()


    def default_publish_speed(self):
        # -- publish speed
        if self.time_elapsed == 0:
            self.pub.publish(0.0)
        else:
            speed = self.dist_moved/self.time_elapsed
            self.pub.publish(self.dist_moved / self.time_elapsed)


    def get_average_speed(self):
        p_x = self.prev_odom.position.x # previous x
        p_y = self.prev_odom.position.y # previous y
        c_x = self.robot_pose.position.x # current x
        c_y = self.robot_pose.position.y # current y
        delta_dist = math.sqrt((c_x - p_x)**2 + (c_y - p_y)**2) 
        self.dist_moved += delta_dist
        self.cur_time_elapsed = rospy.get_time() - self.init_time
        
        total_time = self.cur_time_elapsed + self.time_elapsed
        if total_time == 0.0:
            avg_speed = 0.0
        else:
            avg_speed = self.dist_moved / total_time
        # print ("Get Average Speed: Previous: {:.2f}, {:.2f}\tCurrent: {:.2f}, {:.2f}\tMoved: {:.2f}\tTime Diff: {:.2f}\tAverage Speed: {:.2f}".format(p_x, p_y, c_x, c_y, delta_dist, self.cur_time_elapsed, avg_speed))
        return avg_speed


if __name__ == '__main__':
    tas = TurtlebotAverageSpeed()
