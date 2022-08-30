#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist
import tf.transformations
import math


class DriveRobotAngleDirection:
    def __init__(self,
                 turn_angle = math.pi, #angle the robot turns (radians)
                 ang_vel = 0.1, #angular velocity of robot
                 error_angle = 0.05, # error of final angle
                 clockwise = False): # clockwise turn if true
        # -- initialize node
        rospy.init_node('drive_robot_angle', anonymous=False)
        self.ang_vel = ang_vel
        self.error_angle = error_angle
        self.delta_theta = turn_angle
        self.clockwise = clockwise
        
        #rospy.loginfo("Delta Theta: {}".format(self.delta_theta))
        
        # -- initalize needed global variables
        self.odomData = rospy.wait_for_message('odom', Odometry, None)
        self.twistData = Twist()
        self.angleTurned = Twist()
        self.angleOffset = 0 #to handle negative orientation values
        self.angleFlag = False
        
        # -- initialize odom subscriber to get odometry data
        rospy.Subscriber('odom', Odometry, self.odomCB)
        
        # -- initalize publisher and rate to move the robot
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
        self.pub_rate = rospy.Rate(60)
        
        # -- Turn by particle angle
        try:
            self.turnAngle(turn_angle, self.pub)
        except rospy.ROSInterruptException:
            pass
        #rospy.spin()
        # -- return
        return
        


    ##odom call back function
    def odomCB(self, data):
        self.odomData = data
            
        


    def turnAngle(self, delta_theta, publisher):
        """ rotate the robot until robot has rotated by the delta_theta.
        Args:
            delta_theta: goal rotation angle. 
            publisher: rospy publisher to publish the twist data on
        Returns:
            publishes the twist to publisher
        Notes:
        http://wiki.ros.org/turtlesim/Tutorials/Rotating%20Left%20and%20Right
        """
        #rospy.loginfo("In turn angle function")
        pub = publisher
        q = [0,0,0,0]
        ##boolean to check to see if we have arrived at the correct orientation
        arrived = False
        ##upperBound and lowerBound to make sure we stop within the correct orientation values
        upperBound = delta_theta + .02
        lowerBound = delta_theta - .02
        current_orientation = self._convert_quaternion_2D_euler_z(self.odomData)
        init_theta = current_orientation
        #rospy.loginfo("Initial Theta: {}".format(init_theta))
        #rospy.loginfo("-"*100)
        
        # -- while robot has not completed the turn
        while not arrived and not rospy.is_shutdown():
            # -- save odomData.orientation into a list for the conversion function
            cur_theta = self._convert_quaternion_2D_euler_z(self.odomData)
            #print "Current Theta: ", cur_theta
            theta_turned = cur_theta - init_theta
            #print "Theta turned: ", theta_turned
            # -- if theta has reached the desired orientation set arrived to true
            #rospy.loginfo("Theta turned: {}".format(theta_turned))
            if theta_turned <= upperBound and theta_turned >= lowerBound:
                arrived = True
                break
            # -- keep moving until about if statement has been satisfied. 
            if self.clockwise:
                self.twistData.angular.z = -self.ang_vel
            else:
                self.twistData.angular.z = self.ang_vel
            pub.publish(self.twistData)
            # -- sleep
            self.pub_rate.sleep()
        # -- once out of loop, stop moving
        self.twistData.angular.z = 0
        pub.publish(self.twistData)

    def _convert_quaternion_2D_euler_z(self, odomData):
        """ convert a quaternion to a euler (XYZ) angle
        Args:
            odomData: odometry data of the robot (/odom topic)
        Return:
            theta: list<float>: [X,Y,Z] angle
        """
        q = [0,0,0,0]
        th_z = 0
        q[0] = odomData.pose.pose.orientation.x
        q[1] = odomData.pose.pose.orientation.y
        q[2] = odomData.pose.pose.orientation.z
        q[3] = odomData.pose.pose.orientation.w
        theta = tf.transformations.euler_from_quaternion(q)
        th_z = theta[2]
        #th_z_old = th_z
        #print "_convert_quaternion_2D_euler_z: "
        #print "\t angleFlag: ", self.angleFlag
        if th_z < 0.0 and self.angleFlag == False:
                self.angleOffset += 2*math.pi
                self.angleFlag = True
        if th_z >= 0.0:
            self.angleFlag = False
                
        th_z += self.angleOffset
        
        #rospy.loginfo("angleFlag: {}\tOrientation: {}\tOffset:{}\tFinal: {}".format(self.angleFlag, th_z_old, self.angleOffset, th_z))
        return th_z
    
def main():
    drad = DriveRobotAngleDirection(2 * math.pi)

if __name__=='__main__':
    main()

