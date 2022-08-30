#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry 
from geometry_msgs.msg import Twist
import tf.transformations
from geometry_msgs.msg import Point
import math
import numpy as np


"""
Improvements to be made:
1. The robot should stop once it cannot move from a single place for more than 2 seconds. 
2. Robot takes in a path instead of just a goal point.

The robot should keep a track of its position for at least 2 minutes. If the robot is stuck, 
for at least 1 minute, the robot should understand that is stuck, and it can get out of there. 
To get out of there, it can just travel backwards.

"""

class DriveRobot:
    def __init__(self,
                 goalPose, #Point
                 goalTheta = None, #goal radians (along Z axis)
                 ang_vel = 0.05, #angular velocity
                 lin_vel = 0.1, #linear velocity
                 error_angle=0.03, # approx 2 degrees # 0.0174533, # one degree
                 error_dist=0.05):
        # -- initialize global variables
        self.flag_odom_data = False
        self.goalPose = goalPose #Point object
        self.goalTheta = goalTheta #radians
        self.ang_vel = ang_vel
        self.lin_vel = lin_vel
        self.error_angle = error_angle
        self.error_dist = error_dist # to check if the distance moved is too little
        self.time_limit = 5.0 #seconds
        self.dist_threshold = 0.001

        # print "Drive Robot: Angular Velocity: {}".format(ang_vel)
        
        # -- initalize needed global variables
        self.odomData = rospy.wait_for_message('odom', Odometry, None)
        self.twistData = Twist()
        self.angleOffset = 0 #to handle negative orientation values
        self.angleFlag = False
        # -- initialize odom subscriber to get odometry data
        rospy.Subscriber('odom', Odometry, self.odomCB)
        
        # -- initalize publisher and rate to move the robot
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
        self.pub_rate = rospy.Rate(30)
        
        # -- function to direct robot to goal
        self.flag_robot_stuck = False
        self.moveToGoal()
        
        # -- Turn to a particular angle
        #self.turnTo(turn_angle, self.pub)
        # -- Drive up to the distance
        #self.driveDistance(move_dist, self.pub)
        


    # --------------------------------------------------------------------------
    # Function: Odometry callback
    # --------------------------------------------------------------------------
    def odomCB(self, data):
        self.odomData = data
        self.flag_odom_data = True

    
    def moveToGoal(self, debug_print=False):
        """ Moves robot to the goal position as specified by goalPose and 
        goalTheta
        Return: True if the movement was successful
                False if the robot got stuck
        Step:
        1.  Calculate orientation between current point and goal point.
        2.  Calculate distance and move towards goal point.
        3.  Repeat Step 1 and 2 as necessary till goal point is reached.
        4.  Once these two steps are complete, get the current orientation again
            and turn the robot back to the goal
        """
        # -- step 1 to 3
        cur_pose = self.odomData.pose.pose.position
        if debug_print:
            print ("Current Pose: {} {}".format(np.round(cur_pose.x,2), np.round(cur_pose.y, 2)))
            print ("Goal Pose: {} {}".format(np.round(self.goalPose.x,2), np.round(self.goalPose.y,2)))
        self.driveToPoint(self.goalPose, self.pub)
        # -- check if robot is stuck
        if self.flag_robot_stuck:
            print ("Robot is stuck")
            return False
        # -- Step 4
        if self.goalTheta is not None:
            cur_theta = self._convert_quaternion_2D_euler(self.odomData)
            # cur_pose = self.odomData.pose.pose.position
            goal_theta = self.goalTheta
            angle_diff = goal_theta - cur_theta
            clockwiseTurn = False if angle_diff > 0 else True
            # rospy.loginfo("Current theta: {}".format(cur_theta))
            # rospy.loginfo("Goal theta: {}".format(goal_theta))
            # rospy.loginfo("Angle Difference: {}".format(angle_diff))
            # rospy.loginfo("Clockwise Turn: {}".format(clockwiseTurn))
            if angle_diff > 0:
                self.turnTo(goal_theta, self.pub, False)
            else:
                self.turnTo(goal_theta, self.pub, True)
        return True


    # --------------------------------------------------------------------------
    # Function: Move to goal position with feedback control
    # --------------------------------------------------------------------------
    def driveToPoint(self, goalPose, publisher, debugPrint = False):
        """ Uses a while loop to detect if the orientation is correct, if
        correct then move towards the goal point.
        """
        # -- debug print
        # print "Goal Node: ", goalPose
        # -- initialize the publisher and parameters
        pub = publisher
        init_pose = self.odomData.pose.pose.position
        dx = goalPose.x - init_pose.x
        dy = goalPose.y - init_pose.y
        init_dist_diff = math.sqrt(dy**2 + dx **2)
        prev_x = init_pose.x
        prev_y = init_pose.y
        flag_first_run = True
        # -- while robot has not reached 
        init_stuck_time = 0.0
        flag_robot_stuck = False
        while not rospy.is_shutdown():
            cur_pose = self.odomData.pose.pose.position
            
            # -- fix distance
            dx = goalPose.x - cur_pose.x
            dy = goalPose.y - cur_pose.y
            dist_diff = math.sqrt(dy**2 + dx **2)
            dmx = cur_pose.x - init_pose.x
            dmy = cur_pose.y - init_pose.y
            dist_moved = math.sqrt(dmx ** 2 + dmy ** 2)
            # if debugPrint:
            #     print "\rCurrent Pose: {} {}".format(np.round(cur_pose.x,3), np.round(cur_pose.y,3)),
            # if debugPrint:
                # print "Moved: {}\tDifference: {}".format(dist_moved, dist_diff)
            # to check if robot is within threshold distance of the goal point
            if dist_diff < self.error_dist or dist_moved > (init_dist_diff+0.5):
                """ Checks if distance difference is less than the error distance
                and the dist moved is more than the initial distance (i.e. robot 
                moving backwards).
                """
                #print "Break as the robot is at location or robot moved away from target"
                break
            # -- get angle and check if accurate
            cur_theta = self._convert_quaternion_2D_euler(self.odomData)
            goal_theta = math.atan2((self.goalPose.y - cur_pose.y),
                                    (self.goalPose.x - cur_pose.x))
            cur_theta = cur_theta if cur_theta > 0 else cur_theta + 2 * math.pi
            goal_theta =  goal_theta if goal_theta > 0 else goal_theta + 2*math.pi
            angle_diff = goal_theta - cur_theta
            if angle_diff < -math.pi:
                angle_diff += 2*math.pi
            # -- Fix angle
            if abs(angle_diff) > 2 * self.error_angle: # TODO: Reduce multiplier if unncessary
                angle_diff = goal_theta - cur_theta
                # robot increases angle as 
                if angle_diff > 0:
                    if angle_diff > math.pi:
                        clockwiseTurn = True
                    else:
                        clockwiseTurn = False
                else:
                    if abs(angle_diff) > math.pi:
                        clockwiseTurn = False
                    else:
                        clockwiseTurn = True
                # clockwiseTurn = False if angle_diff > 0 else True
                self.turnTo(goal_theta, publisher, clockwiseTurn)
            
            # print "Travel straight with velocity: ", self.lin_vel
            self.twistData.linear.x = self.lin_vel
            pub.publish(self.twistData)
            flag_first_run = False
            # if debugPrint:
                # print "-"*100
            
        ##once we break out of the loop, stop moving because we have reached the desired point
        self.twistData.linear.x = 0.0
        self.twistData.angular.z = 0.0
        pub.publish(self.twistData)
        self.flag_robot_stuck = flag_robot_stuck
        # -- return
        return
            
    
    # --------------------------------------------------------------------------
    # Function: Move along a straight line towards a goal position
    # --------------------------------------------------------------------------
    def driveDistance(self, goalPose, publisher):##take in a specified distance and a publisher to move the robot
        ##needed global variables
        pub = publisher
        cur_pose = self.odomData.pose.pose.position
        dx = goalPose.x - cur_pose.x
        dy = goalPose.y - cur_pose.y
        init_dist_diff = math.sqrt(dy**2 + dx **2)
        init_pose = cur_pose
        # -- while robot has not reached 
        while True and not rospy.is_shutdown():
            cur_pose = self.odomData.pose.pose.position
            dx = goalPose.x - cur_pose.x
            dy = goalPose.y - cur_pose.y
            dist_diff = math.sqrt(dy**2 + dx **2)
            dmx = cur_pose.x - init_pose.x
            dmy = cur_pose.y - init_pose.y
            dist_moved = math.sqrt(dmx ** 2 + dmy ** 2)
            # rospy.loginfo("Moved: {}\tDifference: {}".format(dist_moved, dist_diff))
            ##set the linear.x velocity and publish to move the robot
            # if dist_diff < self.error_dist:
            #     self.flag_robot_stuck = True
            #     break
            # if dist_moved > (init_dist_diff+0.5):
            #     self.flag_robot_slipping = True
            #     break
            
            if dist_diff < self.error_dist or dist_moved > (init_dist_diff+0.5):
                # to check if robot is within threshold distance of the goal point
                break
            self.twistData.linear.x = self.lin_vel
            pub.publish(self.twistData)
        ##once we break out of the loop, stop moving because we have reached the desired point
        self.twistData.linear.x = 0.0
        pub.publish(self.twistData)
        

    # --------------------------------------------------------------------------
    # Function: Turn to a particlular goal theta
    # --------------------------------------------------------------------------
    def turnTo(self, goal_theta, publisher, clockwise = True, debugPrint=False):
        """ take in a specified orientation and a publisher to move the robot
        Args:
            goal_theta: goal orientation in terms of 
            publisher: rospy publisher to publish the twist data on
        Returns:
            publishes the twist to publisher
        """
        # Ensure that goal_theta is not a negative value
        goal_theta = goal_theta if goal_theta>=0 else goal_theta + 2*math.pi
        pub = publisher
        q = [0,0,0,0] #IDK
        arrived = False
        # -- upperBound and lowerBound to make sure we stop within the correct 
        #    orientation values
        upperBound = goal_theta + self.error_angle / 2 
        lowerBound = goal_theta - self.error_angle / 2
        self.twistData.linear.x = 0.0
        # -- get local angular velocity
        ang_vel = self.ang_vel

        # -- save the orientation first
        prev_clockwise = clockwise
        first_run = True
        prev_abs_delta_theta = True
        # -- while arrived is false
        while not arrived and not rospy.is_shutdown():
            ##save odomData.orientation into a list for the conversion function
            cur_theta = self._convert_quaternion_2D_euler(self.odomData)
            cur_theta = cur_theta if cur_theta >= 0 else cur_theta + 2*math.pi
            # -- if theta has reached the desired orientation set arrived to true
            if cur_theta <= upperBound and cur_theta >= lowerBound:
                arrived = True
                if debugPrint:
                    print ("Arrived!")
                    print ("*"*50)
                break
            # -- calculate theta difference, based on that, use direction
            delta_theta = goal_theta - cur_theta
            if debugPrint:
                print ("Cur Theta: {}\tGoal_theta: {}\tDelta: {}\tClockwise: {}".format(np.round(cur_theta,3), 
                                                                                         np.round(goal_theta,3),
                                                                                         np.round(delta_theta,3), 
                                                                                         clockwise))

            # -- keep moving until about if statement has been satisfied. 
            self.twistData.angular.z = -ang_vel if clockwise else ang_vel
            pub.publish(self.twistData)
            # -- sleep
            self.pub_rate.sleep()
        #once out of loop, stop moving
        self.twistData.angular.z = 0.0
        pub.publish(self.twistData)
        return

    def _convert_quaternion_2D_euler(self, odomData):
        """ convert a quaternion to a euler (XYZ) angle
        Args:
            odomData: odometry data of the robot (/odom topic)
        Return:
            theta: list<float>: [X,Y,Z] angle
        """
        q = [0,0,0,0]
        q[0] = odomData.pose.pose.orientation.x
        q[1] = odomData.pose.pose.orientation.y
        q[2] = odomData.pose.pose.orientation.z
        q[3] = odomData.pose.pose.orientation.w
        theta = tf.transformations.euler_from_quaternion(q)
        return theta[2]
    


def main():
    """ Test the DriveRobot class by moving the robot a sample distance
    
    Example environment: turtlebot house gazebo environment.
        Goalpoint: X: 0.0 Y: 0.3, Orientation: Euler angle
        Starting location: X: -3.0, Y: 1.0
    ----------------------------------------------------------------------------
    Step 1: Find orientation the robot needs to be in.
    Step 2: Make robot orientation to that one.
    Step 3: Move robot forward based on the L2 norm distance.
    Step 4: Orient the robot in final required orientation.
    ----------------------------------------------------------------------------
    The +ve angular velocity refers to anti-clockwise rotation of the robot.
    """
    # -- initialize node
    rospy.init_node('drive_robot', anonymous=False)
    goalPose = Point(0.0, 0.3, 0.0)
    goalTheta = 0.0
    dr = DriveRobot(goalPose, goalTheta)
    
    
    
if __name__=='__main__':
    main()

