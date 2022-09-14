#!/usr/bin/env python3

import rospy 
from std_msgs.msg import Float64

class ExplorationTime:
    def __init__(self):
        # -- initialize node
        rospy.init_node('exploration_time',  anonymous=True)
        # -- get current time
        self.init_time= rospy.get_time()
        # get time often returns the time to be zero. 
        #The while loop ensures that the init_time is not a garbage number
        while (self.init_time < 1):
            self.init_time= rospy.get_time()
            #rospy.loginfo("Init Time: {}".format(self.init_time))
        
        self.pub = rospy.Publisher('time_current', Float64, queue_size=10) # USE THIS
        # the next publisher so that I can take screenshots at the right time
        self.pub_unknwn = rospy.Publisher('time_', Float64, queue_size=10) # DO NOT USE THIS
        self.rate = rospy.Rate(10) 
        # -- debug print
        t_bd = float(rospy.get_param('time_before_imposed_deadline'))
        t_ad = float(rospy.get_param('time_after_imposed_deadline'))
        rospy.loginfo("Time before imposed deadline %f", t_bd)
        rospy.loginfo("Time after imposed deadline %f", t_ad)
        # -- run time publisher
        self.time_publisher()

    def time_publisher(self):
        t_bd = float(rospy.get_param('time_before_imposed_deadline'))
        t_ad = float(rospy.get_param('time_after_imposed_deadline'))
        # -- get parameters 
        while not rospy.is_shutdown():
            cur_time = rospy.get_time() - self.init_time
            rem_time = t_bd + t_ad - cur_time
            if cur_time < t_bd:
                self.pub.publish(float("inf")) 
                self.pub_unknwn.publish(cur_time) # FOR DEBUGGING ONLY
            
            else:
                self.pub.publish(rem_time)
                self.pub_unknwn.publish(cur_time)# FOR DEBUGGING ONLY
            
            # -- debug print
            #rospy.loginfo("Remaining Time: %f\t Current time %f", rem_time, cur_time)
            
            # -- sleep till next message
            self.rate.sleep()
        

    def print_rospy_time(self):
        now = rospy.get_rostime()
        rospy.loginfo("Current time %i %i", now.secs, now.nsecs)
    
def main():
    # -- set up parameters
    rospy.set_param('time_before_imposed_deadline', 0)
    rospy.set_param('time_after_imposed_deadline', 1000)
    
    # -- make exploration Time Class object
    wt = ExplorationTime()


if __name__ == '__main__':
    main()
