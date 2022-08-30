#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import ColorRGBA
from matplotlib import cm

class VertexMarker:
    def __init__(self, p, frontierFlag=False, id=-1):
        self.id = id
        # For data association
        # Position of point
        self.p = p

class MarkOdomety:
    def __init__(self):
        # -- initialize node
        rospy.init_node('mark_odometry', anonymous = True)
        self.first_time = True
        self.first_pub = True
        self.pub_rviz = rospy.Publisher('odom_marker', Marker, queue_size = 1)
        self.odom_list = []
        self.oldMarker = None
        self.agent_type = rospy.get_param('agent_type')
        self.color = self.setColor(self.agent_type)
        # -- get current location
        rospy.Subscriber('/odom', Odometry, self.callback_odom)
        self.rate = 1

        # -- spin
        rospy.spin()

    def setColor(self, agent_type):
        if agent_type == "Bob":
            return ColorRGBA(0.7,0.0,0.7,1.0)
        else:
            return ColorRGBA(1.0,0.0,0.0,1.0)

    def callback_odom(self, msg):
        self.current_pose = msg.pose.pose
        self.cur_time = msg.header.stamp.secs
        
        
        # -- we process the data every self.rate seconds. 
        if self.first_time:
            # print "Current Pose: ", self.current_pose.position.x, 
            # print "\t", self.current_pose.position.y
            # print "First_time: ", self.cur_time
            p_x = self.current_pose.position.x
            p_y = self.current_pose.position.y
            p = Point(p_x, p_y, 0)
            self.odom_list.append(p)
            self.publish_to_rviz()
            self.prev_time = self.cur_time
            self.first_time = False
        else:
            if self.cur_time == self.prev_time + self.rate:
                # print "Current Pose: ", self.current_pose.position.x, 
                # print "Previous time: ", self.prev_time
                # print "Current time:", self.cur_time
                # print "Appending to odom list"
                self.odom_list.append(Point(self.current_pose.position.x,
                                            self.current_pose.position.y,
                                            0))
                self.publish_to_rviz()
                self.prev_time = self.cur_time
        return



    def publish_to_rviz(self):
        # marker = self.createColorRvizMarker(self.odom_list)
        if self.first_pub:
            # print "First pub: ", self.first_pub
            marker = self.createRvizMarker(self.odom_list, None)
            self.first_pub = False
        else:
            marker = self.createRvizMarker(self.odom_list, self.oldMarker)
        self.pub_rviz.publish(marker)
        self.oldMarker = marker
        
    def createColorRvizMarker(self, points):
        """ Creates a set of points based on odometry data. 
        """
        # -- set color
        pointSet = points[-256:]
        color = ColorRGBA()
        color.r = 1.0
        color.b = 1.0
        color.a = 1.0
        pointMarker = Marker()
        # print "rospy.Time(0): ", rospy.get_rostime()
        pointMarker.header.frame_id = 'map'
        pointMarker.header.stamp = rospy.get_rostime()
        pointMarker.ns = ''
        # -- type of marker
        pointMarker.id = 0
        pointMarker.type = Marker.SPHERE_LIST
        pointMarker.action = Marker.ADD
        # -- lifetime of marker
        pointMarker.lifetime = rospy.Duration(10.0)
        # -- set position
        for idx, p in enumerate(pointSet):
            pointMarker.points.append(p)
            # -- set scale
            pointMarker.scale.x = 0.12
            pointMarker.scale.y = 0.12
            color.a -= 1/256
            # -- set color
            pointMarker.colors.append(color)

        # -- return the marker
        return pointMarker


    
    def createRvizMarker(self, pointSet, oldMarker = None):
        """ Creates a set of points based on odometry data. 
        """
        # -- set color
        # color = ColorRGBA()
        # color.r = 1.0
        # color.b = 1.0
        # color.a = 1.0
        pointMarker = Marker()
        # print "rospy.Time(0): ", rospy.get_rostime()
        if oldMarker == None:
            pointMarker.header.frame_id = 'map'
            pointMarker.header.stamp = rospy.get_rostime()
            pointMarker.ns = ''
            # -- type of marker
            pointMarker.id = 0
            pointMarker.type = Marker.SPHERE_LIST
            pointMarker.action = Marker.ADD
            # -- lifetime of marker
            pointMarker.lifetime = rospy.Duration(10.0)
            # -- set position
            for p in pointSet:
                pointMarker.points.append(p)
                # -- set scale
                pointMarker.scale.x = 0.12
                pointMarker.scale.y = 0.12
                # -- set color
                pointMarker.colors.append(self.color)

            # -- return the marker
            return pointMarker
        else:
            pointMarker = oldMarker
            pointMarker.header.stamp = rospy.Time(0)
            pointMarker.points.append(pointSet[-1])
            # -- set scale
            pointMarker.scale.x = 0.05
            pointMarker.scale.y = 0.05
            pointMarker.colors.append(self.color)
        return pointMarker
    
if __name__=='__main__':
    mo = MarkOdomety()
