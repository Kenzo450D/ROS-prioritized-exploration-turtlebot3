#!/usr/bin/env python3

import sys
import rospy
import numpy as np
import sys
from turtlebot3_exploration.srv import ConvertMapGraph
from turtlebot3_exploration.msg import Graph, Graph_nodes, Graph_links
from turtlebot3_exploration.msg import VertexMsg, EdgeMsg, GraphFeatures
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from std_msgs.msg import ColorRGBA


'''
Task:
1. Subscribe to topic: odom and map
2. Convert map to graph using the service ConvertMapGraph
3. Plot the graph using markers

Write a function to convert odom coordinate to map coordinate and vice-versa

Marker documentation here: http://wiki.ros.org/rviz/DisplayTypes/Marker

More information on markers are available at 'rosmsg show Marker' from terminal

kenzo@WS:~$ rosmsg show Marker
[visualization_msgs/Marker]:
uint8 ARROW=0
uint8 CUBE=1
uint8 SPHERE=2
uint8 CYLINDER=3
uint8 LINE_STRIP=4
uint8 LINE_LIST=5
uint8 CUBE_LIST=6
uint8 SPHERE_LIST=7
uint8 POINTS=8
uint8 TEXT_VIEW_FACING=9
uint8 MESH_RESOURCE=10
uint8 TRIANGLE_LIST=11
uint8 ADD=0
uint8 MODIFY=0
uint8 DELETE=2
uint8 DELETEALL=3
std_msgs/Header header
    uint32 seq
    time stamp
    string frame_id
string ns
int32 id
int32 type
int32 action
geometry_msgs/Pose pose
    geometry_msgs/Point position
        float64 x
        float64 y
        float64 z
    geometry_msgs/Quaternion orientation
        float64 x
        float64 y
        float64 z
        float64 w
geometry_msgs/Vector3 scale
    float64 x
    float64 y
    float64 z
std_msgs/ColorRGBA color
    float32 r
    float32 g
    float32 b
    float32 a
duration lifetime
bool frame_locked
geometry_msgs/Point[] points
    float64 x
    float64 y
    float64 z
std_msgs/ColorRGBA[] colors
    float32 r
    float32 g
    float32 b
    float32 a
string text
string mesh_resource
bool mesh_use_embedded_materials

'''
# =============================================================================

class VertexMarker:
    def __init__(self, p, frontierFlag=False, id=-1):
        self.id = id
        # For data association
        # Position of point
        self.p = p
        # Mark frontier
        self.frontierFlag = frontierFlag
        # Msg to publish
        self.msg = VertexMsg(p, self.id)


# =============================================================================

# Make an rviz edge list given a list of edge objects
def buildRvizEdgeList(edgePoints):
    edgeMarker = Marker()
    edgeMarker.header.frame_id = 'map'
    edgeMarker.header.stamp = rospy.Time(0)
    edgeMarker.ns = ''
    # type of edge list
    edgeMarker.id = 0
    edgeMarker.type = Marker.POINTS
    edgeMarker.action = Marker.ADD
    # -- Set lifetime
    edgeMarker.lifetime = rospy.Duration(15.0)
    # set up color
    col = ColorRGBA()
    col.r = 1.0
    col.g = 1.0
    col.a = 1.0
    # Add the edge endpoints to list of points
    for p in edgePoints:
        edgeMarker.points.append(p)
        # -- scale of marker
        edgeMarker.scale.x = 0.05
        edgeMarker.scale.y = 0.05
        # edgeMarker.scale.z = 0.05
        # -- color of marker
        edgeMarker.colors.append(col)
    return edgeMarker
    

def buildRvizVertexList(vertices):
    pointMarker = Marker()
    #TB3
    pointMarker.header.frame_id = 'map'
    pointMarker.header.stamp = rospy.Time(0)
    pointMarker.ns = ''
    # Id and type of marker
    pointMarker.id = 1 # unique id for each elements
    pointMarker.type = Marker.POINTS # 8 - points, 9 - text_view_facing
    pointMarker.action = Marker.ADD # 0 means add
    # Set up colors for frontier and non-frontier node
    frontier_col = ColorRGBA()  # White
    frontier_col.r = 1.0
    frontier_col.g = 1.0
    frontier_col.b = 1.0
    frontier_col.a = 1.0
    non_frontier_col = ColorRGBA() # Blue
    non_frontier_col.b = 0.0
    non_frontier_col.a = 1.0
    # -- set lifetime
    pointMarker.lifetime = rospy.Duration(15.0)
    for v_idx, c in enumerate(vertices):
        pointMarker.points.append(c.p)
        # -- scale of marker
        pointMarker.scale.x = 0.2
        pointMarker.scale.y = 0.2
        pointMarker.scale.z = 0.2
        # -- color of marker
        if c.frontierFlag == True:
            pointMarker.colors.append(frontier_col)
        else:
            pointMarker.colors.append(non_frontier_col)
    return pointMarker


def buildRvizVertexText(vertices, init_idx = 2):
    """ Creates a Marker array for vertex ids at each position. 
    """
    # -- debug print
    print ("Number of vertices: {}".format(len(vertices)))
    markerArray = MarkerArray()
    t = rospy.Time(0)
    # -- initialize color for the text
    col = ColorRGBA()
    col.r = 1.0
    col.g = 0.0
    col.b = 0.0
    col.a = 1.0
    # we consider the id for the vertices to be starting from 2 as the 
    # previous ones are taken by vertex points and line marks.
    for idx in range(0, len(vertices)):
        m = Marker()
        # -- header of marker
        m.header.frame_id = 'map'
        m.header.stamp = t
        m.ns = ''
        # -- id and type of marker
        m.id = init_idx + idx
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        # -- set text and position
        m.text = str(vertices[idx].id)
        m.pose.position = vertices[idx].p
        # -- set scale
        # m.scale.x = 0.2
        # m.scale.y = 0.2
        m.scale.z = 0.2
        # -- set color
        m.color = col
        # -- set lifetime
        m.lifetime = rospy.Duration(15.0)
        # -- add marker to markerArray
        markerArray.markers.append(m)
    return markerArray

# =============================================================================

class PlotRviz:
    def __init__(self, map_topic):
        # -- initialize node
        rospy.wait_for_service('convert_map_to_graph')

        # -- check if parameter exists
        if not rospy.has_param('flag_publish_nodes_rviz'):
            rospy.set_param('flag_publish_nodes_rviz', True)
        
        # -- initialize the service
        self.convert_map_graph_service = rospy.ServiceProxy('convert_map_to_graph', ConvertMapGraph)
        
        # -- initialize the publisher
        # For vertices and edges
        self.pub_rviz = rospy.Publisher('visualization_marker', Marker, queue_size=1)
        # For vertex labels
        self.pub_rviz_array = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=1)
        
        # -- subscribe to map_topic
        rospy.Subscriber(map_topic, OccupancyGrid, self.callback_map)
        # -- spin
        rospy.spin()

    def convert_map_graph_client(self, occgrid_msg):
            
        try:
            # -- convert the occupancy grid to a graph
            resp_graph = self.convert_map_graph_service(occgrid_msg)
            #rospy.loginfo(resp_graph)
            return resp_graph
        
        except rospy.ServiceException as e:
            print ("Service call failed: %s"%e)

    def callback_map(self, msg):
        # callback function cannot return anything, so we use a global variable to store the data
        # https://stackoverflow.com/questions/37373211/update-the-global-variable-in-rospy
        # https://answers.ros.org/question/174485/return-a-value-from-a-callback-function/
        #print ("In subscriber callback function")
        occgrid_msg = msg
        
        # -- get resolution from info of map message
        # get the coordinate of the origin
        # get the final coordinates
        self.resolution = occgrid_msg.info.resolution
        self.origin_x = occgrid_msg.info.origin.position.x
        self.origin_y = occgrid_msg.info.origin.position.y
        
        # -- print the origin
        #print ("Origin: x: ", self.origin_x, "\ty: ", self.origin_y)
        
        # -- get publish parameter
        pub_flag = rospy.get_param('flag_publish_nodes_rviz')
        rospy.loginfo("Publish Flag: "+str(pub_flag))
        if pub_flag: # if do not publish, do not call service to convert to graph
            # -- get the graph
            t1 = rospy.get_time()
            self.graph = self.convert_map_graph_client(occgrid_msg)
            t2 = rospy.get_time()
            rospy.loginfo("Time to convert: "+ str(t2 - t1))
        # -- plot the graph
        self._plot_graph_rviz(self.graph)
        
        # raw_input("press enter to continue...")
        return
    
    def _plot_graph_rviz(self, graph_msg):
        
        # -- get the graph_nodes from the graph
        g_nodes = graph_msg.g.nodes
        g_edges = graph_msg.g.links
        
        # -- form vertex msg objects for rviz plot
        vertex_list = []
        vertex_dict = {}
        for node in g_nodes:
            odom_point = self._map_to_odom_coordinate([node.pts[0], node.pts[1]])
            p_a = Point(odom_point[0], odom_point[1], 0)
            frontierFlag = True if node.priority > 0 else False
            vm = VertexMarker(p_a, frontierFlag, node.id)
            vertex_list.append(vm)
            vertex_dict[node.id] = p_a
            #rospy.loginfo("Node: "+str(node.id)+ " Frontier: "+str(frontierFlag))
        
        # -- form edge msg objects for rviz plot
        edge_list = []
        for edge in g_edges:
            source_id = edge.source
            target_id = edge.target
            points = edge.pts
            points = np.array(points)
            points = points.reshape((points.shape[0]//2,2))
            em = []
            for i in points:
                odom_point = self._map_to_odom_coordinate([i[0],i[1]])
                em.append(Point(odom_point[0], odom_point[1], 0))
            edge_list.extend(em)
        
        # -- publish the marker object 
        m_edges = buildRvizEdgeList(edge_list)
        m_vertices = buildRvizVertexList(vertex_list)
        m_vertex_labels = buildRvizVertexText(vertex_list)
        self.pub_rviz.publish(m_edges)
        self.pub_rviz.publish(m_vertices)
        self.pub_rviz_array.publish(m_vertex_labels)
        
    
    def _map_to_odom_coordinate(self, map_coord):
        
        odom_x = map_coord[0] * self.resolution + self.origin_x
        odom_y = map_coord[1] * self.resolution + self.origin_y
        
        #print ("map Coord: ", map_coord, "\t\t Odom: (", odom_x, odom_y, ")")
        # -- return the odometry coordinate
        # swapping them as x is along the columns and y is along the rows
        return (odom_y, odom_x)
        

    
if __name__ == "__main__":
    rospy.init_node('rviz_graph')
    pr = PlotRviz("/map")
    rospy.set_param('flag_publish_nodes_rviz', True)

"""
kenzo@WS:src$ rostopic echo -n1 /odom
header: 
    seq: 4417
    stamp: 
        secs: 147
        nsecs: 267000000
    frame_id: "odom"
child_frame_id: "base_footprint"
pose: 
    pose: 
        position: 
            x: -2.99997403079
            y: 1.00008592213
            z: -0.00100739917419
        orientation: 
            x: -2.08957574606e-06
            y: 0.00158964999038
            z: 0.000269225186039
            w: 0.999998700262
  covariance: [1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1000000000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001]
twist: 
    twist: 
        linear: 
            x: 6.53500601163e-07
            y: 1.22026072947e-06
            z: 0.0
        angular: 
            x: 0.0
            y: 0.0
            z: 7.90277115773e-06
    covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
---
kenzo@WS:src$ rostopic echo -n1 /map
header: 
    seq: 55
    stamp: 
        secs: 156
        nsecs: 723000000
    frame_id: "map"                                                                                                             
info:                                                                                                                         
    map_load_time:                                                                                                              
        secs: 0                                                                                                                   
        nsecs:         0                                                                                                          
    resolution: 0.0500000007451                                                                                                 
    width: 384                                                                                                                  
    height: 384                                                                                                                 
    origin:                                                                                                                     
        position:                                                                                                                 
            x: -10.0                                                                                                                
            y: -10.0                                                                                                                
            z: 0.0                                                                                                                  
        orientation:                                                                                                              
            x: 0.0                                                                                                                  
            y: 0.0                                                                                                                  
            z: 0.0                                                                                                                  
            w: 1.0                                                                                                                  
data: [-1, ..., -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
---
kenzo@WS:src$ 
"""
