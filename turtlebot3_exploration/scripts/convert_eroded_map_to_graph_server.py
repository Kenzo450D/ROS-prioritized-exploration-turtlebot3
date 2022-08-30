#!/usr/bin/env python3

#from __future__ import print_function

import sys
import rospy
import numpy as np
import sys
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize
np.set_printoptions(threshold=np.inf)
from turtlebot3_exploration.srv import ConvertMapGraph, ConvertMapGraphResponse
from turtlebot3_exploration.msg import Graph, Graph_nodes, Graph_links
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
import cv2
import math
from std_msgs.msg import Int64

# from laser_emulation import LaserFeature
from bob_node_priority import assign_frontier_priority_lidar_emulate
from bob_node_priority import assign_frontier_priority_lidar_emulate_threshold
from bob_node_priority import assign_frontier_priority
from alice_node_priority import assign_frontier_alice

"""
Here is the order of functions being called in this file:
1.  convert_map_graph_server()
2.  handle_graph_conversion()
3.  convert_to_binary()
4.  data_skeleton()
5.  run_skeleton_input()g
6.  convert_to_message()
7.  convert_node_messages()
8.  assign_frontier_priority()
"""

# -- Initial Nodes
agent_name = ""
origin_x = 0
origin_y = 0

# -----------------------------------------------------------------------------
# -- convert the uint8 map image to a binary map
# -----------------------------------------------------------------------------
def convert_to_binary(map_array):
    map_binary = np.zeros(np.shape(map_array), dtype=bool)
    map_binary[np.where(map_array == 255)] = True
    return map_binary


# -----------------------------------------------------------------------------
# -- convert the map to a 2 value np.uin8 map
# -----------------------------------------------------------------------------
def convert_to_two_color(map_array):
    map_binary = np.zeros(np.shape(map_array), dtype=np.uint8)
    map_binary[np.where(map_array == 0)] = 255
    return map_binary


# -----------------------------------------------------------------------------
# -- erode the image to make empty spaces smaller
# -----------------------------------------------------------------------------
def erode_image(map_array, filter_size = 9):
    """ Erodes the image to reduce the chances of robot colliding with the wall
    each pixel is 0.05 meter. The robot is 30 cm wide, that is 0.3 m. Half is
    0.15 m. If we increase the walls by 20 cm on either side, the kernel should
    be 40 cm wide. 0.4 / 0.05 = 8
    """
    kernel = np.ones((filter_size,filter_size), np.uint8)
    eroded_img = cv2.erode(map_array, kernel, iterations = 1)
    return eroded_img


# -----------------------------------------------------------------------------
# -- calculate euclidean distance
# -----------------------------------------------------------------------------
def euclidean_dist(coord_1, coord_2):
        """ Get distance between the current robot pose and a given 2D 
        coordinate in the map.
        """
        dist = math.sqrt(float(coord_1[0] - coord_2[0]) ** 2 + 
                         float(coord_1[1] - coord_2[1]) ** 2)
        return dist

# -----------------------------------------------------------------------------
# -- skeletonize the binary map
# -----------------------------------------------------------------------------    
def data_skeleton(data_binary):
    skeleton = skeletonize(data_binary)
    return skeleton

def neighbors(shape):
    dim = len(shape)
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

def mark(img, nbs): # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        if img[p]==0:continue
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2
        

def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    rst -= 1
    return rst
    
def fill(img, p, num, nbs, acc, buf):
    back = img[p]
    img[p] = num
    buf[0] = p
    cur = 0; s = 1
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==back:
                img[cp] = num
                buf[s] = cp
                s+=1
        cur += 1
        if cur==s:break
    return idx2rc(buf[:s], acc)

def trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, idx2rc(buf[:cur+1], acc))
   
def parse_struc(img, pts, nbs, acc):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    #sys.exit(0)
    num = 10
    nodes = []
    for p in pts:
        if img[p] == 2:
            nds = fill(img, p, num, nbs, acc, buf)
            num += 1
            nodes.append(nds)
    edges = []
    for p in pts:
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges
    
# use nodes and edges build a networkx graph
def build_graph(nodes, edges, multi=False):
    graph = nx.MultiGraph() if multi else nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=nodes[i].mean(axis=0))
    for s,e,pts in edges:
        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s,e, pts=pts, weight=l)
    return graph

def buffer(ske):
    buf = np.zeros(tuple(np.array(ske.shape)+2), dtype=np.uint16)
    buf[tuple([slice(1,-1)]*buf.ndim)] = ske
    return buf

def mark_node(ske):
    buf = buffer(ske)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    return buf
    
def build_sknw(ske, multi=False):
    buf = buffer(ske)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    pts = np.array(np.where(buf.ravel()==2))[0]
    nodes, edges = parse_struc(buf, pts, nbs, acc)
    return build_graph(nodes, edges, multi)
    
# -----------------------------------------------------------------------------
# -- get graph from skeleton input
# -----------------------------------------------------------------------------  
def run_skeleton_input(ske):
    # -- convert skeleton to binary skeleton
    if len(np.unique(ske)) > 2:
        # -- skeleton is not binary
        ske[np.where(ske > 0)] = 1
        ske[np.where(ske < 1)] = 0
    
    # -- generate the graph
    #node_img = mark_node(ske)
    graph = build_sknw(ske)
    
    # -- return the graph
    return graph

# -----------------------------------------------------------------------------
# -- convert nodes to node message
# ----------------------------------------------------------------------------- 
def convert_nodes_messages(g):
    # -- get all the nodes of the graph
    nodes = g.nodes.items()
    
    # -- declare a message of node
    # taken from webpage
    # http://wiki.ros.org/ROS/Tutorials/CustomMessagePublisherSubscriber%28python%29
    graph_nodes = []
    for node in nodes:
        node_msg = Graph_nodes()
        node_msg.id = int(node[0])
        node_msg.pts = list(node[1]['pts'].flatten())
        node_msg.nPoints = node[1]['pts'].shape[0]
        node_msg.o = list(node[1]['o'])
        graph_nodes.append(node_msg)
        # -- initialize priority to 0
        node_msg.priority = 0 # frontier nodes would be marked as 1 later
    return graph_nodes


# -----------------------------------------------------------------------------
# -- convert edges to links message
# ----------------------------------------------------------------------------- 
def convert_links_messages(g):
    graph_edges = []
    # -- get all the links of the graph
    for i, e in g.edges.items():
        # -- initialize edge
        edge_msg = Graph_links()
        edge_msg.source = i[0]
        edge_msg.target = i[1]
        edge_msg.nPoints = e['pts'].shape[0]
        edge_msg.weight = e['weight']
        # -- check if the edge pts are flipped
        edge_pts = e['pts']
        edge_pt1 = e['pts'][0,:]
        edge_pt2 = e['pts'][-1,:]
        v_source_o = g.nodes[i[0]]['o']
        if euclidean_dist(v_source_o, edge_pt1) > euclidean_dist(v_source_o, edge_pt2):
            edge_pts = np.flip(edge_pts, 0)
        edge_msg.pts = list(edge_pts.flatten())
        # -- append the edges
        graph_edges.append(edge_msg)
        
    return graph_edges

# -----------------------------------------------------------------------------
# -- convert graph to graph message
# ----------------------------------------------------------------------------- 
def convert_to_message(g, map_data_array):
    
    # format of message of graph
    
    #beginner_tutorials/Graph g
    #beginner_tutorials/Graph_nodes[] nodes
        #int64 id
        #int64 pts
        #int64 nPoints
        #float64[] o
    #beginner_tutorials/Graph_links[] links
        #int64 source
        #int64 target
        #float64 weight
        #int64 nPoints
        #int64[] pts
    #bool directed
    #bool multigraph
    
    # -------------------------------------------------------------------------
    # -- convert the nodes
    # -------------------------------------------------------------------------
    graph_nodes = convert_nodes_messages(g)
    print ("Graph nodes calculated")
    print (graph_nodes)
    print("-"*100)
    # -- mark the priority of graph_nodes to 1 if a frontier node
    if agent_name == "Bob":
        graph_nodes = assign_frontier_priority_lidar_emulate_threshold(graph_nodes, 
                                                            map_data_array)
        
    elif agent_name == "Alice":
        graph_nodes = assign_frontier_alice(graph_nodes,
                                            map_data_array)
        
    #graph_nodes = assign_frontier_priority(graph_nodes, map_data_array)
    
    # -------------------------------------------------------------------------
    # -- convert the links
    # -------------------------------------------------------------------------
    graph_edges = convert_links_messages(g)
    
    # -------------------------------------------------------------------------
    # -- create the graph message
    # -------------------------------------------------------------------------
    graph_msg = Graph()
    graph_msg.nodes = graph_nodes
    graph_msg.links = graph_edges
    graph_msg.directed = g.is_directed()
    graph_msg.multigraph = g.is_multigraph()
    
    # -- return the message
    return graph_msg
    


def handle_graph_conversion(msg):
    # -------------------------------------------------------------------------
    # -- get the occupancy grid map from the input
    # -------------------------------------------------------------------------
    # occupancy graph format
    
    #kenzo@WS:~/catkin_ws$ rosmsg show nav_msgs/OccupancyGrid 
    #std_msgs/Header header
    #uint32 seq
    #time stamp
    #string frame_id
    #nav_msgs/MapMetaData info
    #time map_load_time
    #float32 resolution
    #uint32 width
    #uint32 height
    #geometry_msgs/Pose origin
        #geometry_msgs/Point position
        #float64 x
        #float64 y
        #float64 z
        #geometry_msgs/Quaternion orientation
        #float64 x
        #float64 y
        #float64 z
        #float64 w
    #int8[] data
    #
    #kenzo@WS:~/catkin_ws$
    
    # print ("Service: Entered Service: Agent: ")
    rospy.loginfo("Service: Entered Service: Agent: {}".format(agent_name))
    #print ("Type msg: ", type(msg))
    res = msg.map.info.resolution
    map_width = msg.map.info.width
    map_height = msg.map.info.height
    global origin_x, origin_y
    origin_x = msg.map.info.origin.position.x
    origin_y = msg.map.info.origin.position.y
    print ("origin_x:", origin_x)
    print ("origin_y:", origin_y)
    map_data = msg.map.data
    map_data_array = np.array(map_data)
    map_data_array = map_data_array.reshape(map_height, map_width)
    print ("Resolution: ", res)
    print ("Map height: ", map_height)
    print ("Map width: ", map_width)
    # -------------------------------------------------------------------------
    # -- send the map_data_array to the function to convert it to a graph
    # -------------------------------------------------------------------------
   
    # convert to uint8 image
    map_data_uint8 = convert_to_two_color(map_data_array)

    # erode image
    eroded_map_data_uint8 = erode_image(map_data_uint8)

    # convert to binary
    binary_map_data = convert_to_binary(eroded_map_data_uint8)
    
    # convert binary map to skeleton
    ske_map = data_skeleton(binary_map_data)
    
    # convert skeleton map to graph
    graph = run_skeleton_input(ske_map)
    
    # convert to message format
    graph_msg = convert_to_message(graph, map_data_array)
    
    # -------------------------------------------------------------------------
    # -- debug print
    # -------------------------------------------------------------------------
    print ("Converted the occupancy grid to a graph")
    
    # -- convert the map to the graph
    return ConvertMapGraphResponse(graph_msg)

def convert_map_graph_server():
    rospy.init_node('convert_map_to_graph_server')
    # Format rospy.Service(name, service_class, handler, buff_size=65536)
    s = rospy.Service('convert_map_to_graph', ConvertMapGraph, handle_graph_conversion)
    print ("Started service to convert map to graph")
    rospy.spin()
    
if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "Alice":
            print ("Priority of nodes based on Alice")
            agent_name = "Alice"
        elif sys.argv[1] == "Bob":
            print ("Priority of nodes based on Bob")
            agent_name = "Bob"
    else:
        print ("#"*100)
        print ("Default node set to Bob")
        print ("#"*100)
        agent_name = "Bob"
    convert_map_graph_server()
