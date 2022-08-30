#!/usr/bin/env python3

# -- python imports
import numpy as np
import sys
import heapq
import math
from datetime import datetime
from collections import defaultdict
from queue import PriorityQueue
import copy

# -- ROS imports
# ---- ROS framework
import rospy

# ---- ROS messages
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

# -- local imports
from int_wrapper import Wrapper
from drive_robot_360_no_head import DriveRobotAngleDirection
from drive_robot_time_limit import DriveRobot
from bresenham_line import check_obstacles
from get_exploration_metrics import ExplorationMetrics
from helper_functions import *
from helper_functions import _get_cost_to_all_vertices

# ---- ROS services
from turtlebot3_exploration.srv import ConvertMapGraph


class PriorityExplorationAgent:
    def __init__(self, exploration_time_topic, occ_map_topic, avg_speed_topic, turn360Flag=True):
        # ---- Input variables ----
        #   exploration_time_topic: Topic on which the whistle is published.
        # -------------------------
        # -- initialize node
        rospy.init_node("exploration_agent", anonymous=False)

        # -- set the time limits
        # use a subscriber to get the current time limit
        # the rostopic in file whistle_time publishes the time remaining.
        # if the topic gets a inf value, then the time limit is unknown, else 
        # it is known
        self._initialize_parameters(exploration_time_topic, occ_map_topic, avg_speed_topic)

        print("Parameters initialized!")

        # -- turn robot 360 degrees at position to get a complete scan
        if turn360Flag == True:
            drad = DriveRobotAngleDirection(2 * math.pi, 0.1, 0.02)

        if not rospy.has_param('flag_publish_nodes_rviz'):
            rospy.set_param('flag_publish_nodes_rviz', True)

        rospy.set_param('robot_moving', False)  # set that the robot is not moving

        self.agent_type = rospy.get_param('agent_type')  # get the agent type from ros parameter
        # possible agent types are: Alice, Bob
        # TODO: is this even necessary, just follow the priority values greedily?

        # -- set up flags for exploration or moving robot
        self.flag_find_next_goal = True
        self.exception_locations = []  # to save the regions that cause problems during exploration
        self._prioritized_exploration()
        print("Exploration ended successfully!")

        # -- get exploration metrics
        em = ExplorationMetrics(self.occgrid_msg)
        # -- print exploration metrics
        print(f"Label count: {em.label_count}")
        print(f"Label percent explored: {em.labels_percent_explored}")
        print(f"Total percent explored: {em.percent_explored}")

    # --------------------------------------------------------------------------
    # -- Function: intiialize subscribers and time parameters
    # --------------------------------------------------------------------------
    def _initialize_parameters(self, exploration_time_topic, occ_map_topic, avg_speed_topic):
        # -- set the time limits
        # use a subscriber to get the current time limit
        # the rostopic in file whistle_time publishes the time remaining.
        # if the topic gets a inf value, then the time limit is unknown, else 
        # it is known
        self.time_remaining = np.Inf  # TODO fix
        self.dist_threshold = 0.3
        # -- average speed estimate
        self.avg_speed_est = 0.03  # average speed estimate to calculate time estimate
        self.avg_speed = 0.03
        rospy.Subscriber(avg_speed_topic, Float64, callback=self.callback_avg_speed)
        # -- get time remaining
        rospy.Subscriber(exploration_time_topic, Float64, self.callback_time)
        self.occ_map = OccupancyGrid()
        # -- occupancy grid map
        self.flag_map_read = False
        rospy.Subscriber(occ_map_topic, OccupancyGrid, self.callback_map)
        # -- to read the odometry only after the map is read
        while not self.flag_map_read:
            continue
        rospy.Subscriber('/odom', Odometry, self.callback_odom)

        # -- set up rate
        self.ros_pub_rate = rospy.Rate(30)  # Publish rate

        # -- sleep to get all values from subscribers
        self.ros_pub_rate.sleep()
        rospy.sleep(1.)  # sleep for one second

        # -- get initial coordinate for the robot
        self.init_coordinate = np.array([[self.current_pose.position.x, self.current_pose.position.y]])
        self.init_map_coordinate = self._odom_to_map_coordinate(self.current_pose)
        # TODO: Delete next two lines
        # self.init_coordinate = np.array([[-2.5, 0]])
        # self.init_map_coordinate = (500, 452)
        # -- print initial position
        print(f"init coordinate: {np.round(self.init_coordinate, 2)}")
        print(f"Init map coordinate: {self.init_map_coordinate}")
        print("#" * 150)

    # --------------------------------------------------------------------------
    # -- Prioritized Exploration agent
    # --------------------------------------------------------------------------
    def _prioritized_exploration(self):
        """ Priority exploration agent
        Has a two way toggle, 
        Step A: Find the next goal.
        Step B: Guide robot to next goal.
        ----------------------------------------------------------------------------
        Functions Called:
        _exploration_completed(): Check if exploration is complete.
        _get_goal_positions(): returns a list of goal positions, a sequence of robot
                            positions that the robot should visit to reach the
                            goal position. Once it reaches the goal position, 
                            change the flag_find_next_goal to False
        """
        print("In Prioritized Exploration")
        self.flag_find_next_goal = True
        self.robot_stuck = False
        self.robot_stuck_turns = 0
        self.prev_map_pose = self.cur_map_pose
        while not self._exploration_completed():
            print("Prioritized Exploration incomplete!")
            # -- check for remaining time
            print("Time remaining: ", self.time_remaining)
            if self._check_time_remaining:  # True is time remaining is less than 1.0
                print("Time remaining to explore! Time: ", self.time_remaining)

                # -- check flag of robot
                if self.flag_find_next_goal:  # Flag is true if finding next location
                    self.time_1 = rospy.get_time()
                    print("Finding next goal!")
                    self.goal_list = self.get_goal_path()
                    if len(self.goal_list) == 0:  # no nodes can be visited within time limit
                        self.flag_find_next_goal = False
                        rospy.set_param('flag_publish_nodes_rviz', False)
                        break
                    # -- print goal coordinates
                    print("Got Goal coordinates as:")
                    for item in self.goal_list:
                        print(f"{item[0]:.3f} {item[1]:.3f}")
                    print("-" * 100)
                    # -- set flag to move robot and stop finding goal positions
                    self.flag_find_next_goal = False
                else:
                    # -- save robot pose before movement
                    self.prev_map_pose = self.cur_map_pose
                    # -- move to goal
                    self.time_2 = rospy.get_time()
                    print("Calculated Goal: Time Remaining: ", self.time_remaining)
                    print("Time taken to calculate path: ", (self.time_2 - self.time_1))
                    print("_prioritized_exploration:: move to goal position")
                    rospy.set_param('flag_publish_nodes_rviz', False)
                    self._move_to_goal()
                    # -- if robot reaches goal position
                    self.flag_find_next_goal = True
                    print("Robot has moved to goal position!")
                    rospy.set_param('flag_publish_nodes_rviz', True)
            else:
                break
        self.return_home()

    # --------------------------------------------------------------------------
    # Function: return_home 
    # --------------------------------------------------------------------------
    # Gets the waypoints for the robot to head back home.
    # --------------------------------------------------------------------------
    def return_home(self, debug_print=True):
        """
        Input: 
        calculatedPath: True If the node distances to home is calculated for 
        using the function distance_to_home is calculated. The variables 
        self.return_home_cost and self.source_nodes_home are available.
        """
        print("#" * 100)
        print("Return Home function")
        print("#" * 100)

        # -- check if robot is already is in the last node
        dist = self._get_euclidean_dist(self.cur_map_pose, self.init_map_coordinate)
        if dist < self.dist_threshold / self.resolution:
            print("Robot is already in home node")
            return

        occ_graph = self.convert_map_graph_client(self.occ_map)
        occ_graph_dict = self._get_graph_as_dictionary(occ_graph)
        home_node_idx, cost_home, source_nodes_home = self.distance_to_home(occ_graph, occ_graph_dict)
        self.home_node_idx = home_node_idx
        self.return_home_cost = cost_home
        self.source_nodes_home = source_nodes_home
        # ----------------------------------------------------------------------
        # -- Task 2: Calculate Path to home
        # -- Task 2.1: Get closest vertex to home location
        # ----------------------------------------------------------------------
        closest_edge_point, min_dist_edge_point, min_dist_source, min_dist_target, edge_idx = self._find_closest_point_in_graph(
            occ_graph, self.cur_map_pose)

        print("Robot map position: ", self.cur_map_pose)
        print("Closest edge point", closest_edge_point)
        print("Min Dist Edge point: ", min_dist_edge_point)
        print("Edge souce: ", min_dist_source)
        print("Edge target: ", min_dist_target)

        # -- closest node based on the number of points to either vertex
        edge_pts = np.array(occ_graph.g.links[edge_idx].pts)
        n_edge_pts = occ_graph.g.links[edge_idx].nPoints
        edge_pts = edge_pts.reshape(n_edge_pts, 2)
        startingClosest = True
        for idx, edge_point in enumerate(edge_pts):
            # -- check if it is equal to closest edge point
            if np.array_equal(closest_edge_point, edge_point):
                if idx < n_edge_pts / 2:
                    startingClosest = True
                else:
                    startingClosest = False
                break
        if startingClosest:
            min_dist_idx = min_dist_source
        else:
            min_dist_idx = min_dist_target
        # min_dist_idx is the closest point to the current node
        rev_path = [min_dist_idx]
        # set current node
        cur_node = min_dist_idx

        # -- check time remaining
        if self.avg_speed >= 0.03:
            time_est = cost_home[cur_node] * self.resolution / self.avg_speed
        else:
            time_est = cost_home[cur_node] * self.resolution / 0.03
        print("Current Node: ", cur_node)
        print("Time estimate to return home :", time_est)
        print("Time remaining: ", self.time_remaining)
        # Robots usually reach faster than they explore, we assume 10 seconds
        # for the corner cases when the robot tries to return home.
        if 0.9 * time_est > self.time_remaining:
            print("Time estimate is larger than time remaining")
            print("Robot cannot return home!")
            return

        # ----------------------------------------------------------------------
        # -- Task 2.1: Get path from current location to home location
        # ----------------------------------------------------------------------
        # Loop through nodes till current node is source
        while cur_node != home_node_idx:
            # -- get source of target node
            source_cur_node = source_nodes_home[cur_node]
            if source_cur_node == cur_node:
                print("Path to home node not found!")
                break
            # -- add source_cur_node to rev_path
            rev_path.append(source_cur_node)
            # -- change the current node
            cur_node = source_cur_node

        # rev_path.reverse() # no reviewser as we are headed towards the target node
        print("Path to home (fixed): {}".format(rev_path))
        print("Path with Odometry coordinates: ")
        for v in rev_path:
            map_coord = occ_graph_dict[v][0].o
            # print "Goal Path to coordinates: Map Coord: ", map_coord
            odom_coord = self._map_to_odom_coordinate(map_coord)
            if debug_print:
                print("Node: {}\tCoordinate: {}".format(v, odom_coord))

        gpc_init = np.array([])
        # -- Convert path to coordinates and add the last mile
        if len(rev_path) > 1 and min_dist_source == rev_path[1]:
            print("return_home :: min_dist_source is the second node")
            # -- get the edge from the closest_edge_point all the way to the source
            gpc_init = self._get_cells_through_edge(closest_edge_point, True, occ_graph.g.links[edge_idx])
            del (rev_path[0])
        elif len(rev_path) > 1 and min_dist_target == rev_path[1]:
            print("return_home :: min_dist_target is the second node")
            gpc_init = self._get_cells_through_edge(closest_edge_point, False, occ_graph.g.links[edge_idx])
            del (rev_path[0])
        elif min_dist_source == rev_path[0]:
            print("return_home :: min_dist_source is the first node")
            gpc_init = self._get_cells_through_edge(closest_edge_point, True, occ_graph.g.links[edge_idx])
        elif min_dist_target == rev_path[0]:
            print("return_home :: min_dist_target is the first node")
            gpc_init = self._get_cells_through_edge(closest_edge_point, False, occ_graph.g.links[edge_idx])

        # -- print the gpc init
        print("return home: GPC_init:")
        print(np.round(gpc_init, 2))

        # -- add the goal path with edges
        gpc = self.goal_path_to_coordinates_with_edges(occ_graph_dict, rev_path, True)

        # -- add gpc init to gpc
        gpc = np.array(gpc)
        gpc = np.vstack((gpc_init, gpc))
        gpc = gpc.tolist()

        print("return home: GPC")
        print(np.round(gpc, 2))

        # ----------------------------------------------------------------------
        # -- Task 2.2: Get path from goal node to goal coordinate
        # ----------------------------------------------------------------------
        print("Calculate post goal path: Last mile to home node")
        # home_closest_edge_point, home_min_dist_edge_point, home_min_dist_source, home_min_dist_target, home_edge_idx = self._find_closest_point_in_graph(occ_graph, self.init_map_coordinate)
        home_closest_edge_point = self.home_closest_edge_point
        home_min_dist_edge_point = self.home_min_dist_edge_point
        home_min_dist_source = self.home_min_dist_source
        home_min_dist_target = self.home_min_dist_target
        home_edge_idx = self.home_edge_idx
        print("Home Closest Edge Point: ", home_closest_edge_point)
        print("Home Min Dist Edge Point: ", home_min_dist_edge_point)
        print("home_min_dist_source: ", home_min_dist_source)
        print("home_min_dist_target: ", home_min_dist_target)
        print("home_edge_idx: ", home_edge_idx)

        # -- print all edges
        print("All edges:")
        occ_graph_edges = occ_graph.g.links
        for ei, e in enumerate(occ_graph_edges):
            print("\t\tIndex: {}\tSource: {}\tTarget: {}".format(ei, e.source, e.target))
        print("End All edges")
        print("rev_path nodes idx: ", rev_path)

        # -- convert path to coordinates and add the last mile
        gpc_post = np.array([])
        if len(rev_path) > 1 and home_min_dist_source == rev_path[-2]:
            print("return_home :: calc gpc_post: home_min_dist_source is the second last node")
            gpc_post = self._get_cells_through_edge(home_closest_edge_point, True, occ_graph.g.links[home_edge_idx])
            del (rev_path[-1])
        elif len(rev_path) > 1 and home_min_dist_target == rev_path[-2]:
            print("return_home :: calc gpc_post: home_min_dist_target is the second last node")
            gpc_post = self._get_cells_through_edge(home_closest_edge_point, False, occ_graph.g.links[home_edge_idx])
            del (rev_path[-1])
        elif home_min_dist_source == rev_path[-1]:
            print("return_home :: calc gpc_post: min_dist_source is the last node")
            gpc_post = self._get_cells_through_edge(home_closest_edge_point, True, occ_graph.g.links[home_edge_idx])
        elif home_min_dist_target == rev_path[-1]:
            print("return_home :: calc gpc_post: min_dist_target is the last node")
            gpc_post = self._get_cells_through_edge(home_closest_edge_point, False, occ_graph.g.links[home_edge_idx])
        # -- gpc_post gives the points from closest_edge_point to the nearest vertex, 
        # this should be reversed as the robot should head to the point

        gpc_post = np.flip(gpc_post, 0)
        print("return home:: gpc_post : ")
        print(np.round(gpc_post, 2))
        print("#" * 100)
        print("Adding the home coordinate to the last of gpc_post")
        gpc_post = np.vstack((gpc_post, self.init_coordinate))
        print("GPC Post after adding last coordinate: ")
        print(np.round(gpc_post, 2))
        print("#" * 150)

        # -- create gpc including gpc, gpc_init, gpc_post
        gpc = self.goal_path_to_coordinates_with_edges(occ_graph_dict, rev_path, True)

        # -- add gpc init to gpc
        gpc = np.array(gpc)
        gpc = np.vstack((gpc_init, gpc, gpc_post))
        gpc = gpc.tolist()

        # -- print 
        print("GPC final")
        print(np.round(gpc, 2))

        # -- Set flag to move to next goal
        self.flag_find_next_goal = False
        self.prev_map_pose = self.cur_map_pose
        # -- set params
        rospy.set_param('flag_publish_nodes_rviz', False)
        self.goal_list = gpc
        self._move_to_goal()
        print("Robot has moved to goal position!")
        rospy.set_param('flag_publish_nodes_rviz', True)

    # --------------------------------------------------------------------------
    # -- Function: _move_to_goal
    # --------------------------------------------------------------------------
    # Moves robot through the goal list till the last point on the goal list
    # --------------------------------------------------------------------------
    # Input:
    #   goal_list: list of intermediate goal positions from the current
    #               position to the goal position.
    # --------------------------------------------------------------------------
    def _move_to_goal(self):
        """
        Input: 
            self.goal_list:
        The goal_list is a list of Points (ROS geometry_msgs/Point). The goal
        of this function is to provide adequate velocity (twist and linear)
        to move the robot in the correct direction towards the last goal 
        position.
        http://docs.ros.org/en/melodic/api/geometry_msgs/html/msg/Point.html
        
        A while loop checks if goal_list is empty or not. 
        
        Within the while loop, an if statement checks if the current pose 
        of the robot is within a threshold of the current target. A threshold 
        is a global variable self.dist_threshold. If the robot is within the 
        threshold distance, the robot is assumed to have reached that target
        location, the first element is popped from the list, and the while
        loop continues.
        
        """
        print("Move to goal: Before robot movement: Time Remaining: ", self.time_remaining)
        rospy.set_param('robot_moving', True)
        while self.goal_list and self._check_time_remaining:
            # -- get current point
            # print "Exploration Agent: _move_to_goal :: Goal List[0]: {}".format(self.goal_list[0])
            self.current_target = self.goal_list.pop(0)
            # print "Exploration Agent: _move_to_goal :: Current Target: {}".format(self.current_target)
            # -- check if robot is within current_target
            gpx = self.current_target[0]
            gpy = self.current_target[1]
            goalPoint = Point(gpx, gpy, 0)
            # print "Exploration Agent: _move_to_goal :: Goal Point x: {}\ty: {}".format(np.round(goalPoint.x,2), np.round(goalPoint.y,2))
            # print "-"*100
            DriveRobot(goalPoint, None, 0.1, 0.1)
        rospy.set_param('robot_moving', False)
        print("Move to goal: After robot movement:  Time Remaining: ", self.time_remaining)
        return

    # -------------------------------------------------------------------------
    # -- Check if exploration is complete
    # Also sets the occ_graph so that it is not called the second time
    # -------------------------------------------------------------------------
    def _exploration_completed(self):
        exploration_completed_flag = False

        # -- get the current graph from the occupancy grid
        occ_graph = self.convert_map_graph_client(self.occ_map)
        self.occ_graph = occ_graph
        # -- get the nodes from the graph
        occ_graph_vertices = occ_graph.g.nodes

        # -- check the number of vertices that are frontier
        n_frontier_nodes = 0
        for node in occ_graph_vertices:
            if node.priority > 0:
                n_frontier_nodes += 1

        if n_frontier_nodes == 0:
            exploration_completed_flag = True

        # -- if time remaining is zero
        if self.time_remaining <= 0:
            return True

        return exploration_completed_flag

    # -------------------------------------------------------------------------
    # -- Convert Map to Graph Client
    # -------------------------------------------------------------------------
    def convert_map_graph_client(self, occgrid_msg):
        """ Converts Map to a graph.
        Args:
            occgrid_msg: Occupancy grid message
        Return:
            Graph of the environment. Check the msg directory for format.
        """
        try:
            # -- convert the occupancy grid to a graph
            convert_map_graph = rospy.ServiceProxy('convert_map_to_graph', ConvertMapGraph)
            # -- convert the occupancy grid to a graph
            resp_graph = convert_map_graph(occgrid_msg)
            # rospy.loginfo(resp_graph)
            return resp_graph

        except rospy.ServiceException as e:
            print("Serice call failed: %s" % e)

    # -------------------------------------------------------------------------
    # Function: _check_time_remaining
    # Check if time is remaining to continue exploration
    # -------------------------------------------------------------------------
    def _check_time_remaining(self):
        # -- check for time limits
        if self.time_remaining < 1.0:
            return True
        else:
            return False


    def add_nodes_exception_vertices(self, node_path, occ_graph_dict):
        """ Add the vertices in the path to exception vertices, so that
        the next time, these vertices, the priority is turned to zero and
        the robot selects other vertices.
        """
        for v in node_path:
            map_coord = occ_graph_dict[v][0].o
            self.exception_locations.append((map_coord, rospy.get_time()))

    # -------------------------------------------------------------------------
    # Function: get_goal_path
    # -------------------------------------------------------------------------
    def get_goal_path(self, debug_print=False):
        """ Get the goal path from the current position to the goal position

        Functions Called:
        1. _get_goal_positions
        2. add_nodes_exception_vertices
        3. _find_closest_point_in_graph
        4. _get_cells_through_edge
        Other functions:
        1. _map_to_odom_coordinate
        """
        # -- get the occupancy grid graph
        occ_graph = self.occ_graph

        # -- get goal positions
        # Returns: return occ_graph_dict, lowest_cost_action, source_all_nodes, min_dist_idx
        occ_graph_dict, p_node_list, source_all_nodes, source_idx = self._get_goal_positions(occ_graph, False)
        print("Returned from get goal positions!")
        # p_node_list: Priority node list
        """
        Example output of print is here: https://pastebin.com/raw/1nEnKNu5
        """

        # -- print
        if debug_print == True:
            # if True:
            print("#" * 100)
            print("get_goal_path:: Prioritized Node List: ")
            for item in p_node_list:
                print("-" * 50)
                print(item)
            print("-" * 100)
            print("get_goal_path:: Source of all nodes:")
            print(source_all_nodes)
        print("-" * 100)

        # -- if p_node_list is empty
        if len(p_node_list) == 0:
            gpc = []
            return gpc
        # then we do not have any exploration to complete. 

        # -- Get a path from source_idx to p_node_list using source_all_nodes
        if debug_print:
            print("get_goal_path:: Get a path from source node to best choice node")
        target_idx = 0
        while True:
            # if parent of target_idx node is the same node (i.e.) no parents
            # change the index as target_idx node is probably a disjoined graph
            init_target_node_id = p_node_list[target_idx][-1].val.id
            rev_path = [init_target_node_id]
            # set current node
            cur_node = init_target_node_id
            # Loop through nodes till current node is source
            while cur_node != source_idx:
                # -- get source of target node
                source_cur_node = source_all_nodes[cur_node]
                if source_cur_node == cur_node:
                    break
                # -- add source_cur_node to rev_path
                rev_path.append(source_cur_node)
                # -- change the current node
                cur_node = source_cur_node
            if cur_node == source_idx:
                # print "Path discovered!"
                break
            else:
                target_idx += 1
        rev_path.reverse()

        # -- debug print
        print("Robot position: ", np.round([self.current_pose.position.x, self.current_pose.position.y], 2))
        print("Path (fixed): {}".format(rev_path))
        print("Path with Odometry coordinates: ")
        for v in rev_path:
            map_coord = occ_graph_dict[v][0].o
            # print "Goal Path to coordinates: Map Coord: ", map_coord
            odom_coord = self._map_to_odom_coordinate(map_coord)
            print("Node: {}\tCoordinate: {}".format(v, odom_coord))

        # -- add the node locations to the exception locations so that nodes get de-prioritized next time
        self.add_nodes_exception_vertices(rev_path, occ_graph_dict)

        # -- first mile travel to nearest node
        # closest_edge_point, min_dist_edge_point, min_dist_source, min_dist_target, edge_idx = self._find_closest_point_in_graph(occ_graph, self.cur_map_pose)
        closest_edge_point = self.closest_edge_point
        min_dist_edge_point = self.min_dist_edge_point
        min_dist_source = self.min_dist_source
        min_dist_target = self.min_dist_target
        edge_idx = self.edge_idx

        # print "=*"*50
        # print "Min Dist Idx Closest Vertex: ", min_dist_idx
        # print "get_goal_path :: Min Dist Idx from any edge:", closest_edge_point
        # print "get_goal_path :: Min Dist Idx Edge Source: ", min_dist_source
        # print "get_goal_path :: Min Dist Idx Edge Target: ", min_dist_target
        # print "=*"*50
        # -- check if the first or second node is the closest node from the robot
        # print "get_goal_path :: Current robot position", np.round(self.current_pose.position.x,2), "\t", np.round(self.current_pose.position.y,2)
        gpc_init = np.array([])
        if len(rev_path) > 1 and min_dist_source == rev_path[1]:
            # print "get_goal_path :: min_dist_source is the second node"
            # -- get the edge from the closest_edge_point all the way to the source
            gpc_init = self._get_cells_through_edge(closest_edge_point, True, occ_graph.g.links[edge_idx])
            del (rev_path[0])
        elif len(rev_path) > 1 and min_dist_target == rev_path[1]:
            # print "get_goal_path :: min_dist_target is the second node"
            gpc_init = self._get_cells_through_edge(closest_edge_point, False, occ_graph.g.links[edge_idx])
            del (rev_path[0])
        elif min_dist_source == rev_path[0]:
            # print "get_goal_path :: min_dist_source is the first node"
            gpc_init = self._get_cells_through_edge(closest_edge_point, True, occ_graph.g.links[edge_idx])
        elif min_dist_target == rev_path[0]:
            # print "get_goal_path :: min_dist_target is the first node"
            gpc_init = self._get_cells_through_edge(closest_edge_point, False, occ_graph.g.links[edge_idx])
        # -- print gpc init
        if debug_print:
            print("get_goal_path:: GPC init: ")
            print(np.round(gpc_init, 2))

        if debug_print:
            print("rev_path: ", rev_path)
            print("get_goal_path:: Calling Goal path to coordinates with edges")
        # gpc: goal_path_corrds
        gpc = self.goal_path_to_coordinates_with_edges(occ_graph_dict, rev_path, True)
        if debug_print:
            print("get_goal_path:: GPC: ")
            print(np.round(gpc, 2))

        gpc = np.array(gpc)
        # if euclidean distance from the last node of gpc_init is further to
        # the first node of gpc_init, then flip gpc_init
        """
        if len(gpc_init) > 0:
            if self._get_euclidean_dist(gpc_init[-1], gpc[0]) > self._get_euclidean_dist(gpc_init[0], gpc[0]):
                # -- flip gpc_init
                gpc_init = np.flip(gpc_init,0)
                print "flipped gpc init"
        """
        # -- merge gpc init and gpc to create the final goal path
        if debug_print:
            print("GPC size: ", gpc.shape)
        gpc = np.vstack((gpc_init, gpc))
        if debug_print:
            print("GPC size (after gpc_init): ", gpc.shape)
        # -- convert gpc to list
        gpc = gpc.tolist()
        return gpc

    def _get_cells_through_edge(self, start_cell, source_flag, edge, sparse=5, debug_print=False):
        ''' Get edge pts from a certain point to the end of the edge_pts array

        Input:
            start_cell: the first cell the robot should reach to
            source_flag: True if cells are requires from the 0 to start_cell
        '''
        edge_pts = np.array(edge.pts)
        edge_pts = edge_pts.reshape((edge.nPoints, 2))
        idx = 0
        if debug_print:
            print("_get_cells_through_edge:")
            print("edge_pts: ")
            print(edge_pts)
            print("-" * 50)
            print("start_cell: ", start_cell)
        for edge_points in edge_pts:
            if np.array_equal(edge_points, start_cell):
                break
            idx += 1
        # print "Index: ", idx
        # print "source_flag: ", source_flag
        if source_flag:
            # require the cells from the start to the start_cell
            pts_required = edge_pts[:idx + 1, :]
            # -- ensure that the pts are switched order so that the start is 
            # the first node.
            pts_required = np.flip(pts_required, 0)
        else:
            pts_required = edge_pts[idx:, :]
        # -- sparsify the array according to requirement
        pts_required = pts_required[0:pts_required.shape[0]:sparse, :]

        # -- debug print
        if debug_print:
            print("_get_cells_through_edge :: points: map coordinates")
            for v in pts_required:
                print(v)
            print("-" * 100)

        # Step 2: Convert to odom coordinates
        coordinates = []
        for v in pts_required:
            odom_coord = self._map_to_odom_coordinate(v)
            coordinates.append(odom_coord)
        return coordinates

    # --------------------------------------------------------------------------
    # Function: Get odom coordinates from vertex indices
    # --------------------------------------------------------------------------
    def goal_path_to_coordinates(self, occ_graph_dict, path, debug_print=True):
        """ Convert goal path to odom coordinates.
        """
        coordinates = []
        for v in path:
            map_coord = occ_graph_dict[v][0].o
            # print "Goal Path to coordinates: Map Coord: ", map_coord
            odom_coord = self._map_to_odom_coordinate(map_coord)
            coordinates.append(odom_coord)
        return coordinates

    # --------------------------------------------------------------------------
    # Function: Get odom coordinates from vertex indices and edge information
    # --------------------------------------------------------------------------
    def goal_path_to_coordinates_with_edges(self, occ_graph_dict, path, debug_print=False):
        """
        Convert goal path to odom coordinates including the vertices and edge
        positions as well.
        Tasks:
        1.  Check if the edge between two vertices are within a straight line
        2.  If they are not within a straight line, take the center of the 
            edge vertex, now check if the two lines are within a straight line
            If not, split each part to two more. 
            The other way one can do it is, to consider every 10th point as the

        Edge message structure:
            int64 source
            int64 target
            float64 weight
            int64 nPoints
            int64[] pts
        The points in pts are arranged in [row_p1,col_p1,row_p2,col_p2] format.
        So for the 5th point, we'll access the 10th element and the 11th 
        element


        """
        # Do I need a function to convert points to 
        # Step 1: Get the vertex coordinates and edges in the path
        v_coords = []
        path_edges = []
        edge_reverse_flags = []  # used to set whether the edge pixels
        # need to be reversed. They need to be reversed if the source of edge is
        # the next vertex
        for idx, v in enumerate(path):
            # print "calculate gpc: Vertex: ", v
            map_coord = occ_graph_dict[v][0].o
            v_coords.append(map_coord)
            # -- check for edge
            # check for edge if a next vertex is not indexed
            if idx < len(path) - 1:
                next_v = path[idx + 1]
                # get the edge and append it to edges
                for e in occ_graph_dict[v][1]:
                    # the target and source check sometimes seem to fail
                    # the bug may be because of how the edge is saved 
                    # from the service. It does not happen always, but
                    # sometimes. So the current check if a euclidean distance
                    # check.
                    if e.target == next_v:
                        path_edges.append(e)
                        edge_reverse_flags.append(False)
                        break
                    elif e.source == next_v:
                        path_edges.append(e)
                        edge_reverse_flags.append(True)
                        break

        # Print the path edges to debug
        # for eidx, e in enumerate(path_edges):
        #    print "Edge:: \tSource: {}\tTarget: {}".format(e.source, e.target)
        #    print "Edge reverse_flags: {}".format(edge_reverse_flags[eidx])

        # Step 2: Check if the vertices are in a straight line
        """
        Traverse the edge points to get every 2nd point. 
        """
        edge_sampling_threshold = 4
        v_e_coords = np.array([v_coords[0]])
        for idx, e in enumerate(path_edges):
            e_pts = np.array(e.pts)
            e_pts = e_pts.reshape(e.nPoints, 2)
            e_pts = e_pts[0:e.nPoints:edge_sampling_threshold, :]
            if edge_reverse_flags[idx]:
                e_pts = np.flip(e_pts, 0)
                # print "Current edge points: "
                # print e_pts
            v_e_coords = np.vstack((v_e_coords, e_pts))
            v_e_coords = np.vstack((v_e_coords, np.array(v_coords[idx + 1])))
        
        # Step 3: Convert to odom coordinates
        coordinates = []
        for v in v_e_coords:
            odom_coord = self._map_to_odom_coordinate(v)
            coordinates.append(odom_coord)
        return coordinates

    # --------------------------------------------------------------------------
    # Function: _handle_exception_vertices
    # --------------------------------------------------------------------------
    def _handle_exception_vertices(self, occ_graph, timeout=5000, distance_threshold=4, debug_print=True):
        """ Make vertex priority to 0 if they fall within a threshold distance
        from any of the exception vertices.

        Input:
            occ_graph: occupancy map graph
            distance_threshold: 
            self.exception_locations: A list of tuples, each tuple[0] is a location
                                       and tuple[1] is the time in which the location
                                       was set.
            
        
        Tasks:
            Removes locations that are more than 100 seconds old.
            For nodes within a certain threshold distance from a exception
            location, the priority of the node is reduced to zero.
        """
        # -- remove locations that have timed out.
        cur_time = rospy.get_time()
        reduce_priority_locations = []
        if debug_print:
            print("#" * 100)
            print("Function: handle_exception_vertices")
            print("Exception Locations: ")
            print(self.exception_locations)
            print("#" * 100)
        for idx, el in enumerate(self.exception_locations):
            node_o = el[0]
            # print "Node o: ", el[0]
            node_time = el[1]
            if cur_time - node_time < timeout:
                reduce_priority_locations.append(el)
                # -- go through the nodes, if a node is within threshold 
                # distance, reduce the priority to zero.
                for node_id, node in enumerate(occ_graph.g.nodes):
                    dist = self._get_euclidean_dist(node.o, node_o)
                    if dist < distance_threshold:
                        occ_graph.g.nodes[node_id].priority = 0

        self.exception_locations = reduce_priority_locations
        return occ_graph

    # --------------------------------------------------------------------------
    # Function: _get_goal_positions
    # --------------------------------------------------------------------------
    def _get_goal_positions(self, occ_graph, debug_print=True):
        """ Gets the next position for the robot to visit. 
        Input:
            occ_graph
        Return:
        1.  occ_graph_dict: A dictionary of the graph.
        2.  lowest_cost_action: Prioritized list of frontier points
            Format: List l1
            l1 element: Tuple t1
            t1 element: [0]: Priority of node
                        [1]: Cost to visit that node
                        [2]: Node parameters (node object)
        3.  source_all_nodes
        4.  min_dist_idx: closest vertex to the current node
            
        The output is a list of points(tuples) the robot needs to visit to
        reach the overall goal position.
            
        
        The output is based on the occupancy grid coordinate of the robot.
        Tasks:
            1. Get the current position of the robot.
            2. Find out the closest vertex to the robot that vertex is the first
               goal point of the robot.
            3. Followed by which, find using dijstras' algorithm, the distance 
               to all the remaining vertices in the graph.
            4. Choose the vertex with the highest priority and lowest distance.
        """

        # -- get the nodes from the graph
        occ_graph_vertices = occ_graph.g.nodes

        # -- get the current robot position
        """
        If the robot is at the same position as any frontier node, then make 
        the node non-frontier. (Priority = 0)
        """
        min_dist, min_dist_idx = self._find_closest_vertex(occ_graph)
        if min_dist < 3:
            # -- robot is at node that is a frontier node
            min_vertex_coord = occ_graph_vertices[min_dist_idx].o
            dist_prev_pose = self._get_euclidean_dist(min_vertex_coord, self.prev_map_pose)
            if occ_graph_vertices[min_dist_idx].priority > 0 and dist_prev_pose < 3:
                # -- robot was here last iteration
                # the robot is stuck at this position after an 
                # entire iteration. 
                print("Robot is stuck at current location")
                print("Forcing current node to be non-priority node")
                occ_graph.g.nodes[min_dist_idx].priority = 0
                # -- add this location as an exception location
                """ Exception locations are locations that should be marked as 
                non frontier locations. This would force the robot to explore 
                """
                self.exception_locations.append((min_vertex_coord, rospy.get_time()))

        # -- handle exception priority
        occ_graph = self._handle_exception_vertices(occ_graph)

        # -- Find closest vertex to robot position
        # TODO Replace this function
        min_dist, min_dist_idx = self._find_closest_vertex(occ_graph)
        print("_get_goal_positions: Min Dist Idx Closest Vertex through vertex search: ", min_dist_idx)
        closest_edge_point, min_dist_edge_point, min_dist_source, min_dist_target, edge_idx = self._find_closest_point_in_graph(
            occ_graph, self.cur_map_pose)
        self.closest_edge_point = closest_edge_point
        self.min_dist_edge_point = min_dist_edge_point
        self.min_dist_source = min_dist_source
        self.min_dist_target = min_dist_target
        self.edge_idx = edge_idx
        # -- closest node based on the number of pts to either vertex
        edge_pts = np.array(occ_graph.g.links[edge_idx].pts)
        n_edge_pts = occ_graph.g.links[edge_idx].nPoints
        edge_pts = edge_pts.reshape(n_edge_pts, 2)
        startingClosest = True
        for idx, edge_point in enumerate(edge_pts):
            # -- check if it is equal to closest edge point
            if np.array_equal(closest_edge_point, edge_point):
                if idx < n_edge_pts / 2:
                    startingClosest = True
                else:
                    startingClosest = False
                break
        if startingClosest:
            min_dist_idx = min_dist_source
        else:
            min_dist_idx = min_dist_target

        print("_get_goal_positions: Min Dist Idx Closest Vertex through closest edge point: ", min_dist_idx)
        if debug_print:
            print("#" * 100)
            print("Function: _get_goal_positions")
            print("-" * 100)
            print("Closest vertex to robot Position: ", min_dist_idx)
            print("-" * 100)

        # -- print the graph
        # print_occ_graph(occ_graph)

        # -- convert the occ_graph to a dictionary
        # -- convert graph to dictionary
        occ_graph_dict = self._get_graph_as_dictionary(occ_graph)

        # -- Use dijkstra's algorithm to get vertices in priority
        # dist_all_nodes is a dictionary
        #   key: node_idx
        #   val: float
        # source_all_nodes is a dictionary
        #   key: node_idx
        #   val: node_idx
        # source_all_nodes is a dictionary
        dist_all_nodes, source_all_nodes = _get_cost_to_all_vertices(
            occ_graph_vertices[min_dist_idx],
            occ_graph_dict,
            False)  # debug print = False

        if True:
            print("-" * 100)
            print("_get_goal_positions: distance_all_nodes: ")
            print(dist_all_nodes)
            print("-" * 100)

        # -- get the list of frontier vertices
        frontier_vertices = self._get_frontier_vertices(occ_graph.g.nodes)
        if debug_print:
            print("Frontier Vertices: ", frontier_vertices)

        # -- delete non frontier nodes
        distance_frontier_nodes = self.del_non_frontier_nodes(frontier_vertices,
                                                              dist_all_nodes)

        # -- calculate distance to home if time is not infinity
        if self.time_remaining != np.inf:
            # -- Get the cost to home.
            # cost_home is a dictionary, id = vertex_id and value is distance (in map coordinates)
            home_node_idx, cost_home, source_nodes_home = self.distance_to_home(occ_graph, occ_graph_dict)
            print("_get_goal_positions: calculated cost home.")
            # above function returned return min_dist_idx, cost_home, source_all_nodes_home
            # print "Cost to home: "
            # print cost_home
            # print "End: Cost to home"
            self.home_node_idx = home_node_idx
            self.return_home_cost = cost_home
            self.source_nodes_home = source_nodes_home

        if debug_print:
            print("-" * 100)
            print("_get_goal_positions: distance_frontier_nodes: ")
            print(distance_frontier_nodes)
            print("-" * 100)

        # -- Arrange the frontier nodes with priority
        # -- if the time limit is known, calculate the priority queue with time
        # -- print the time remaining
        print("Time remaining: ", self.time_remaining)
        # Arrange the nodes based on priority and distance. Priority first, 
        # distance second. Make a wrapper class if required
        lowest_cost_action = []
        print("Before function add nodes to priority queue")
        self._add_nodes_priority_queue(distance_frontier_nodes,
                                       occ_graph,
                                       occ_graph_dict,
                                       lowest_cost_action,
                                       False)  # debug print

        print("_get_goal_positions: Added nodes to priority Queue")

        # -- print the goal point
        print("-*" * 50)
        if True:
            print("Lowest cost goal point: {}".format(lowest_cost_action))
            print("-*" * 50)

        # return the goal point heap and the source to all nodes dictionary
        return occ_graph_dict, lowest_cost_action, source_all_nodes, min_dist_idx

    def distance_to_home(self, occ_graph, occ_graph_dict):
        """ Function: distance_to_home
        Calculates path and distance to home location following the 
        occupancy Graph
        """
        # initial coordinate is the home coordinate.

        # -- Task 1: Closest vertex to home location
        print("Calculating distance to home: closest point to home node")
        closest_edge_point, min_dist_edge_point, min_dist_source, min_dist_target, edge_idx = self._find_closest_point_in_graph(
            occ_graph, self.init_map_coordinate)

        # -- closest node based on the number of points to either vertex
        edge_pts = np.array(occ_graph.g.links[edge_idx].pts)
        n_edge_pts = occ_graph.g.links[edge_idx].nPoints
        edge_pts = edge_pts.reshape(n_edge_pts, 2)
        startingClosest = True
        for idx, edge_point in enumerate(edge_pts):
            # -- check if it is equal to closest edge point
            if np.array_equal(closest_edge_point, edge_point):
                if idx < n_edge_pts / 2:
                    startingClosest = True
                else:
                    startingClosest = False
                break
        if startingClosest:
            min_dist_idx = min_dist_source
        else:
            min_dist_idx = min_dist_target
        self.home_closest_edge_point = closest_edge_point
        self.home_min_dist_edge_point = min_dist_edge_point
        self.home_min_dist_source = min_dist_source
        self.home_min_dist_target = min_dist_target
        self.home_min_dist_idx = min_dist_idx
        self.home_edge_idx = edge_idx
        # -- Task 2: Dijkstra's algorithm to calculate distance to other nodes
        occ_graph_vertices = occ_graph.g.nodes  # get the nodes from the graph
        print("Function: Distance to home: ")
        print("Closest index to home: ", min_dist_idx)
        print("Occupancy graph vertices:")
        # for v in occ_graph_vertices:
        cost_home, source_all_nodes_home = _get_cost_to_all_vertices(
            occ_graph_vertices[min_dist_idx],
            occ_graph_dict,
            False)  # debug print = False
        # -- return
        return min_dist_idx, cost_home, source_all_nodes_home

    # --------------------------------------------------------------------------
    # Function: _find_closest_vertex
    # BUG: This code does not work if there is a subgraph which does not have 
    # a connection anywhere else, it might turn out that the nearest node is
    # unusable.
    # --------------------------------------------------------------------------
    def _find_closest_vertex(self, occ_graph, cur_pos=None):
        """ Find the closest vertices from the 
        Args:
        """
        # -- get the nodes from the graph
        occ_graph_vertices = occ_graph.g.nodes
        # -- if current pose is not set
        if cur_pos is None:
            cur_pos = self.cur_map_pose
        # -- Find the closest vertex to robot position
        min_dist = np.Inf
        min_dist_idx = 0
        for idx, node in enumerate(occ_graph_vertices):
            # -- get distance between current position and node
            node_coord = (node.pts[0], node.pts[1])
            cur_dist = math.sqrt(float(cur_pos[0] - node_coord[0]) ** 2 +
                                 float(cur_pos[1] - node_coord[1]) ** 2)
            if cur_dist < min_dist:
                min_dist = cur_dist
                min_dist_idx = idx

        return min_dist, min_dist_idx

    def _find_closest_point_in_graph(self, occ_graph, cur_pos):
        """ Finds the closest point in the graph. 
        Args:
            occ_graph
            cur_pos: robot position in map coordinate
        Output:
            map coordinate of the closest point with a clear line of view

        BUG: The nearest node calculated may be on the other side of an obstacle
        Check with a 2D line that the path from robot position to obstacle is 
        free from anything but explored unobstructed cells.
        """
        # -- get min dist based on vertex. that should be the minimum value of lowest dist
        # min_dist_v, min_dist_v_idx = self._find_closest_vertex(occ_graph, cur_pos)
        # print "Find closest point in graph"
        # print "Min Dist based on vertex: ", min_dist_v
        # print "Min Idx based on vertex: ", min_dist_v_idx
        # -- get the edges from the graph
        # occ_graph_vertices = occ_graph.g.nodes
        occ_graph_edges = occ_graph.g.links
        # -- initialize variables to calculate point closest to 
        lowest_dist = np.inf  # min_dist_v
        lowest_dist_point = [0, 0]
        lowest_dist_edge_source = 0
        lowest_dist_edge_target = 0
        lowest_dist_edge_idx = 0
        for edge_idx, edge in enumerate(occ_graph_edges):
            pts = np.array(edge.pts)
            pts = pts.reshape((edge.nPoints, 2))
            # -- find distance between robot pose and edge point
            for edge_point in pts:
                dist = math.sqrt(float(cur_pos[0] - edge_point[0]) ** 2 +
                                 float(cur_pos[1] - edge_point[1]) ** 2)
                if dist < lowest_dist:  # and dist > 6:
                    # the dist>6 allows the point to be not decided too close to the robot. (Does not work)
                    # -- check if no obstacles in the path
                    if not check_obstacles(cur_pos, edge_point, self.occ_map_data_array):
                        continue
                    lowest_dist = dist
                    lowest_dist_point[0] = edge_point[0]
                    lowest_dist_point[1] = edge_point[1]
                    lowest_dist_edge_source = edge.source
                    lowest_dist_edge_target = edge.target
                    lowest_dist_edge_idx = edge_idx
        # -- post calculation print
        print("Find closest point in graph: ")
        print("lowest dist point = ", lowest_dist_point)
        print("lowest_dist_edge_source: ", lowest_dist_edge_source)
        print("lowest_dist_edge_target: ", lowest_dist_edge_target)
        print("lowest_dist_edge_idx: ", lowest_dist_edge_idx)

        return lowest_dist_point, lowest_dist, lowest_dist_edge_source, lowest_dist_edge_target, lowest_dist_edge_idx

    # --------------------------------------------------------------------------
    # -- Function:_get_2D_coord_distance
    # --------------------------------------------------------------------------
    def _get_2D_coord_distance(self, goal_map_coord):
        """ Get distance between the current robot pose and a given 2D 
        coordinate in the map.
        
        The function get the current position (map coordinate). Calculates the 
        2D distance between the current map position of the robot and the 2D 
        goal distance in a straight line path.
        ------------------------------------------------------------------------
        Args:
            goal_map_coord: Goal map coordinate. A tuple (row idx, col idx)
        Output:
            dist: (float) L2 distance between the two points.
        """
        # -- get odometry
        cur_pos = self.cur_map_pose
        # -- calculate distance
        dist = math.sqrt(float(cur_pos[0] - goal_map_coord[0]) ** 2 +
                         float(cur_pos[1] - goal_map_coord[1]) ** 2)
        return dist

    # --------------------------------------------------------------------------
    # -- Function:_get_2D_coord_distance
    # --------------------------------------------------------------------------
    def _get_euclidean_dist(self, coord_1, coord_2):
        """ Get distance between the current robot pose and a given 2D 
        coordinate in the map.
        
        The function get the current position (map coordinate). Calculates the 
        2D distance between the current map position of the robot and the 2D 
        goal distance in a straight line path.
        ------------------------------------------------------------------------
        Args:
            goal_map_coord: Goal map coordinate. A tuple (row idx, col idx)
        Output:
            dist: (float) L2 distance between the two points.
        """
        # -- calculate distance
        dist = math.sqrt(float(coord_1[0] - coord_2[0]) ** 2 +
                         float(coord_1[1] - coord_2[1]) ** 2)
        return dist

    # --------------------------------------------------------------------------
    # Function: _get_frontier_vertices
    # --------------------------------------------------------------------------
    def _get_frontier_vertices(self, occ_graph_vertices):
        """ Identifies frontier vertices in occupancy graph vertices.
        """

        # -- get the list of frontier vertices
        frontier_vertices = []
        for node in occ_graph_vertices:
            if node.priority > 0:
                frontier_vertices.append(node.id)

        # -- return
        return frontier_vertices

    # --------------------------------------------------------------------------
    # Function: Convert the graph to a dictionary
    # --------------------------------------------------------------------------       
    def _get_graph_as_dictionary(self, occ_graph):
        """
        Step 1: Initialize nodes to the dictionary. The node ids are dictionary
                keys in the dictionary. 
        Step 2: Go through the edges, the edge indices are arranged in the list.
                For a given edge in the list, add the index of the edge to the
                participating vertices.
        
        The dictionary 'occ_graph_dict' has the same number of elements as the 
        number of nodes in the array. The val of the dictionary is based on the
        
        Edge message structure:
            int64 source
            int64 target
            float64 weight
            int64 nPoints
            int64[] pts
        
        Dictionary Structure:
            idx: node_id
            value: List: [node object, [edge1, edge2, edge3, ...]]
        """
        occ_graph_dict = {}
        for node in occ_graph.g.nodes:
            # -- made the id of the node as the key of the dictionary
            if node.id in occ_graph_dict:
                rospy.loginfo("Vertices repeated twice!")
                rospy.loginfo(str(occ_graph))
                raise ValueError
            occ_graph_dict[node.id] = [node, []]

        # -- traverse through all the edges
        for edge in occ_graph.g.links:
            occ_graph_dict[edge.source][1].append(edge)
            occ_graph_dict[edge.target][1].append(edge)

        return occ_graph_dict

    # --------------------------------------------------------------------------
    # Function: Delete non frontier nodes from dictionary: dict_all_nodes
    # --------------------------------------------------------------------------
    def del_non_frontier_nodes(self, frontier_vertices, dist_all_nodes):
        # copy the dictionary
        distance_frontier_nodes = dist_all_nodes.copy()

        del_node = []
        for nodeIdx, cost in dist_all_nodes.items():
            if nodeIdx not in frontier_vertices:
                del_node.append(nodeIdx)

        # get unique del_node (Seems unnecessary)
        del_node = list(set(del_node))

        # delete the nodes in del_node
        for nodeIdx in del_node:
            del (distance_frontier_nodes[nodeIdx])

        return distance_frontier_nodes

    # --------------------------------------------------------------------------
    # Function: _add_nodes_priority_queue
    # --------------------------------------------------------------------------
    def _add_nodes_priority_queue(self,
                                  distance_frontier_nodes,
                                  occ_graph,
                                  occ_graph_dict,
                                  lowest_cost_action,
                                  debug_print=True):
        """ Function: _add_nodes_priority_queue
        Adds a list of nodes to a priority queue based on the distance from a given
        node. When the time available is not equals to infinity, it also checks for 
        nodes that are not possible to be reached by the 
        Node description:
            int64 id
            int64[] pts
            int64 nPoints
            float64[] o
            float64 priority   
        """

        # -- distance_frontier_nodes
        if debug_print:
            print("-" * 100)
            print("_add_nodes_priority_queue: ")
            print("-" * 100)
            print("distance_frontier_nodes: ")
            print(distance_frontier_nodes)
            print("lowest cost action: ")
            print(lowest_cost_action)
            print("-" * 100)

        # insert the nodes into a heap push
        for node_idx, node_cost in distance_frontier_nodes.items():
            # -- get node priority
            node_priority = occ_graph_dict[node_idx][0].priority
            if debug_print:
                print("Occupancy graph at node_idx: ", node_idx, "::")
                print(occ_graph_dict[node_idx])
                print("-" * 50)
                print("Node priority: ", node_priority)
                print("-" * 50)

            # -- change to use node object
            # heapq.heappush(lowest_cost_action, (node_priority,
            # node_cost,
            # Wrapper(node_idx)))
            time_est = node_cost * self.resolution / self.avg_speed
            if self.time_remaining != np.inf:
                # -- initialize time estimates to infinity
                time_est_target_home = np.inf
                time_est_to_home = np.inf

                # Time estimate = distance in odom coord / speed in odom coord
                if self.avg_speed > 0.025:
                    # average speed goes < 0.025 only when the robot has taken
                    # a certain number of turns along the path. While returning
                    # back to the home location, the robot usually follows
                    # straight line segment edges, and takes less time.
                    time_est_target_home = (self.return_home_cost[node_idx] * self.resolution) / self.avg_speed
                    time_est_to_home = (node_cost + self.return_home_cost[node_idx]) * self.resolution / self.avg_speed
                else:
                    time_est_target_home = (self.return_home_cost[node_idx] * self.resolution) / 0.025
                    time_est_to_home = (node_cost + self.return_home_cost[node_idx]) * self.resolution / 0.025

                # adding offset to account for delays while taking turns
                time_est_target_home += 20
                time_est_to_home += 20

                if time_est_to_home < self.time_remaining:
                    heapq.heappush(lowest_cost_action, (node_priority,
                                                        time_est, time_est_to_home,
                                                        Wrapper(occ_graph_dict[node_idx][0])))
            else:
                heapq.heappush(lowest_cost_action, (node_priority,
                                                    time_est,
                                                    Wrapper(occ_graph_dict[node_idx][0])))

        return

    # -------------------------------------------------------------------------
    # -- Callback functions
    # -------------------------------------------------------------------------
    def callback_map(self, msg):
        """ Callback function for map data
        """
        # callback function cannot return anything, so we use a global variable to store the data
        # https://stackoverflow.com/questions/37373211/update-the-global-variable-in-rospy
        # https://answers.ros.org/question/174485/return-a-value-from-a-callback-function/
        self.occgrid_msg = msg
        self.occ_map = msg

        self.resolution = self.occgrid_msg.info.resolution
        self.origin_x = self.occgrid_msg.info.origin.position.x
        self.origin_y = self.occgrid_msg.info.origin.position.y
        self.flag_map_read = True
        map_width = self.occgrid_msg.info.width
        map_height = self.occgrid_msg.info.height
        map_data = msg.data
        map_data_array = np.array(map_data)
        self.occ_map_data_array = map_data_array.reshape(map_height, map_width)

    def callback_avg_speed(self, msg):
        """ Subscriber callback function for robot average speed
        """
        avg_speed = msg.data
        if avg_speed <= 0.0:
            self.avg_speed = self.avg_speed_est
        else:
            self.avg_speed = avg_speed
        return

    def callback_time(self, msg):
        """ Subscriber callback function for robot time_remaining message
        """
        self.time_remaining = msg.data
        return

    def callback_odom(self, msg):
        """ Subscriber callback function for robot odom message
        """
        self.current_pose = msg.pose.pose
        self.cur_map_pose = self._odom_to_map_coordinate(self.current_pose)
        return

    # --------------------------------------------------------------------------
    # Functions to change coordinate frames
    # --------------------------------------------------------------------------
    def _map_to_odom_coordinate(self, map_coord) -> tuple:
        """ Change map to odometry coordinates
        """
        odom_x = map_coord[0] * self.resolution + self.origin_x
        odom_y = map_coord[1] * self.resolution + self.origin_y
        # -- return the odometry coordinate
        # TODO: maybe source of error (if error, then swap x and y)
        # return (odom_x, odom_y)
        return odom_y, odom_x

    def _odom_to_map_coordinate(self, odom):
        """ Change odometry to map coordinates
        ---------------------------------------------------
        map_coord = [100,200]
        calculation
        100 * 0.05 = 5 + (-10) = -5
        200 * 0.05 = 10 + (-10) = 0
        odom.x = 0
        odom.y = -5
        ---------------------------------------------------
        odom.x = 0, odom.y = -5
        map_coord_col = (0 - (-10)) / 0.05 = 10/0.05 = 200
        map_coord_row = (-5 -(-10)) / 0.05 = 5/0.05 = 100
        coordinate = (100,200)
        ---------------------------------------------------
        """
        map_coord_col = (odom.position.x - self.origin_x) // self.resolution
        map_coord_row = (odom.position.y - self.origin_y) // self.resolution

        return map_coord_row, map_coord_col

if __name__ == "__main__":
    ''' The parameters input are:
    1. Turn_360_at_start: Robot turns 360 degrees at the start of the exploration
    
    This makes sure that the initial scan explores a part of the map. This is 
    especially important if this code is paired up with robots that only have a
    sensor pointed towards a single direction instead of unidirectional lidar.
    '''
    if len(sys.argv) > 1:
        if sys.argv[1] == "False":
            print("Priority exploration without 360 turn")
            pea = PriorityExplorationAgent('/time_current',
                                           '/map',
                                           '/average_speed', False)
        else:
            print("Priority exploration with 360 turn")
            pea = PriorityExplorationAgent('/time_current', '/map', '/average_speed', True)
    else:
        print("Priority exploration with 360 turn")
        pea = PriorityExplorationAgent('/time_current', '/map', '/average_speed', True)
