#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64
import sys
import re
import cv2
from ground_truth_labels import *

np.set_printoptions(threshold=sys.maxsize, linewidth=3000)
'''
To calculate the exploration metrics, we would consider a ground truth map of 
the environment. 
Input:
1.  A map file showing the explored map of the environment.
2.  A current map from a subscriber.

Task:
1.  Calculate the number of occupancy grid cells explored
2.  Divide the number of cells explored by the number of explored cells in the 
    explored map.
    a. Read the map from the yaml file.
    


'''

# -- convert map coordinate to odometry coordinate
def _map_to_odom_coordinate(map_coord, resolution, origin_x, origin_y):
    odom_x = map_coord[0] * resolution + origin_x
    odom_y = map_coord[1] * resolution + origin_y
    # -- return the odometry coordinate
    #TODO: maybe source of error (if error, then swap x and y)
    #return (odom_x, odom_y)
    return(odom_y, odom_x)


class ExplorationMetrics:
    def __init__(self, map_msg):
        
        # -- load params for ground truth cells
        self.gt_label_count = {'corridor':0, 'large_room':0, 'small_room':0}
        self.gt_label_count['corridor'] = rospy.get_param('corridor_total')
        self.gt_label_count['large_room'] = rospy.get_param('large_room_total')
        self.gt_label_count['small_room'] = rospy.get_param('small_room_total')

        self.gt_free_cells = 0
        for label, count in self.gt_label_count.items():
            self.gt_free_cells += count
        # -- room type
        self.env_name = rospy.get_param('env_name')

        # -- get data
        self.callback_map(map_msg)
        
    
        
    def count_saved_map_free_cells(self, occ_map_array, threshold_free):
        """
        Input:
            occ_map_array: occupancy map array (numpy array)
            threshold_free: value above which the numpy array value is considered free
        """
        return np.count_nonzero(occ_map_array >= threshold_free)
    
    def count_callback_map_free_cells(self, map_data_array, threshold):
        #return np.count_nonzero(map_data_array <= threshold)
        return np.count_nonzero(map_data_array == 0)
    
    def callback_map(self, msg):
        res = msg.info.resolution
        map_width = msg.info.width
        map_height = msg.info.height
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        rospy.loginfo('seq: {}'.format(msg.header.seq))
        map_data = msg.data
        #rospy.loginfo('Map Data: shape: {}'.format(len(map_data)))
        # The map data is presented in row-major order, starting with (0,0).
        # Occupancy probabilities are in the range [0,100].  Unknown is -1.
        # -- if first run, calculate the ground truth labels of node
        
        # -- print the map_data
        map_data_array = np.array(map_data)
        map_data_array = map_data_array.reshape(map_height, map_width)

        label_count = self.count_explored_rooms(map_data_array, origin_x, origin_y, res)
        print ("Label count: ", label_count)

        labels_percent_explored = self.calculate_labels_percent_explored(label_count, self.gt_label_count)
        print ("labels_percent_explored: ", labels_percent_explored)
        
        # -- call the metric and publish the data
        unoccupied_cells_current = self.count_callback_map_free_cells(map_data_array, 1)
        rospy.loginfo('unoccupied cells: {}'.format(unoccupied_cells_current))
        
        # -- calculate the percentage of explored cells from gt_data
        percent_explored = float(unoccupied_cells_current)/float(self.gt_free_cells)
        
        # -- log info of percent explored
        rospy.loginfo('Percentage explored: {}'.format(percent_explored))

        # -- return the values
        self.label_count = label_count
        self.percent_explored =percent_explored
        self.labels_percent_explored = labels_percent_explored
    
    def calculate_labels_percent_explored(self, label_count, gt_label_count):
        percent_explored = {}
        for label, count in label_count.items():
            percent_explored[label] = float(count) / float(gt_label_count[label])
        return percent_explored

    def count_explored_rooms(self, map_data_array, origin_x, origin_y, resolution):
        rooms = {'corridor':0, 'large_room':0, 'small_room':0}
        # -- traverse through the map data array
        for row_idx in range(map_data_array.shape[0]):
            for col_idx in range(map_data_array.shape[1]):
                # -- if value is 0 (visited and unoccupied)
                if map_data_array[row_idx,col_idx] == 0:
                    # -- convert the map coordinate to odom coord
                    odom_coord = _map_to_odom_coordinate((row_idx, col_idx), resolution, origin_x, origin_y)
                    label = ""
                    if self.env_name == "circular_corridor":
                        label = get_gt_cc_env(odom_coord[0], odom_coord[1])
                    elif self.env_name == "straight_corridor":
                        label = get_gt_straight_corridor_env(odom_coord[0], odom_coord[1])
                    elif self.env_name == "branched_corridor" or \
                         self.env_name == "branched_corridor_real" or \
                         self.env_name == "branched_corridor_obstacle_real":
                        label = get_gt_branched_corridor_env(odom_coord[0], odom_coord[1])
                    rooms[label] += 1
        return rooms

