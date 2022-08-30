#!/usr/bin/env python3

from __future__ import print_function

import rospy
import numpy as np
import scipy.misc
import os
import cv2

from turtlebot3_exploration.msg import Graph_nodes

def get_bounding_box(node, map_data_array, bb_side:int = 40, debug_print:bool = False)->int:
    center_coord = np.array(node.o)
    center_coord = np.round(center_coord)
    center_coord = center_coord.astype('int')
    if debug_print:
        print ("Bounding box: Center coord: ", center_coord)
    bb_side_half = bb_side//2 #use // to avoid getting float values
    left_top_coord = center_coord - bb_side_half
    right_bottom_coord = center_coord + bb_side_half
    # set offset as zero
    # -- if bounding box is below zeroes
    offset = 0
    offset_1 = -left_top_coord[0] if left_top_coord[0] < 0 else 0
    offset_2 = -left_top_coord[1] if left_top_coord[1] < 0 else 0

    if offset_1 > 0 or offset_2 > 0:
        left_top_coord[0] += offset_1
        left_top_coord[1] += offset_2
        right_bottom_coord[0] += offset_1
        right_bottom_coord[1] += offset_2

    # -- if bounding box is greater than size
    offset_1 = right_bottom_coord[0] - map_data_array.shape[0]
    offset_2 = right_bottom_coord[1] - map_data_array.shape[1]
    if offset_1 > 0 or offset_2 > 0:
        left_top_coord[0] -= offset_1
        left_top_coord[1] -= offset_2
        right_bottom_coord[0] -= offset_1
        right_bottom_coord[1] -= offset_2
    # -- fill the left bottom and right top coordinates a part of
    #    output coordinates
    #out_coords = [left_top_coord, right_bottom_coord]
    # -- create the bounding box image
    #bb_img = map_data_array[left_top_coord[0]:right_bottom_coord[0], left_top_coord[1]:right_bottom_coord[0]]

    # -- print debug
    if debug_print:
        print ("Left top Coordinate: ", left_top_coord)
        print ("Right bottom Coordinate: ", right_bottom_coord)
    return left_top_coord, right_bottom_coord
    
