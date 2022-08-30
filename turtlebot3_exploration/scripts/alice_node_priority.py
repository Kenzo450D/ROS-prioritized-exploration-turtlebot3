#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from PIL import Image
import math
import uuid
from collections import defaultdict
# -- for saving to dataset
import os.path
# -- END: for saving to dataset

from turtlebot3_exploration.msg import Graph_nodes
from laser_emulation import LaserFeature
from bounding_box_nodes import get_bounding_box
from bob_node_priority import assign_frontier_priority_lidar_emulate_threshold
from bresenham_line import _createLineIterator
from ground_truth_labels import *

np.set_printoptions(threshold=np.inf)


# -- convert map coordinate to odometry coordinateg
def _map_to_odom_coordinate(map_coord, resolution, origin_x, origin_y):
    odom_x = map_coord[0] * resolution + origin_x
    odom_y = map_coord[1] * resolution + origin_y
    # -- return the odometry coordinate
    # TODO: maybe source of error (if error, then swap x and y)
    # return (odom_x, odom_y)
    return (odom_y, odom_x)


# -- get the ground truth label
def get_gt_cc_env(x_coord, y_coord):
    if abs(x_coord) > 3.2:
        return "small_room"
    elif abs(y_coord) > 3.2:
        return "small_room"
    if abs(x_coord) < 1.6 and abs(y_coord) < 1.6:
        return "large_room"
    else:
        return "corridor"


def get_gt_sc_env(x_coord, y_coord):
    # -- corridor
    if abs(y_coord) < 0.85:
        if x_coord < 12 and x_coord > -7:
            return "corridor"
        else:
            return "large_room"
    else:
        if x_coord < 6.4:
            return "small_room"
        else:
            return "large_room"


def get_gt_u_env(x_coord, y_coord):
    # -- small room
    if abs(y_coord) > 6.45:
        return "small_room"
    if x_coord < -2.9:
        if abs(y_coord) < 0.8:
            return "corridor"
        else:
            return "small_room"
    elif x_coord > -1.3:
        if y_coord < 4.9 and y_coord > 1:
            return "large_room"
        elif y_coord > -4.9 and y_coord < -1.075:
            return "large_room"
        elif y_coord >= - 1.0725 and y_coord <= 1:
            return "small_room"
        else:
            return "corridor"
    else:
        return "corridor"


# -- corner detection
def find_corners(occ_map_image, debug_print=False):
    # -- convert image to format used in pgm file
    gray = occ_map_image

    # -- save the file
    # cv2.imwrite('test_image_Jan26.png', gray)  # DEBUG

    gray_float = np.float32(gray)
    # rospy.loginfo("Type of gray: {}".format(gray.dtype))

    # cornerHarris(input_image, 
    #       Neighborhood size 
    #       Aperture parameter for the Sobel operator.
    #       Harris detector free parameter
    # )
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.cornerHarris(gray, 7, 7, 0.1)

    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # The first centroid is the center of the image
    centroids = np.delete(centroids, 0, 0)

    # -- Debug ros loginfo
    if debug_print:
        count = 0
        for c in centroids:
            print("Centroid: {} :: {}".format(count, c))
            count += 1
        print("-" * 50)
    # -- return
    return centroids


# -----------------------------------------------------------------------------
# Function: calculate_line_relations
# -----------------------------------------------------------------------------
# Calculates parallel lines, orthonogal lines and returns the parallel 
# lines and orthogonal lines. For parallel lines, it returns the distance
# between other parallel lines.
# -----------------------------------------------------------------------------
def calculate_line_relations(line_end_points, lines, slopes, threshold=0.175 * 2, debug_print=False):
    ''' Calculates parallel lines, orthonogal lines and returns the parallel 
    lines and orthogonal lines. For parallel lines, it returns the distance
    between other parallel lines.

    Output:
     parallel_lines_mat: A nxn array, (n = number of lines), where if the value is np.inf
                     then the lines are not parallel to each other. Any other value,
                     and the lines are parallel to each other.
     orthogonal_lines_mat: A nxn array, where each element is zero if the two lines are orthogonal
                       to each other, and 1 if the lines intersect.


    '''
    # https://en.wikipedia.org/wiki/Distance_between_two_parallel_lines#Formula_and_proof
    # -- initialize matrix to store relation between parallel lines
    n_lines = len(line_end_points)
    parallel_lines_mat = np.ones((n_lines, n_lines)) * np.inf
    parallel_lines = defaultdict(lambda: [])
    orthogonal_lines_mat = np.zeros((n_lines, n_lines))
    orthogonal_lines = defaultdict(lambda: [])
    intersecting_lines_mat = np.zeros((n_lines, n_lines))
    if debug_print:
        print("-" * 50)
        print("Function: Calculate Line Relations: ")
    for l_idx in range(0, n_lines):
        for l2_idx in range(l_idx + 1, n_lines):
            # -- check for match in slope
            slope_diff = abs(slopes[l_idx] - slopes[l2_idx])
            slope_diff_2 = np.pi - slope_diff
            slope_diff = slope_diff if slope_diff < slope_diff_2 else slope_diff_2
            if debug_print:
                print("\tLine 1: ", line_end_points[l_idx], "\tSlope: ", slopes[l_idx])
                print("\tLine 2: ", line_end_points[l2_idx], "\tSlope: ", slopes[l2_idx])
                print("\tSlope Diff: ", slope_diff)
            if slope_diff < threshold:
                if debug_print: print("\tLines are parallel...\tCalculating distance between lines")
                # -- represent the lines as the format of y = mx + b
                # Calculating distance between two lines using 
                # https://en.wikipedia.org/wiki/Distance_between_two_parallel_lines#Formula_and_proof
                # is not useful as the parameters becomes very sensitive to slope values. 
                # instead use
                # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
                x1 = line_end_points[l_idx][0][0]
                y1 = line_end_points[l_idx][0][1]

                x2 = line_end_points[l_idx][1][0]
                y2 = line_end_points[l_idx][1][1]

                x0 = line_end_points[l2_idx][1][0]
                y0 = line_end_points[l2_idx][1][1]
                if debug_print:
                    print("\tLine 1:")
                    print("\t\tx1: ", x1)
                    print("\t\ty1: ", y1)
                    print("\tLine 1:")
                    print("\t\tx2: ", x2)
                    print("\t\ty2: ", y2)
                    print("\tLine 2: ")
                    print("\t\tx0: ", x0)
                    print("\t\ty0: ", y0)

                dist_num = np.abs(((x2 - x1) * (y1 - y0)) - ((x1 - x0) * (y2 - y1)))
                dist_den = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                dist = dist_num / dist_den

                # -- calculate the distance between the lines
                if debug_print:
                    print("\tdist: numerator: ", float(dist_num))
                    print("\tdist: denominator: ", dist_den)
                    print("\tDistance between lines: ", dist)

                parallel_lines_mat[l_idx, l2_idx] = dist
                parallel_lines_mat[l2_idx, l_idx] = dist
                parallel_lines[l_idx].append((l2_idx, dist))
                parallel_lines[l2_idx].append((l_idx, dist))

            if slope_diff > ((np.pi / 2) - threshold) and slope_diff < ((np.pi / 2) + threshold):
                if debug_print:
                    print("Lines are orthogonal")
                # -- lines are orthogonal to one another
                orthogonal_lines_mat[l_idx, l2_idx] = 1
                orthogonal_lines_mat[l2_idx, l_idx] = 1
                orthogonal_lines[l_idx].append(l2_idx)
                orthogonal_lines[l2_idx].append(l_idx)

            if check_intersecting_walls(line_end_points[l_idx], line_end_points[l2_idx], lines[l_idx], lines[l2_idx]):
                if debug_print:
                    print("Lines intersect!")
                intersecting_lines_mat[l_idx, l2_idx] = 1
                intersecting_lines_mat[l2_idx, l_idx] = 1

    if debug_print: print("Function: Calculate Line Relations: End of function")
    # -- return the two matrices
    return parallel_lines_mat, orthogonal_lines_mat, intersecting_lines_mat, parallel_lines, orthogonal_lines


def threshold_obstacles_occ_map(map_array, obstacle_threshold, obstacle_value):
    image_th = np.zeros(np.shape(map_array), dtype=np.uint8)
    image_th[np.where(map_array > 30)] = 255
    # cv2.imwrite("image_threshold_alice.png", image_th)
    return image_th


def dilate_image(map_array, filter_size=3):
    """ Dilate the image to accomodate thicker obstacles. 
    """
    kernel = np.ones((filter_size, filter_size), np.uint8)
    dilated_img = cv2.dilate(map_array, kernel, iterations=1)
    # cv2.imwrite("image_dilated_threshold_alice.png", dilated_img)
    return dilated_img


def get_line_slope(p1, p2):
    ''' Get slope of the line between p1 and p2

    Input:
        p1: (x,y) coordinate
        p2: (x,y) coordinate
    
    Output:
        m: (float) in radians
    '''
    m = math.atan2((float(p2[1]) - float(p1[1])), (float(p2[0]) - float(p1[0])))
    # if m < 0:
    #     m += math.pi # slope is correct either way. (A->B or B->A)
    return m


def print_image(img):
    print("*", end="")
    idx = 0
    for i in range(0, img.shape[0]):
        print(idx, end="")
        idx += 1
        if idx > 9:
            idx = 0
    print()
    idx = 0
    for i in range(0, img.shape[0]):
        print(idx, end="")
        for j in range(0, img.shape[1]):
            if img[i, j] == 255:
                print("#", end="")
            elif img[i, j] == 0:
                print(" ", end="")
        print()
        idx += 1
        if idx > 9:
            idx = 0
    return


# ------------------------------------------------------------------------------
# Function: check_lines_between_points
# ------------------------------------------------------------------------------
# Calculates line and slopes between corner coordinates in a thresholded dilated
# image
# ------------------------------------------------------------------------------
def check_lines_between_points(corner_coords, img, obstacle_val=255, debug_print=False):
    ''' Checks if there are obstacles between a pair of corner coordinates
    Input:
        coords: set of coordinates (x,y) as a numpy array, nrows = number of elements
        img: the thresholded image where the obstacles are white everything else is black
        obstacle_val: the pixel value of obstacles. Default is 255
    Output:
        line_end_points: A tuple of two corner points that make a line of the obstacle
        lines: A list of coordinates that make the obstacle.
        slopes: A list of slopes for each of the lines.
    '''
    print("Check lines between corners")
    coords = corner_coords
    # Switch first and second col as opencv goes column first
    # coords[:,[1,0]] = coords[:,[0,1]]
    coords = coords.astype('int')
    unique_val = np.unique(img)
    # print ("Function: CHECK LINE BETWEEN POINTS: UNIQUE VALUES: ", unique_val)
    line_end_points = []  # to store lines
    lines = []
    slopes = []
    # if debug_print:
    #     print ("Check lines between points:")
    #     print ("\t Unique values in img: ")
    #     print ("\t{}".format(np.unique(img)))
    #     print_image(img)
    # -- for loop to permute between all centroid locations:
    for i in range(0, len(coords)):
        for j in range(i + 1, len(coords)):
            if debug_print:
                print("Function: check line between points :: Line between: ", coords[i], coords[j])
                print("Image shape: ", img.shape)
                print("Coordinates: {}({}) and {}({})".format(i, coords[i], j, coords[j]))
                print("Image values at coordinate {}".format(tuple(coords[i])))
                print("Value of image i: {}".format(img[coords[i][0], coords[i][1]]))
                print("Image values at coordinate {} :: {}".format(coords[j], img[coords[j][0], coords[j][1]]))
            # -- draw a line between two points
            itbuf = _createLineIterator(coords[i], coords[j])
            # -- get pixel values at coordinates
            n_pixel_obstacles = 0
            for c in itbuf:
                if img[tuple(c)] == obstacle_val:
                    n_pixel_obstacles += 1
            if debug_print:
                print("\tPercent Coverage: ", (float(n_pixel_obstacles) / float(len(itbuf))))
            if n_pixel_obstacles > 0.95 * len(itbuf):
                if debug_print: print("Function: found line between points :: Line between: ", coords[i], coords[j])
                # -- get the slope
                m = get_line_slope(coords[i], coords[j])
                if debug_print: print("\tSlope: ", m)
                if m < 0:
                    m += math.pi
                    line_end_points.append((coords[j], coords[i]))
                    lines.append(np.flip(itbuf, 0))
                else:
                    # -- store the end points and line pixels
                    line_end_points.append((coords[i], coords[j]))
                    lines.append(itbuf)
                # -- calculate slope
                slopes.append(m)
    # print "Lines: "
    # print lines
    # print ("Calculated lines between corners")
    # print ("*" *50)
    return line_end_points, lines, slopes


def draw_centroids_point(image, coords):
    """ Colors the coordinates
    Input:
    image: np array for image
    coords: n x 2 array for coordinates of circles. 
    """
    print("Number of centroids: ", coords.shape[0])
    color = (255, 0, 0)  # BGR
    coords = coords.astype('uint8')
    for i in range(0, coords.shape[0]):
        c = tuple(coords[i, :])
        # image[c[1]-1: c[1]+1, c[0]-1:c[0]+2,:] = color
        image[c[1], c[0]] = color

    return image


def save_cropped_image(node_idx, crop_map_data_th_dl, corner_coords, postfix=""):
    im_out = np.zeros((crop_map_data_th_dl.shape[0], crop_map_data_th_dl.shape[1], 3), dtype=np.uint8)
    im_data = np.copy(crop_map_data_th_dl)
    im_data = np.where(im_data < 0, 0, im_data)
    im_data = np.where(im_data > 99, 255, im_data)
    im_out[:, :, 0] = np.copy(crop_map_data_th_dl)
    im_out[:, :, 1] = np.copy(crop_map_data_th_dl)
    im_out[:, :, 2] = np.copy(crop_map_data_th_dl)
    image_circles = draw_centroids_point(im_out, corner_coords)
    outfilename = "Node_" + str(node_idx) + postfix + ".png"
    # cv2.imwrite(outfilename, image_circles)
    im = Image.fromarray(im_out)
    im.save(outfilename)
    return


def get_ground_truth_label(node_map_coord, origin_x, origin_y, resolution):
    odom_coord = _map_to_odom_coordinate(node_map_coord, resolution, origin_x, origin_y)
    # -- get the labels 
    env_name = rospy.get_param('env_name')
    label = ""
    if env_name == "circular_corridor":
        label = get_gt_cc_env(odom_coord[0], odom_coord[1])
    elif env_name == "straight_corridor":
        label = get_gt_sc_env(odom_coord[0], odom_coord[1])
    elif env_name == "branched_corridor":
        label = get_gt_u_env(odom_coord[0], odom_coord[1])
    print("Ground truth label: ", label)
    return label


def get_gt_priority(node_map_coord, origin_x, origin_y, resolution):
    label = get_ground_truth_label(node_map_coord, origin_x, origin_y, resolution)
    print(label)
    # priority_dict = {3.0:"corridor", 2.0:"large room", 1.0:"small room"}
    if label == "small_room":
        return 3.0
    if label == "large_room":
        return 2.0
    if label == "corridor":
        return 1.0


def save_cropped_image_dataset(node_idx, node_map_coord, crop_map, resolution, origin_x, origin_y):
    # -- saves images to a particular directory 
    im_out = np.zeros((crop_map.shape[0], crop_map.shape[1], 3))
    im_out[:, :, 0] = np.copy(crop_map)
    im_out[:, :, 1] = np.copy(crop_map)
    im_out[:, :, 2] = np.copy(crop_map)
    # -- if saving for dataset
    outfilename_prefix = "/home/kenzo/node_dataset/"
    path, dirs, files = next(os.walk(outfilename_prefix))
    file_count = len(files)
    # -- get ground truth label for the node
    # -- get param of environment
    odom_coord = _map_to_odom_coordinate(node_map_coord, resolution, origin_x, origin_y)

    # -- get the labels 
    label = get_ground_truth_label(node_map_coord, origin_x, origin_y, resolution)
    print("Label: ", label)
    outfilename = outfilename_prefix + "Node-" + label + "-" + str(file_count) + "-" + str(node_idx) + ".png"
    print("outfilename: ", outfilename)
    cv2.imwrite(outfilename, im_out)
    return


def pad_image(img, padding_size):
    """ Pad image with zeros around it.
    """
    # print ("Pad image")
    # print ("Shape of original Image: ", np.shape(img))
    nrows = img.shape[0] + padding_size * 2
    ncols = img.shape[1] + padding_size * 2
    img_padded = np.zeros((nrows, ncols), dtype=img.dtype)
    # print ("Shape of padded image: ", img_padded.shape)
    # print ("Index Row: ", padding_size, " : ", nrows-padding_size)
    # print ("Index Col: ", padding_size, " : ", ncols-padding_size)
    img_padded[padding_size: nrows - padding_size, padding_size: ncols - padding_size] = img
    # print ("Image padded successfully!")
    return img_padded


def check_intersecting_walls(lep_1, lep_2, l1, l2, end_threshold=8, threshold=8, debug_print=False):
    """ Calculates if the walls are interesecting or not
    Input:
        lep_1: Line end points for line 1
        lep_2: Line end points for Line 2
        l1: line points for line 1
        l2: line points for line 2
        end_threshold: threshold for lines at the end
        threshold: threshold for points not at the end
    Output:
        True if any of the end points are close to one another
        False: if none of the end points are close to one another.
    """
    # -- checks if the end points are close to one another
    if debug_print:
        print("Function: Check intersecting walls: Threshold: ", threshold)
        print("\tWall 1: ", lep_1)
        print("\tWall 2: ", lep_2)
    if euclidean_dist(lep_1[0], lep_2[0]) < end_threshold:
        if debug_print: print("\tClose: ", lep_1[0], " and ", lep_2[0])
        return True
    elif euclidean_dist(lep_1[0], lep_2[1]) < end_threshold:
        if debug_print: print("\tClose: ", lep_1[0], " and ", lep_2[1])
        return True
    elif euclidean_dist(lep_1[1], lep_2[0]) < end_threshold:
        if debug_print: print("\tClose: ", lep_1[1], " and ", lep_2[0])
        return True
    elif euclidean_dist(lep_1[1], lep_2[1]) < end_threshold:
        if debug_print: print("\tClose: ", lep_1[1], " and ", lep_2[1])
        return True
    # -- check for points in the line
    # print ("Function: Check intersecting walls: Check in individual line elements: ")
    # print ("\tLine Points L1: ", l1)
    # print ("\tLine Points L2: ", l2)
    min_dist = np.inf
    for lp1 in l1:
        for lp2 in l2:
            eu_dist = euclidean_dist(lp1, lp2)
            # print ("Distance between {} and  {} :: {}".format(lp1, lp2, eu_dist)) 
            if eu_dist < min_dist:
                min_dist = eu_dist
            if eu_dist < threshold:
                # print ("Close: {} and {}".format(lp1,lp2))
                return True
    if debug_print: print("Check interesecting walls: Minimum Distance: ", min_dist)
    return False


def euclidean_dist(p1, p2):
    dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


def get_parallel_wall_relations(parallel_lines, intersecting_lines_mat, debug_print=False):
    # most_parallel_idx = 0
    most_parallel_lines = 0
    walls_with_doors = []
    walls_of_corridor = []
    walls_of_small_room = []
    # -- go through the parallel lines dictionary
    for l_key, l_val in parallel_lines.items():
        if debug_print:
            print("line key: ", l_key)
            print("line_val: ", l_val)
        for l_item in l_val:
            # l_item[0]: index of another line
            # l_item[1]: distance from another line
            if debug_print:
                print("l_item[0]: ", l_item[0])
                print("l_item[1]: ", l_item[1])
            if l_item[1] < 8:
                if debug_print:
                    print("get_parallel_wall_relations: distance is less than 8: ")
                    print("Line_1: {}\tLine_2: {}".format(l_key, l_item[0]))
                    print("Intersecting matrix: {}".format(intersecting_lines_mat[l_key, l_item[0]]))
                if intersecting_lines_mat[l_key, l_item[0]] == 0:
                    walls_with_doors.append((l_key, l_item[0]))
            elif l_item[1] < 1.9 / 0.05:
                walls_of_corridor.append((l_key, l_item[0]))
            elif l_item[1] < 3.2 / 0.05:  # to avoid inf values.
                walls_of_small_room.append((l_key, l_item[0]))
        if len(l_val) > most_parallel_lines:
            most_parallel_lines = len(l_val)
            # most_parallel_idx = l_key
    if debug_print: print("-- End of parallel lines dictionary")
    # -- return parallel line relations
    return most_parallel_lines, walls_with_doors, walls_of_corridor, walls_of_small_room


def less_than_four_lines(lines, parallel_lines, parallel_lines_mat, orthogonal_lines_mat, intersecting_lines_mat,
                         debug_print=False):
    priority_small_room = 3.0  # priority should be lowest first
    priority_large_room = 2.0
    priority_corridor = 1.0

    if len(lines) == 0:
        print("Number of lines = 0")
        print("Large room, no walls found")
        return priority_large_room

    # -- if there is only one line detected, then there may be just one wall
    if len(lines) == 1:
        print("Number of lines = 1")
        print("Large room: only one wall found")
        return priority_large_room

    # -- if number of lines is 2
    elif len(lines) == 2:
        most_parallel_lines, walls_with_doors, walls_of_corridor, walls_of_small_room = get_parallel_wall_relations(
            parallel_lines, intersecting_lines_mat)
        # -- if the two lines are orthogonal
        if orthogonal_lines_mat[0, 1] == 1:
            print("Large room: Only orthogonal walls found")
            return priority_large_room
        if parallel_lines_mat[0, 1] != np.inf:
            if parallel_lines_mat[0, 1] < 0.4 / 0.05:
                # if there is only one door detected, then there may be just one 
                # wall (with door)
                print("Large Room: Two walls divided by one door")
                return priority_large_room
            elif len(walls_of_corridor) > 0:  # parallel_lines_mat[0,1] < (2/0.05):
                print("Wall distance: ", parallel_lines_mat[0, 1])
                print("2 Lines found: Distant parallel walls less than 2 meters")
                return priority_corridor
            else:
                print("2 Lines found: parallel walls found beyond 2 meter apart")
                return priority_small_room
        return priority_large_room
    # -- if number of lines is 3
    elif len(lines) == 3:
        print("Number of lines = 3")
        # if there is a wall with doors and there is a orthogonal wall, it may 
        # be a small room or a large room. As of now, lets consider that to be
        # a large room. As a small room might have two walls visible as well.
        most_parallel_lines, walls_with_doors, walls_of_corridor, walls_of_small_room = get_parallel_wall_relations(
            parallel_lines, intersecting_lines_mat)

        len_walls_with_doors = len(walls_with_doors) // 2
        len_walls_of_corridor = len(walls_of_corridor) // 2
        len_walls_of_small_room = len(walls_of_small_room) // 2

        most_orthogonal_idx = np.argmax(np.sum(orthogonal_lines_mat, axis=1))
        most_orthogonal_lines = np.max(np.sum(orthogonal_lines_mat, axis=1))

        most_intersections = np.max(np.sum(intersecting_lines_mat, axis=1))

        if debug_print:
            print("Most parallel lines: ", most_parallel_lines)
            print("Walls with doors: ", walls_with_doors)
            print("Walls with corridors: ", walls_of_corridor)
            print("Walls of small room: ", walls_of_small_room)
            print("Length: Walls with doors: ", len_walls_with_doors)
            print("Length: Walls with corridors: ", len_walls_of_corridor)
            print("Length: Walls of small room: ", len_walls_of_small_room)
            print("Most ORthogonal Lines: ", most_orthogonal_lines)
            print("Most Intersections: ", most_intersections)

        intersecting_index = []
        for i in range(0, len_walls_of_corridor * 2):
            # print ("index: ", i)
            l_idx_1 = walls_of_corridor[i][0]
            l_idx_2 = walls_of_corridor[i][1]
            # print ("Corridor wall index: {} and {}".format(l_idx_1, l_idx_2))
            for int_idx in range(0, len(lines)):
                # print ("Intersecting lines mat[{},{}]: {}".format(int_idx,l_idx_1, intersecting_lines_mat[int_idx,l_idx_1] ))
                # print ("Intersecting lines mat[{},{}]: {}".format(int_idx,l_idx_2, intersecting_lines_mat[int_idx,l_idx_1] ))
                if intersecting_lines_mat[int_idx, l_idx_1] and intersecting_lines_mat[int_idx, l_idx_2]:
                    # there is a line interesecting two parallel lines
                    intersecting_index.append((int_idx, (l_idx_1, l_idx_2)))
                    if debug_print: print(
                        "Line: {} intersects two walls of corridor: {}".format(int_idx, walls_of_corridor[i]))

        if len(intersecting_index) > 0:
            print("Corridor distanced parallel lines intersected by an obstacle!")
            print("Small room identified!")
            return priority_small_room

        if len_walls_of_corridor == 1:
            return priority_corridor
        if len_walls_of_small_room == 1:
            return priority_small_room
        if most_parallel_lines == 1:
            if most_orthogonal_lines == 2 and most_intersections == 1:
                if len_walls_with_doors > 0:
                    print("Large Room: 2 Parallel Lines, One wall with door, one line orthogonal to 2, intersections 1")
                    return priority_large_room
                else:
                    print("Corridor: Parallel lines 2, orthogonal_lines = 1, intersections = 1")
                    return priority_corridor

        if most_parallel_lines == 2:
            # -- check the distance between the lines
            if len_walls_of_small_room > 0 and len_walls_with_doors > 0:
                print("Small Room: Walls of small room > 0 and walls with doors > 0")
                return priority_small_room
            if len_walls_of_corridor > 0 and len_walls_with_doors > 0:
                print("Corridor: Walls of corridor > 0 and walls with doors > 0")
                return priority_corridor

        if most_parallel_lines == 2:
            print("Corridor: parallel_lines = 2")
            return priority_corridor

        if len_walls_with_doors == 1 and len_walls_of_corridor == 0 and len_walls_of_small_room == 0:
            return priority_large_room

    return -1


def get_node_priority(crop_map_data_pad,
                      line_end_points,
                      lines,
                      slopes,
                      debug_print=False):
    parallel_lines_mat, orthogonal_lines_mat, intersecting_lines_mat, parallel_lines, orthogonal_lines = calculate_line_relations(
        line_end_points, lines, slopes)
    if debug_print:
        print("Intersecting Lines Mat: ")
        print(intersecting_lines_mat)
        print("Parallel lines matrix")
        print(np.round(parallel_lines_mat, 2))
        print(" -- End of parallel lines matrix")
    # -- check the distance between the parallel lines
    # -- check if there are pairs of parallel lines
    # -- check if orthogonal lines are close to parallel lines
    # -- add more parameters as required.
    priority = 1.0
    """ 
    Input:
        parallel_lines_mat: matrix float values. size: n_lines x n_lines
                        value is np.inf if the lines are not parallel to each other
                        else, the value is the distance between the lines
        orthogonal_lines_mat: numpy matrix with float values. size: n_lines x n_lines
                          the values are 0 is the lines are not orthogonal.
    """

    """
    Rules:
    1.  Corridor:
     a. Parallel Lines:
        i.  If the node is not a function of corridor and door, then there would
            be just a pair of lines as the two walls of the corridor.
        ii. If the node is at a junction, with a door on one side, then there would
            be three parallel lines for the walls. Two walls on either side of the 
            door and one wall on the opposite side.
        iii.If there are are two doors, then there would be four lines parallel to
            each other, two of them for one wall (either side of a door) and two of
            them on the opposide side, (either side of the other door).
     b. Orthogonal Lines:
        iv. A part of the rooms may be visible for conditions ii and iii
            respectively. For these rooms, a part of the walls of the 
            room would be visible. These walls are considered as lines.
            These lines would be orthogonal to the walls of the corridor. 
            There is no specific way to ignore the walls of the rooms.
     c. Challenges:
        v.  At the start of the corridor, which has a end to one side, it might be
            misclassified as a small room.
            For example: https://i.ibb.co/5XDHgH9/Screenshot-20210201-021131.png
                Node 5, might be classified as a small room.
        
    2.  Small Room:
        If the area is completely explored, then we can see part of the 3 walls
        of the room as well as the walls of the corridor. 
        However, if the room is partly explored, then we can see the 
     a. Parallel Lines:
        i.  The walls of the room are parallel to one another.
        ii. The walls on the either side of the room will be parallel to one 
            another.
     b. Orthogonal Lines:
        iii.The walls from the corridor will be perpendicular to the lines of
            the walls of the small room.
     c. Challenges:
        iv. The small room might get mixed with the corridor as the the features
            detected might be similar.
    
    3.  Large Room:
     a. Parallel Lines:
        i. The entrance to the room might have parallel lines
     b. No Lines:
        ii. The node may not have any lines or obstacles near it. 
    """
    if debug_print:
        print("Function: get_node_priority:")
        print("\tget_node_priority:: Number of lines: ", len(lines))
        print("\tget_node_priority:: Number of parallel_lines: ")
        print("\t\t", dict(parallel_lines))
        print("\tget_node_priority:: Orthogonal Lines")
        print("\t\t", dict(orthogonal_lines))
    priority_small_room = 3.0  # priority should be lowest first
    priority_large_room = 2.0
    priority_corridor = 1.0
    priority_node = 0.0
    # -- tests for large room
    # ---- first test : no obstacles
    # the unique values in the image are -1, 0 and 100
    # 100 are obstacles, 0 is free space.
    # -- if there are no obstacles, then it is a large room
    if np.count_nonzero(crop_map_data_pad == 100) == 0:
        if debug_print: print("Large room: no obstacles found!")
        return priority_large_room
    # -- if the number of lines is greater than 3
    if len(lines) < 4:
        print("Number of lines < 4")
        priority_val = less_than_four_lines(lines, parallel_lines, parallel_lines_mat, orthogonal_lines_mat,
                                            intersecting_lines_mat)
        return priority_val
    else:
        print("Number of lines >= 4")
        len_orthogonal_walls = 0
        intersecting_walls = 0
        most_parallel_lines, walls_with_doors, walls_of_corridor, walls_of_small_room = get_parallel_wall_relations(
            parallel_lines, intersecting_lines_mat)
        # print ("Most Parallel Index: ", most_parallel_idx)
        len_walls_with_doors: int = len(walls_with_doors) // 2
        len_walls_of_corridor: int = len(walls_of_corridor) // 2
        len_walls_of_small_room: int = len(walls_of_small_room) // 2
        if debug_print:
            print("Most parallel lines: ", most_parallel_lines)
            print("Walls with doors: ", walls_with_doors)
            print("Walls with corridors: ", walls_of_corridor)
            print("Walls of small room: ", walls_of_small_room)
            print("Length: Walls with doors: ", len_walls_with_doors)
            print("Length: Walls with corridors: ", len_walls_of_corridor)
            print("Length: Walls of small room: ", len_walls_of_small_room)

        if len_walls_with_doors >= 2:
            print("Number of walls with doors = 2")
            if len_walls_of_corridor >= 2:
                print("walls of corridor >= 2")
                print("Node priority: corridor")
                return priority_corridor

        # -- check if distances are corridor but there is a line intersecting them
        if debug_print:
            print("Calculating lines intersecting lines that are parallel")
            print("Intersecting Lines Mat:")
            print(intersecting_lines_mat)
        intersecting_index = []
        for i in range(0, len_walls_of_corridor * 2):
            # print ("index: ", i)
            l_idx_1 = walls_of_corridor[i][0]
            l_idx_2 = walls_of_corridor[i][1]
            # print ("Corridor wall index: {} and {}".format(l_idx_1, l_idx_2))
            for int_idx in range(0, len(lines)):
                # print ("Intersecting lines mat[{},{}]: {}".format(int_idx,l_idx_1, intersecting_lines_mat[int_idx,l_idx_1] ))
                # print ("Intersecting lines mat[{},{}]: {}".format(int_idx,l_idx_2, intersecting_lines_mat[int_idx,l_idx_1] ))
                if intersecting_lines_mat[int_idx, l_idx_1] and intersecting_lines_mat[int_idx, l_idx_2]:
                    # there is a line interesecting two parallel lines
                    intersecting_index.append((int_idx, (l_idx_1, l_idx_2)))
                    if debug_print: print(
                        "Line: {} intersects two walls of corridor: {}".format(int_idx, walls_of_corridor[i]))

        if len(intersecting_index) > 0:
            print("Corridor distanced parallel lines intersected by an obstacle!")
            print("Small room identified!")
            return priority_small_room

        if len_walls_of_corridor > len_walls_of_small_room:
            print("More corridor wall differences found!")
            return priority_corridor

        if len_walls_of_small_room >= 1:
            print("Found walls of small room")
            return priority_small_room

        if len_walls_of_corridor >= 1:
            print("Found walls of corridor")
            return priority_corridor

        if len_walls_with_doors >= 2:
            if len(walls_of_corridor) == 0 and len(walls_of_small_room) == 0:
                return priority_large_room

        if debug_print:
            print("#" * 100)
            print("Node priority could not be determined!")
            print("Number of intersecting line pairs", np.count_nonzero(intersecting_lines_mat) // 2)
            print("#*" * 50)
    return priority


def assign_frontier_alice(graph_nodes,
                          map_data_array,
                          offset_size=15,
                          laser_sensor_size=80,
                          lidar_threshold=0.05,
                          bb_side=120,
                          debug_print=False):
    """ Priority assignment for frontier nodes by first emulating lidar sensor,
    and then calculating the parallel and orthogonal lines to calculate whether
    the node is corridor, small room or large room.
    """
    if debug_print:
        print("Assign frontier Alice!")
    # -- initialize
    obstacle_threshold = 30
    obstacle_value = 255

    # -- identify the frontier nodes
    graph_nodes = assign_frontier_priority_lidar_emulate_threshold(graph_nodes, map_data_array, offset_size,
                                                                   laser_sensor_size, lidar_threshold)

    """
    Currently the graph nodes that are marked with priority as 1 are re-evaluated
    by considering a bounding box around them in the occupancy map.
    """

    # -- threshold the map
    map_data_th = threshold_obstacles_occ_map(map_data_array, obstacle_threshold,
                                              obstacle_value)  # makes obstacle values 255
    # -- dilate the thresholded image
    map_data_th_dl = dilate_image(map_data_th, 4)
    # dilating the image more would shift the corners more than necessary. Instead dilate after finding corners. 
    map_data_tl_xdl = dilate_image(map_data_th, 8)  # xdl : extra large dilation

    # -- debug save the dilated image
    # cv2.imwrite('dilated_img_ros.png', map_data_th_dl)

    # TODO: Remove the check priority for node when priority is not equals to zero when testing for

    # -- get bounding boxes
    for node in graph_nodes:
        if node.priority > 0:  # node is frontier node
            # -- get bounding box coordinates
            lt_coord, rb_coord = get_bounding_box(node, map_data_array, bb_side)

            # -- crop the thresholded image and dilated image to calculate 
            crop_map_data_th_dl = map_data_th_dl[lt_coord[0]:rb_coord[0], lt_coord[1]:rb_coord[1]]
            crop_map_data_th_xdl = map_data_tl_xdl[lt_coord[0]:rb_coord[0], lt_coord[1]:rb_coord[1]]
            crop_map_data = map_data_array[lt_coord[0]:rb_coord[0], lt_coord[1]:rb_coord[1]]

            # -- to make sure we get corners from the edges of the image, we add padding to the image
            padding_size = 4  # on each side
            crop_map_data_th_dl_pad = pad_image(crop_map_data_th_dl, padding_size)
            crop_map_data_th_xdl_pad = pad_image(crop_map_data_th_xdl, padding_size)
            crop_map_data_pad = pad_image(crop_map_data, padding_size)

            # -- get the centroid from the thresholded image
            corner_coords_opencv = find_corners(crop_map_data_th_dl_pad)

            # -- while not working with opencv, the indices of coordinates
            # must be flipped to correctly work with the images.
            # -- swap the centroid locations as opencv uses column first representation
            corner_coords = np.copy(corner_coords_opencv)
            corner_coords[:, 1] = corner_coords_opencv[:, 0]
            corner_coords[:, 0] = corner_coords_opencv[:, 1]

            # -- save the cropped image
            # save_cropped_image(node.id, crop_map_data_pad, corner_coords_opencv)
            # save_cropped_image(node.id, crop_map_data_th_dl_pad, corner_coords_opencv, "th_dl")
            # save_cropped_image(node.id, crop_map_data_th_xdl_pad, corner_coords_opencv, "th_xdl")

            # -- save for dataset
            # save_cropped_image_dataset(node.id, node.pts, crop_map_data_pad, lidar_threshold, origin_x, origin_y)

            print("node_idx: {}:: priority: ".format(node.id), end="")
            # node_priority_gt = get_gt_priority(node.pts,origin_x, origin_y, lidar_threshold)
            # print ("node priority ground truth: ", node_priority_gt)

            # TODO: Important: Remove this if and reinstate if at start of for loop
            # if node.priority >0:
            #     node.priority = node_priority_gt

            # -- check for lines between corners
            line_end_points, lines, slopes = check_lines_between_points(corner_coords, crop_map_data_th_xdl_pad,
                                                                        obstacle_value)
            if debug_print:
                print("Function: assign_frontier_alice :: Calculated lines between points")
            # -- debug print
            if debug_print:
                print("Function: assign_frontier_alice :: Alice Priority")
                print("\tNode: {} Initial Priority: {}".format(node.id, node.priority))
                print("\tNode Coordinate: {}".format(node.o))
                print("\tCrop Coordinates: {}\t{}".format(lt_coord, rb_coord))
                print("\tNumber of lines found: {}".format(len(lines)))
                print("\tNumber of corners found: {}".format(len(corner_coords)))
                print("\tLines Found: ")
                for l_idx, lep in enumerate(line_end_points):
                    print("\t", lep, "\tSlope: ", slopes[l_idx])
                print("-" * 50)

            # print ("##"*100)
            print("Function: assign_frontier_alice :: Calling get_node_priority")
            # -- assign priority based on line relations
            priority_val = get_node_priority(crop_map_data_pad,
                                             line_end_points,
                                             lines,
                                             slopes)
            print("\nFunction: assign_frontier_alice:: Calculated node priority: {}".format(priority_val))
            if node.priority > 0 and priority_val > 0:
                node.priority = priority_val
                print("Node Idx: ", node.id)
                print("Node Priority Set: ", node.priority)
            else:
                print("Node Idx: ", node.id)
                print("Node priority not set as init priority is ", node.priority)
                print("Calculated priority is : ", priority_val)

            # print ("FOR LOOP GOES TO NEXT INDEX")
            # print ("##"*100)

    priority_dict = {0.0: "non-frontier", 1.0: "corridor", 2.0: "large room", 3.0: "small room"}
    for node in graph_nodes:
        print("Node {}".format(node.id))
        print("\tCoordinate: {}".format(node.o))
        print("\tpriority: {}".format(node.priority))
        print("\tNode Type: {}".format(priority_dict[node.priority]))

    print("##" * 100)
    return graph_nodes
