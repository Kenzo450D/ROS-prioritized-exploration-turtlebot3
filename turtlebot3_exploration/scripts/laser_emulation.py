#!/usr/bin/env python3

import numpy as np

"""
Simulates lidar from any position and orientation in a map
Input:
    Set of Particles(x,y, theta)
    Map
    Explored Map Pixel value
    Unexplored Map Pixel value
    Occupied Map Pixel value
    Resolution of laser sensor
    Range of laser sensor
Output:
    For each node in set of nodes, it would return a set of #resolution
    numbers referring to distances of the laser sensor. 
"""

class LaserFeature:
    def __init__(self,
                 occ_map,
                 resolution,  # angles in scan 
                 lidar_max_range,
                 free_map_val = 0, 
                 occupied_map_val = 100, 
                 unexplored_map_val = -1):
        # -- initialize
        self.resolution = resolution
        self.lidar_max_range = lidar_max_range
        self.free_val = free_map_val
        self.occ_val = occupied_map_val
        self.occ_map = occ_map
        self.occupied_map_val = occupied_map_val
        self.unknown_val = unexplored_map_val

        self.circle_points = self._get_points_in_a_circle(lidar_max_range, resolution)
        self.image_scanlines, self.enabled_pixels = self._calculate_image_scanlines()
    
    def get_laser_data(self, robot_position):
        ''' Get laser data from a robot position.
        
        Input:
            robot_position: (x,y) coordinate
        ''' # robot position in terms of map coordinate x and y coordinate
        # print "Robot position: x: ", robot_position[0]
        # print "Robot position: y: ", robot_position[1]
        lidar_scanlines = np.copy(self.image_scanlines)
        lidar_scanlines[:,:,0] += int(np.round(robot_position[0]))
        lidar_scanlines[:,:,1] += int(np.round(robot_position[1]))
        return lidar_scanlines

        # TODO: Check if the rest if required
        # -- initialize laser pixels
        laser_pixels = lidar_scanlines[:,-1,0:2]
        # -- loop through all the image scanlines
        for i in range(0, lidar_scanlines.shape[0]):
            for j in range(0,self.enabled_pixels[i]):
                if np.array_equal(self.occ_map[tuple(lidar_scanlines[i,j,:])], self.occupied_map_val):
                    if j == 0:
                        laser_pixels[i,:] = robot_position
                    else:
                        laser_pixels[i,:] = lidar_scanlines[i,j-1,0:2]
                    break
                if j == (self.enabled_pixels[i] - 1):
                    laser_pixels[i,:] = lidar_scanlines[i,j,0:2]
        # -- get the euclidean distance
        distances = np.zeros(self.enabled_pixels.shape)
        for i in range(0,lidar_scanlines.shape[0]):
            d = self._get_2D_distance(robot_position, laser_pixels[i,:])
            if d >= (self.lidar_max_range - 1): #Robot checks up to (lidar_max_range - 1) in a direction
                d = np.inf
            distances[i] = d
        self.distances = distances
        return distances

    def get_distances(self):
        return self.distances

    def _get_2D_distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _get_points_in_a_circle(self, radius, discretization):
        """ 
        The robot is assumed to be pointed towards the positive x axis. The 0 degree points towards the
        direction of the robot.
        """
        cp = [0,0] # center point
        op = np.tile(cp, (discretization, 1)) # output points
        for i in range(discretization):
            d_x = radius * np.cos((i *  2 * np.pi) / discretization)#along positive axis
            d_y = radius * np.sin((i *  2 * np.pi) / discretization)#along positive axis
            op[i,0] = np.round(d_y) # rows change y axis
            op[i,1] = np.round(d_x) # columns change x axis
        # -- check if unique values, if not unique, then reduce
        unique_op = np.unique(op,axis=0)
        if unique_op.shape[0] == op.shape[0]:
            return op
        else:
            self.resolution = unique_op.shape[0]
            return unique_op


    def _createLineIterator(self, p1, p2, include_starting_point=False):
        """
        Produces and array that consists of the coordinates and intensities of each pixel in a line between two points
        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
        Returns:
            -itbuffer: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
        """
        # -- define local variables for readability
        p1_row = p1[0]
        p1_col = p1[1]
        p2_row = p2[0]
        p2_col = p2[1]
        
        #print("Line Iterator: P1: ", P1, "\tP2: ", P2)

        # difference and absolute difference between points
        #used to calculate slope and relative location between points
        d_col = p2_col - p1_col     # difference column
        d_row = p2_row - p1_row     # difference row
        d_col_abs = np.abs(d_col)
        d_row_abs = np.abs(d_row)

        # -- predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(d_row_abs,d_col_abs),2),dtype=np.float32)
        #print ("Create Line Iterator: size of itbuffer: ", np.shape(itbuffer))
        itbuffer.fill(np.nan)

        # -- Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = p1_row > p2_row
        negX = p1_col > p2_col
        if p1_col == p2_col: #vertical line segment
            itbuffer[:,1] = p1_col
            
            if negY:
                itbuffer[:,0] = np.arange(p1_row - 1,p1_row - d_row_abs - 1,-1)
            
            else:
                itbuffer[:,0] = np.arange(p1_row+1,p1_row+d_row_abs+1) 
        
        elif p1_row == p2_row: #horizontal line segment
            itbuffer[:,0] = p1_row
            
            if negX:
                itbuffer[:,1] = np.arange(p1_col-1,p1_col-d_col_abs-1,-1)
            
            else:
                itbuffer[:,1] = np.arange(p1_col+1,p1_col+d_col_abs+1)
        
        else: #diagonal line segment[resolution, radius, 3]
            steepSlope = d_row_abs > d_col_abs
            
            if steepSlope:
                slope = d_col.astype(np.float32)/d_row.astype(np.float32)
                
                if negY:
                    itbuffer[:,0] = np.arange(p1_row-1,p1_row-d_row_abs-1,-1)
                
                else:
                    itbuffer[:,0] = np.arange(p1_row+1,p1_row+d_row_abs+1)
                itbuffer[:,1] = (slope*(itbuffer[:,0]-p1_row)) + float(p1_col)
            
            else:
                slope = d_row.astype(np.float32)/d_col.astype(np.float32)
                
                if negX:
                    itbuffer[:,1] = np.arange(p1_col-1,p1_col-d_col_abs-1,-1)
                
                else:
                    itbuffer[:,1] = np.arange(p1_col+1,p1_col+d_col_abs+1)
                itbuffer[:,0] = (slope*(itbuffer[:,1]-p1_col)) + float(p1_row)

        # round off the points
        itbuffer = np.round(itbuffer,0)
        itbuffer = itbuffer.astype(np.int)
        
        # -- add the starting point
        if include_starting_point:
            starting_point = np.array([p1[0], p1[1], 0,0,0])
            itbuffer = np.vstack((itbuffer, starting_point))
        
        return itbuffer

    def _calculate_image_scanlines(self):
        circle_points = self.circle_points
        image_scanlines = np.zeros((circle_points.shape[0], int(np.round(self.lidar_max_range)), 2))
        enabled_pixels = np.zeros(circle_points.shape[0], dtype=np.int)
        cp = np.array([0,0])
        for i in range(0,circle_points.shape[0]):
            pp = circle_points[i,:] # peripherical point
            # we need a line between the center point and periphery point
            line_data = self._createLineIterator(cp, pp)
            # print "Line data: "
            # print line_data.shape
            enabled_pixels[i] = line_data.shape[0]
            # save the line
            image_scanlines[i,:line_data.shape[0],:line_data.shape[1]] = line_data
        image_scanlines = image_scanlines.astype(int)
        enabled_pixels = enabled_pixels.astype(np.uint)
        return image_scanlines, enabled_pixels
