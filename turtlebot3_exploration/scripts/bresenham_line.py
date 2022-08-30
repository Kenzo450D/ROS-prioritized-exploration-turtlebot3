#!/usr/bin/env python3

import numpy as np

def _createLineIterator(p1, p2):
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

    return itbuffer

def check_obstacles(map_coord1, map_coord2, occ_map, occupied_map_value = 100):
    mc1 = np.array(map_coord1, dtype='int')
    mc2 = np.array(map_coord2, dtype='int')
    #print "check_obstacles: ", mc1, "\t", mc2
    line_pixels = _createLineIterator(mc1, mc2)
    for i in range(line_pixels.shape[0]):
        #print line_pixels[i,:]
        if occ_map[tuple(line_pixels[i,:])] == occupied_map_value:
            return False
    return True

if __name__=='__main__':
    a = [[100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0],
         [100,0,0,0,0,100,0,0,0,0]]
    check_obstacles([5,5],[10,10],None)
