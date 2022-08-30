#!/usr/bin/env python3

import numpy as np
from laser_emulation import LaserFeature

def assign_frontier_priority_lidar_emulate(graph_nodes,
                                           map_data_array, 
                                           offset_size=15,
                                           laser_sensor_size = 100):
    """ Priority Assignment for frontier nodes by emulating lidar sensor.
    Assigns a priority of 1 for frontier nodes and keeps the priority unchanged
    for other nodes. For each node position, the laser scan is simulated,
    and if there are pixels that have a value of -1 (unexplored) within the
    laser scan pixels, the node is considered to a frontier node.
    """
    """
    Steps:
    1.  Make a template of the pixels that needs to be scanned, considering
        the center of the laser source is 0,0. The value would be negative
        and positive. This template would be pasted on the node positions
        of the graph.
        SKIP STEP 2
    2.  Scroll through all the cells in the map, for the cells that are marked
        as explored and unoccupied (i.e. value of 0), scan the neighboring 
        elements. If the neighboring elements are unknown, mark them as 
        'frontier_value_marker'. 
    3.  Go through all graph node positions on the occupancy grid. For each
        position, apply the template calculated above, get the pixels scanned
        from the occupancy grid. If any pixel is a frontier pixel, mark it as
        a frontier node.
    """
    # STEP 1:
    # create the template for laser
    lidar_emulation = LaserFeature(map_data_array, 360, 100)
    # STEP 2:
    print ("Number of graph nodes: ", len(graph_nodes))
    # STEP 3:
    # To merge step 2 and 3, while scanning through the pixels, if a -1
    # value is found which is preceded by a 0 value, the node is considered
    # to be a frontier node.
    for node in graph_nodes:
        # -- flag to set the priority
        flag_priority_set = False
        # -- get the map coordinate of the node
        node_coord_i = node.pts[0] # index 1(row) of node point
        node_coord_j = node.pts[1] # index 2(col) of node point
        # -- debug print
        print ("Node Coord: {}, {}".format(node_coord_i, node_coord_j))
        # -- get lidar_scanline
        lidar_scanlines = lidar_emulation.get_laser_data((node_coord_i, node_coord_j))
        print ("Shape of lidar_scanlines at node: {} :: {}".format(node.id, lidar_scanlines.shape))
        # print (lidar_scanlines.shape)
        # -- loop through the lidar_scanlines, get if there is a unexplored 
        #    pixel after a free and explored pixel
        for i in range(0, lidar_scanlines.shape[0]): # resolution
            print (map_data_array[tuple(lidar_scanlines[i,0])], end=" ")
            for j in range(1, lidar_scanlines.shape[1]): # distance
                cur_idx = tuple(lidar_scanlines[i,j])
                prev_idx = tuple(lidar_scanlines[i,j-1])
                cur_val = map_data_array[cur_idx]
                prev_val = map_data_array[prev_idx]
                print (cur_val, end= " ")
                if cur_val == -1 and prev_val == 0:
                    # we found a pixel that can be explored if the robot is in node position
                    print ("Node priority set to 1")
                    node.priority  = 1
                    flag_priority_set = True
                    break
                if cur_val == -1 and prev_val == -1:
                    break
                if cur_val == 100:
                    break
            print ("\n"+ "-"*50)
            if flag_priority_set:
                break
        print ("="*100)
    return graph_nodes

def assign_frontier_priority_lidar_emulate_threshold(graph_nodes,
                                                     map_data_array, 
                                                     offset_size=15,
                                                     laser_sensor_size = 100,
                                                     lidar_threshold = 0.05):
    """ Priority Assignment for frontier nodes by emulating lidar sensor.
    Assigns a priority of 1 for frontier nodes and keeps the priority unchanged
    for other nodes. For each node position, the laser scan is simulated,
    and if there are pixels that have a value of -1 (unexplored) within the
    laser scan pixels, the node is considered to a frontier node.
    """
    """
    Steps:
    1.  Make a template of the pixels that needs to be scanned, considering
        the center of the laser source is 0,0. The value would be negative
        and positive. This template would be pasted on the node positions
        of the graph.
        SKIP STEP 2
    2.  Scroll through all the cells in the map, for the cells that are marked
        as explored and unoccupied (i.e. value of 0), scan the neighboring 
        elements. If the neighboring elements are unknown, mark them as 
        'frontier_value_marker'. 
    3.  Go through all graph node positions on the occupancy grid. For each
        position, apply the template calculated above, get the pixels scanned
        from the occupancy grid. If any pixel is a frontier pixel, mark it as
        a frontier node.
        Change : Instead of marking a node as frontier node on the first occurance
        of a frontier cell, if 5% of the lidar emulation is frontier cells, then
        we mark the cell as frontier cell.
    """
    # STEP 1:
    # create the template for laser
    resolution = 360
    lidar_emulation = LaserFeature(map_data_array, resolution, laser_sensor_size)
    lidar_threshold_count = lidar_threshold * resolution
    # STEP 2:
    print ("Number of graph nodes: ", len(graph_nodes))
    # STEP 3:
    # To merge step 2 and 3, while scanning through the pixels, if a -1
    # value is found which is preceded by a 0 value, the node is considered
    # to be a frontier node.
    for node in graph_nodes:
        # -- flag to set the priority
        flag_priority_set = False
        # -- get the map coordinate of the node
        node_coord_i = node.pts[0] # index 1(row) of node point
        node_coord_j = node.pts[1] # index 2(col) of node point
        # -- debug print
        #print ("Node Coord: {}, {}".format(node_coord_i, node_coord_j))
        # -- get lidar_scanline
        lidar_scanlines = lidar_emulation.get_laser_data((node_coord_i, node_coord_j))
        #print ("Shape of lidar_scanlines at node: {} :: {}".format(node.id, lidar_scanlines.shape))
        # print (lidar_scanlines.shape)
        # -- loop through the lidar_scanlines, get if there is a unexplored 
        #    pixel after a free and explored pixel
        n_frontier_cells_discovered = 0
        for i in range(0, lidar_scanlines.shape[0]): # resolution
            #print (map_data_array[tuple(lidar_scanlines[i,0])], end=" ")
            for j in range(1, lidar_scanlines.shape[1]): # distance
                cur_idx = tuple(lidar_scanlines[i,j])
                prev_idx = tuple(lidar_scanlines[i,j-1])
                cur_val = map_data_array[cur_idx]
                prev_val = map_data_array[prev_idx]
                #print (cur_val, end= " ")
                if cur_val == -1 and prev_val == 0:
                    n_frontier_cells_discovered += 1
                    break
                if cur_val == -1 and prev_val == -1:
                    break
                if cur_val == 100:
                    break
            #print ("\n"+ "-"*50)
            if flag_priority_set:
                break
        #print ("="*100)
        if n_frontier_cells_discovered > lidar_threshold_count:
            node.priority = 1
    return graph_nodes

def assign_frontier_priority(graph_nodes, map_data_array, offset_size=15):
    """ Priority Assignment for Frontier Nodes.
    Assigns a priority of 1 for frontier nodes and keeps the priority unchanged
    for other nodes. 
    This function would be replace by a priority assignment algorithm when that 
    is ready.
    """
    """
    Steps:
    1.  Scroll through all the cells in the map, for the cells that are marked
        as explored and unoccupied (i.e. value of 0), scan the neighboring 
        elements. If the neighboring elements are unknown, mark them as 
        'frontier_value_marker'. 
    2.  Go all the graph node positions on the occupancy grid. For each of the
        positions, consider a neighborhood, for each cells in the neighborhood,
        check if there are cells with a frontier marker present. If frontier
        marker is present go to step 3, else step 4.
    3.  If a frontier marker is present in the neighborhood, make a second 
        smaller neighborhood to check if a occupancy location. This step is
        specifically made to ensure that the frontier node is not too close to
        the obstacle. If no obstacle is found, mark this node as a frontier
        node.
    4.  Mark the node as not a frontier node.
    Some modifications are necessary for the eroded map. To start off, the
    input to the function would be the eroded map as well the the unchanged
    map. As the nodes were calculated based on the eroded map, the node 
    positions are already expected to be slightly far off from the obstacles
    in the occupancy grid. So we would not require step 3. 
    TODO: Check the results from the rviz plot and then proceed if check
    if step 3 is required or not.
    """
    frontier_value_marker = 1000
    occupied_cell_value = 100
    offset = offset_size # The offset is 15 as 15 *0.05 = 75cm and it should cover more
    # than the 13 pixel gap for 
    
    # STEP 1:
    # -- identify cells that are free and neighbor to unexplored cells
    map_data_marked_frontiers = np.copy(map_data_array)
    frontier_cells = []
    for i in range(map_data_array.shape[0]):
        for j in range(map_data_array.shape[1]):
            # unoccupied and explored
            if map_data_array[i,j] == 0:
                neighbors = map_data_array[i-1:i+2, j-1:j+2].flatten()
                if -1 in neighbors:
                    # neighbor is unexplored
                    # frontier cell
                    frontier_cells.append((i,j))
                    map_data_marked_frontiers[i,j] = frontier_value_marker
    
    print ("Number of graph nodes: ", len(graph_nodes))
    # STEP 2: 
    # -- loop through nodes to identify frontier nodes
    for node in graph_nodes:
        node_coord_i = node.pts[0] # index 1(row) of node point
        node_coord_j = node.pts[1] # index 2(col) of node point
        
        # -- check the neighborhood of this node coordinate
        node_neighborhood = map_data_marked_frontiers[node_coord_i - offset:
                                                          node_coord_i + offset + 1,
                                                      node_coord_j - offset:
                                                          node_coord_j + offset + 1]
        if frontier_value_marker in node_neighborhood:
            # -- modify the vertex priority to reflect the frontier coordinate 
            # if 25 neighbor is occupied then skip
            neighbors_25 = map_data_array[node_coord_i-2:node_coord_i+3, 
                                          node_coord_j-2:node_coord_j+3]
            # -- if an occupied cell is not in the neighborhood, change the
            # node priority to 1
            if occupied_cell_value not in neighbors_25:
                node.priority = 1
                
            # -- another corner is that if there are corners next to a wall
            # but is close to a frontier. So if we find that there are more 
            # than 50% of the total number of cells
            n_unexplored_cells = np.count_nonzero(node_neighborhood == -1)
            n_frontier_cells = np.count_nonzero(node_neighborhood == frontier_value_marker)
            if (n_unexplored_cells > 0.5*(offset*offset)) and n_frontier_cells > 0.5 * offset:
                node.priority = 1
            
    # -- return the graph nodes
    return graph_nodes
