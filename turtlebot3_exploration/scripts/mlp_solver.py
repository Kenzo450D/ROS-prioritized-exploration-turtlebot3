#!/usr/bin/python3

import numpy as np
import copy as cp

import mlppy  # IMPORTANT: mlppy.so (or a symbolic link to it) must be in the directory where this script is located

from all_pairs_shortest_path import all_pairs_shortest_path

class MLPSolver:
    def __init__(self, occ_graph_dict: dict,
                 time_remaining: float,
                 prize_map:dict,
                 cur_vertex:int,
                 home_vertex:int,
                 dist_all_nodes_cur:dict,
                 resolution,
                 avg_speed):
        # -- get the vertices of the graph
        self.cur_vertex = cur_vertex
        self.explored_graph = occ_graph_dict
        self.home_vertex = home_vertex
        self.node_priority = prize_map
        self.time_remaining = time_remaining
        self.dist_all_nodes_current_vertex = dist_all_nodes_cur
        self.resolution = resolution
        self.avg_speed = avg_speed
        self.time_offset = 40
        self.return_home_time_multiplier = 0.9
        # print ("MLP Solver Initialized")
        # print ("Current Vertex: ", self.cur_vertex)
        # print ("Explored Graph: ", self.explored_graph)
        # print ("="*100 + "END of explored graph")
        # print ("Home Vertex: ", self.home_vertex)
        # print ("Node Priority: ", self.node_priority)
        # print ("Time Remaining: ", self.time_remaining)
        # print ("Dist all nodes: ", self.dist_all_nodes_current_vertex)
        # print ("Resolution: ", self.resolution)
        # print ("Avg Speed: ", self.avg_speed)
        return

    def get_action(self):
        """ Set up the MLP solver and get action
        1. Creates a new vertex index map
        2. Set up prizes for each vertex
        3. Prepares and runs the MLP solver
        """
        # -- get discovered states
        self.discovered_states = self._get_discovered_vertices()

        self.vertex_map_explored, self.reverse_vertex_map_explored = \
            self.__calculate_vertex_map_all_vertices(self.explored_graph, self.home_vertex)

        # -- calculate distance matrix
        self.distance_matrix = self.__calculate_distance_matrix(self.explored_graph, self.vertex_map_explored, self.resolution, self.avg_speed)

        # -- choose action based on time remaining        
        if self.time_remaining == np.inf:
            print ("Time Remaining: ", self.time_remaining)
            print ("Deadline is unknown!")
            # input("Continue?")
            # -- deadline is unknown
            return  self._choose_mlp_action()
        else:
            print ("Time Remaining: ", self.time_remaining)
            # input("Continue?")
            # -- deadline is known
            return self._choose_mlp_action_time()

    def _convert_node_prize_mlp_prize(self, node_prize):
        if node_prize == 1.0: #corridor
            return self.node_priority["corridor"]
        elif node_prize == 2.0: #large_room
            return self.node_priority["large_room"]
        elif node_prize == 3.0: #small_room
            return self.node_priority["small_room"]
        else:
            return 0.0

    def _choose_mlp_action(self, debug_print: bool = False):
        states_nodes = [self.cur_vertex] + [node_idx for node_idx, _ in self.discovered_states.items()]
        states_costs_from_curr = [0.0] + [node_cost for _, node_cost in self.discovered_states.items()]
        return self._select_action_based_on_mlp(states_nodes, states_costs_from_curr)

    def _choose_mlp_action_time(self, debug_print: bool = False):
        # Get the home node and costs of returning home from other nodes.
        costs_to_home = self.distance_matrix[self.vertex_map_explored[self.home_vertex]]
        # Check if returning home is possible.
        time_est_to_home =  costs_to_home[self.vertex_map_explored[self.cur_vertex]] + self.time_offset 
        print ("Time estimate to home: ", time_est_to_home)
        print ("Time remaining: ", self.time_remaining)
        # input("_choose_ml_action_time: Press Enter to continue...")
        return_home_possible = costs_to_home[self.vertex_map_explored[self.cur_vertex]] * self.return_home_time_multiplier <= self.time_remaining
        # Initialize nodes with the current node.
        states_nodes = [self.cur_vertex]
        states_costs_from_curr = [0.0]
        # Consider only nodes that can be reached from the current node before the deadline and also (if possible) allow returning home before the deadline.
        for node_idx, node_cost in self.discovered_states.items():
            time_estimate_to_node_idx = node_cost + costs_to_home[self.vertex_map_explored[node_idx]] + self.time_offset
            print ("\tnode_idx: ", node_idx)
            print ("\ttime_estimate_to_node_idx: ", time_estimate_to_node_idx)
            print ("\ttime_remaining: ", self.time_remaining)
            print ("\treturn_home_possible: ", return_home_possible)
            print ("\t" + "-"*100)
            if (return_home_possible and time_estimate_to_node_idx <= self.time_remaining) or (not return_home_possible and node_cost <= self.time_remaining):
                states_nodes.append(node_idx)
                states_costs_from_curr.append(node_cost)
        if len(states_nodes) == 1:  # No other nodes were added.
            if return_home_possible:
                if self.cur_vertex == self.home_vertex:
                    print ("Robot already at home vertex!")
                    return 0.0, None
                else:
                    return costs_to_home[self.vertex_map_explored[self.cur_vertex]] + self.time_offset, self.home_vertex  # Return home.
            else:
                print ("Robot should stay in place!")
                return 0.0, None # Stay at place.
        print ("_choose_mlp_action_time :: States nodes: ", states_nodes)
        print ("_choose_mlp_action_time :: States costs from curr: ", states_costs_from_curr)
        # input("_choose_mlp_action_time :: Press Enter to continue...")
        return self._select_action_based_on_mlp(states_nodes, states_costs_from_curr)

    def _get_discovered_vertices(self)->dict:
        ''' Discovered vertices consists of the vertices that have not been visited. 
        Global variables used: 
            self.explored_graph
            self.dist_all_nodes_current_vertex
        Output:
            discovered_states: a dictionary of the discovered vertices and their costs from the current vertex.
        '''
        # -- initialize the list of discovered vertices
        discovered_vertices = {}

        # -- loop through the vertices to create the discovered_vertex list
        for vertex in self.dist_all_nodes_current_vertex.keys():
            vertex_priority = self.explored_graph[vertex][0].priority
            if (vertex not in discovered_vertices) and (vertex_priority > 0) and (self.dist_all_nodes_current_vertex[vertex] != np.inf):
                discovered_vertices[vertex] = self.dist_all_nodes_current_vertex[vertex]
        return discovered_vertices

    def _select_action_based_on_mlp(self, discovered_vertices, discovered_vertex_costs_from_curr):
        discovered_vertex_map = {node_idx: i for i, node_idx in enumerate(discovered_vertices)}
        # -- change the state priority
        states_priorities = [int(round(self._convert_node_prize_mlp_prize(self.explored_graph[v_idx][0].priority))) for v_idx in discovered_vertices]
        states_cost_matrix = self._construct_all_pairs_shortest_paths_cost_matrix(discovered_vertices, discovered_vertex_map)
        mlp_solver = mlppy.Solver()
        # mlp_solver.set_time_limit(1e-3)
        mlp_solver.solve(states_cost_matrix, states_priorities)
        action_idx = mlp_solver.solution()[1]
        if discovered_vertices[action_idx] == self.cur_vertex: # sending back current vertex makes the robot takes spot turns for a significant cost
            return 0.0, None
        return discovered_vertex_costs_from_curr[action_idx], discovered_vertices[action_idx]


    def _construct_all_pairs_shortest_paths_cost_matrix(self, states_nodes, states_mapping):
        # Init the matrix.
        n_states = len(states_nodes)
        states_cost_matrix_float = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
        max_cost = -1.0

        # Compute the cost matrix.
        # cost matrix is a symmetric matrix with the current vertex and eligible discovered vertices
        # that can be visited within the time remaining.
        # We will use the explored graph 
        for idx_row, row_data in enumerate(self.distance_matrix):
            if self.reverse_vertex_map_explored[idx_row] in states_nodes:
                state_matrix_row = states_mapping[self.reverse_vertex_map_explored[idx_row]]
                for idx_col, col_data in enumerate(row_data):
                    if self.reverse_vertex_map_explored[idx_col] in states_nodes:
                        state_matrix_col = states_mapping[self.reverse_vertex_map_explored[idx_col]]
                        states_cost_matrix_float[state_matrix_row][state_matrix_col] = col_data
                        if col_data > max_cost:
                            max_cost = col_data

        # Convert the costs to integers (MLP solver works only with integers because of speed and numeric stability).
        states_cost_matrix_int = [[0 for _ in range(n_states)] for _ in range(n_states)]
        multiplier = 1000.0
        for i, vec in enumerate(states_cost_matrix_float):
            for j, value in enumerate(vec):
                states_cost_matrix_int[i][j] = int(round(multiplier * value / max_cost))
        # Return the result.
        return states_cost_matrix_int

    def __calculate_vertex_map_all_vertices(self, occ_graph_dict, home_vertex_idx):
        """ Creates a vertex map of the vertices that are present in the graph.
        The mapped vertices are integers that start with a vertex of 0.
        The home node has a vertex of 0.
        """
        # --- initialize the vertex map
        vertex_map = {}
        reverse_vertex_map = {} 
        vertex_map[home_vertex_idx] = 0
        reverse_vertex_map[0] = home_vertex_idx

        # -- loop through the vertices to createa a vertex map

        last_index = 1
        for node in occ_graph_dict.keys():
            if node not in vertex_map:
                vertex_map[node] = last_index
                reverse_vertex_map[last_index] = node
                last_index += 1

        # print ("Function: __calculate_vertex_map_all_vertices::  ")
        # print ("\tVertex Map: ", vertex_map)
        # print ("\tReverse Vertex Map: ", reverse_vertex_map)
        return vertex_map, reverse_vertex_map

    def __calculate_distance_matrix(self, explored_graph, vertex_map_explored, resolution, avg_speed):
        '''
        Make a distance matrix. A set of integers showing weights for each edge weight
        Input:
            graph: A dictionary, key value is (ki) index id, value is list of tuples,
                  where each tuple is a (vertex_id, distance_to_ki).
            vertex_map: A map of the vertices and the indices.
        '''
        n_vertices = len(explored_graph)
        # -- initialize the distance matrix
        dist_matrix = np.ones((n_vertices, n_vertices)) * np.inf
        # -- loop through the vertices to create the distance matrix
        for k_i, val_i in explored_graph.items():
            v_idx = vertex_map_explored[k_i]
            dist_matrix[v_idx, v_idx] = 0
            # -- loop through the values
            for edge in val_i[1]:
                target_v_weight = edge.weight * resolution / avg_speed
                if edge.source == k_i:
                    target_v_idx = edge.target
                else:
                    target_v_idx = edge.source
                t_v_idx = vertex_map_explored[target_v_idx]
                dist_matrix[v_idx, t_v_idx] = target_v_weight
                dist_matrix[t_v_idx, v_idx] = target_v_weight

        # -- call all pair shortest path to calculate the adj matrix
        dist_matrix = all_pairs_shortest_path(dist_matrix)

        # -- print the new distance matrix
        # print("Function: Make distance Matrix")
        # print("distance matrix after all pairs shortest path:")
        # self.__print_distance_matrix(dist_matrix)

        # -- return the distance matrix and vertex map
        return dist_matrix#.tolist()

    def __print_distance_matrix(self, dist_matrix):
        print("Distance Matrix:")
        for row in dist_matrix:
            print(row)
        return
