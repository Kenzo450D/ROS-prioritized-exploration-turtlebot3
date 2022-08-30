import numpy as np
import copy
from collections import defaultdict
import heapq

def quaternion_to_euler_angle_vectorized2(w: float, x: float, y: float, z: float) -> tuple:
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)

    t2 = np.clip(t2, a_min=-1.0, a_max=1.0)
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z

def _get_cost_to_all_vertices(source_node,
                              occ_graph_dict,
                              debug_print=True):
    """ https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm#Algorithm
    """

    if debug_print:
        print("#" * 100)
        print("Function: _get_cost_to_all_vertices")
        print("Source node: ", source_node)
        print("-" * 100)

    # Step 1: initialize variables
    last_node = None
    cur_node = copy.deepcopy(source_node)
    cost_to_cur_node = 0
    lowest_distance_priority_queue = []
    visited_nodes = []
    visited_nodes.append(cur_node.id)
    unvisited_nodes = []  # to store unvisited nodes
    for v_idx, v in occ_graph_dict.items():  # populate unvisited nodes
        if v_idx != cur_node.id:
            unvisited_nodes.append(v_idx)

    # Step 2: initialize dictionaries
    min_dist_to_v = defaultdict(lambda: np.inf)
    source_of_v = defaultdict(lambda: source_node.id)
    source_of_v[source_node.id] = source_of_v[source_node.id]
    min_dist_to_v[source_node.id] = 0.0

    print("_get_cost_to_all_vertices:: Before the while loop")
    print("Source Node: ", source_node)

    # Step 3: loop through nodes to get distance to all nodes
    while True:
        if debug_print:
            print("=" * 100)
            print("\t Back to the start of the loop")
        edges_incident = occ_graph_dict[cur_node.id][1]
        # edges have no particular order in source and target idx. Any 
        # of them could be the particular current node.

        for edge in edges_incident:
            if edge.source == cur_node.id:
                nodeIdx = edge.target
            elif edge.target == cur_node.id:
                nodeIdx = edge.source
            node_cost = edge.weight

            # -- Get the distance to the node
            dist_to_node = node_cost + cost_to_cur_node

            if debug_print:
                print("Node Index: ", nodeIdx)
                print("Node cost: ", node_cost)
                print("Distance to node: ", dist_to_node)
                print("-" * 50)

            # -- compare if distance is lower than minimum recorded dist
            if dist_to_node < min_dist_to_v[nodeIdx]:
                min_dist_to_v[nodeIdx] = dist_to_node
                source_of_v[nodeIdx] = cur_node.id
                # -- add to queue
                heapq.heappush(lowest_distance_priority_queue, (dist_to_node, nodeIdx))

            if debug_print:
                print("-" * 50)

        # -- EXIT condition: get the lowest cost node
        if len(lowest_distance_priority_queue) == 0:
            break
        last_node = cur_node

        # -- get cur_node_idx as node idx
        upcoming_node = heapq.heappop(lowest_distance_priority_queue)
        cur_node_idx = upcoming_node[1]
        cost_to_cur_node = upcoming_node[0]

        # -- get cur_node as node
        cur_node = occ_graph_dict[cur_node_idx][0]

        # -- add node to visited nodes
        if cur_node_idx not in visited_nodes:
            visited_nodes.append(cur_node_idx)

    # -- convert default dictionary to dictionary
    # min_dist_to_v = dict(min_dist_to_v)
    source_of_v = dict(source_of_v)

    # BEGIN debug print
    if debug_print:
        # print ("EXPLORATION AGENT: _get_cost_to_all_vertices:: All remote nodes visited!")
        print("#" * 100)
        print("_get_cost_to_all_vertices:: Minimum distances are: ")
        for key, val in dict(min_dist_to_v).items():
            print(key, val)
        print("#" * 100)
        print("_get_cost_to_all_vertices:: Sources of vertices are: ")
        for key, val in dict(source_of_v).items():
            print("\tNode: ", key, "Source: ", val)
        print("#" * 100)
        print("_get_cost_to_all_vertices:: Return back to control")
        print("=" * 100)
    # END debug print

    # stop code here
    # sys.exit(0)
    print("_get_cost_to_all_vertices:: End of function")
    # -- return
    return min_dist_to_v, source_of_v

# -------------------------------------------------------------------------
# Function: print_occ_graph
# -------------------------------------------------------------------------
def print_occ_graph(occ_graph):
    """ Prints the occupancy graph on terminal
    """
    print("-" * 100)
    print("Occupancy Graph:")
    print("Vertices:")
    for node in occ_graph.g.nodes:
        # -- made the id of the node as the key of the dictionary
        print("\t{}".format(str(node.id)))

    print("Edges: ")
    # -- traverse through all the edges
    for edge in occ_graph.g.links:
        print("{} -- {} : {}".format(edge.source, edge.target, edge.weight))

    print("=" * 100)
    print("=" * 100)
    print("=" * 100)
    return


# -------------------------------------------------------------------------
# Function: print_occ_graph_dict
# -------------------------------------------------------------------------
def print_occ_graph_dict(occ_graph_dict):
    """ Prints the occupancy graph on terminal
    """
    print("-" * 100)
    print("Function: Print Occuapcny Graph Dictionary:")
    print("Vertices:")
    for key, item in occ_graph_dict.items():
        print("=" * 80)
        print("Node ID: ", key)
        print("-" * 50)
        print("Node info: ")
        print(item[0])
        print("-" * 50)
        print("Edges incident: ")
        edges_to = []
        for edge in item[1]:
            e_s = edge.source
            e_t = edge.target
            # edge_to = np.inf
            if e_s == key:
                edges_to.append((e_t, np.round(edge.weight, 2)))
            else:
                edges_to.append((e_s, np.round(edge.weight, 2)))
        for e in edges_to:
            print(e)
        print("-" * 50)
    print("=" * 100)
    print("=" * 100)
    print("=" * 100)
    return