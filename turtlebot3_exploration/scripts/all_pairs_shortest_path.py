import numpy as np
from copy import deepcopy

def printSolution(dist):
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            print(dist[i,j], end = "\t")
        print()
    return


def all_pairs_shortest_path(init_adj_matrix):
    """ Calculates the all pairs shortest path among all vertices.

    Implements the Floyd Warshall algorithm. 
    """
    if type(init_adj_matrix) == list:
        init_adj_matrix = np.array(init_adj_matrix)
    costMatrix = np.ones_like(init_adj_matrix)
    costMatrix *= np.inf
    costMatrix = deepcopy(init_adj_matrix)
    n_nodes = len(init_adj_matrix)
    for k in range(0, n_nodes):
        for i in range(0, n_nodes):
            for j in range(0, n_nodes):
                if costMatrix[i,j] > (costMatrix[i,k] + costMatrix[k,j]) and costMatrix[i,k] != np.inf and costMatrix[k,j] != np.inf:
                    costMatrix[i,j] = costMatrix[i,k] + costMatrix[k,j]
    
    # printSolution(costMatrix)

    return costMatrix


if __name__ == '__main__':
    print ("Main code: ")
    adj_matrix = np.array([[0, 3, 6, np.inf, np.inf, np.inf, np.inf],
                           [3, 0, 2, 1, np.inf, np.inf, np.inf],
                           [6, 2, 0, 1, 4, 2, np.inf],
                           [np.inf, 1, 1, 0, 2, np.inf, 4],
                           [np.inf, np.inf, 4, 2, 0, 2, 1],
                           [np.inf, np.inf, 2, np.inf, 2, 0, 1],
                           [np.inf, np.inf, np.inf, 4, 1, 1, 0]])
    new_adj_matrix = all_pairs_shortest_path(adj_matrix)
