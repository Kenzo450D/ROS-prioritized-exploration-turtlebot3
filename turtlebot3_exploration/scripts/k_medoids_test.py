import numpy as np

from k_medoids import KMedoidClustering

# from sklearn_extra.cluster import KMedoids
# from pyclustering.utils.metric import distance_metric
# from pyclustering.utils.metric import type_metric
# from collections import defaultdict


# def get_user_metric(distance_matrix):
#     usr_metric = lambda p1, p2: distance_matrix[int(p1),int(p2)]
#     dmat_metric = distance_metric(type_metric.USER_DEFINED, func=usr_metric)
#     return dmat_metric

# def cluster_vertices(distance_matrix, n_clusters):
#     """ Cluster vertex nodes based on the distance matrix between them.
#     Input:
#         distance_matrix
#         n_clusters
#     """
#     # -- get the metric
#     usr_def_metric = get_user_metric(distance_matrix)
#     n_vertices = len(distance_matrix)
#     # make the variables of a graph in a linear space X
#     # this is possible as we have a custom distance metric defined
#     X = np.linspace(0,n_vertices-1,n_vertices).astype('int')
#     X = X.reshape(-1,1) # make it a column array
#     # print ("Distance_matrix: \n")
#     # print (distance_matrix)
#     # print (distance_matrix.shape)
#     # print (X)
#     cobj = KMedoids(n_clusters, metric=usr_def_metric, method='pam', init='build').fit(X)
#     labels = cobj.labels_
#     cluster_centers = cobj.cluster_centers_
#     return labels, cluster_centers

if __name__=='__main__':
    dist_mat = np.array([[0.0,68.5,56.4,32.1,58.2,40.1,112.6,106.1,108.1,160.1,118.1,141.3,143.7,154.1],
                        [68.5,0.0,60.7,36.4,46.4,28.3,100.7,94.3,96.3,148.3,106.3,129.5,131.9,142.3],
                        [56.4,60.7,0.0,24.3,50.5,32.4,104.8,98.4,100.4,152.4,110.4,133.5,135.9,146.4],
                        [32.1,36.4,24.3,0.0,26.1,8.1,80.5,74.1,76.1,128.0,86.1,109.2,111.6,122.0],
                        [58.2,46.4,50.5,26.1,0.0,18.1,90.5,84.1,86.1,138.0,96.1,119.2,121.6,132.0],
                        [40.1,28.3,32.4,8.1,18.1,0.0,72.4,66.0,68.0,120.0,78.0,101.1,103.6,114.0],
                        [112.6,100.7,104.8,80.5,90.5,72.4,0.0,74.4,76.4,128.4,86.4,109.6,112.0,122.4],
                        [106.1,94.3,98.4,74.1,84.1,66.0,74.4,0.0,2.0,54.0,12.0,35.1,37.6,48.0],
                        [108.1,96.3,100.4,76.1,86.1,68.0,76.4,2.0,0.0,56.0,10.0,33.1,35.6,46.0],
                        [160.1,148.3,152.4,128.0,138.0,120.0,128.4,54.0,56.0,0.0,66.0,89.1,91.5,101.9],
                        [118.1,106.3,110.4,86.1,96.1,78.0,86.4,12.0,10.0,66.0,0.0,43.1,25.6,36.0],
                        [141.3,129.5,133.5,109.2,119.2,101.1,109.6,35.1,33.1,89.1,43.1,0.0,68.7,79.1],
                        [143.7,131.9,135.9,111.6,121.6,103.6,112.0,37.6,35.6,91.5,25.6,68.7,0.0,61.5],
                        [154.1,142.3,146.4,122.0,132.0,114.0,122.4,48.0,46.0,101.9,36.0,79.1,61.5,0.0]])

    nclusters = 10
    home_vertex = 0
    cur_vertex = 10
    prize_list = [0,10000,10,100,100,10,10,10000,10000,10,10,10,10,100]
    # labels, cc = cluster_vertices(dist_mat, nclusters)
    # print (cc)
    kmc = KMedoidClustering(dist_mat, prize_list, nclusters, home_vertex, cur_vertex)
