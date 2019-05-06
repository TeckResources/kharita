"""
author: sofiane
Create the road network by merging trajectories.

"""
from typing import Tuple

import geopy
import math
import logging
import networkx as nx
from scipy.spatial import cKDTree
import pickle
import pandas as pd

from kharita.methods import create_trajectories, diffangles, partition_edge, \
    vector_direction_re_north, Cluster


def save_graph(G, fname):
    with open('{}.pkl'.format(fname), 'wb') as output_file:
        nx.write_gpickle(G, output_file)


def save_cluster_tree(tree, fname):
    with open(fname, "wb") as output_file:
        pickle.dump(tree, output_file)


def _add_node(network: nx.Graph, node: Cluster):
    network.add_node(node.cid, location=(node.lat, node.lon))


def reconstruct_road(input_data: pd.DataFrame,
                     radius_meter: int = 100,
                     sampling_distance: int = 50,
                     heading_angle_tolerance: int = 360,
                     waiting_threshold: int = 5) -> Tuple[nx.Graph, cKDTree]:
    """
    Reconstructs the road network from the dataframe

    Arguments:
    input_data: pandas DataFrame with the following scheme:
        trajectory_id, timestamp, latitude, longitude, speed, bearing_angle
        0,2011-04-09 20:04:56+03,41.869327,-87.666068,20,266.904808057
    radius_meter: the radius (cr) in meters used for the clustering
    sampling_distance: the densification distance (sr) in meters
    heading_angle_tolerance: the angle heading tolerance in degrees
    waiting_threshold: the maximum threshold time between the gps records within the trajectory

    Returns:
        road network, kd-tree
    """


    RADIUS_DEGREE = radius_meter * 10e-6
    clusters = []
    cluster_kdtree = None
    roadnet = nx.Graph()
    p_X = []
    p_Y = []
    trajectories = create_trajectories(input_data,
                                       waiting_threshold=waiting_threshold)

    for i, trajectory in enumerate(trajectories[:-1]):
        logging.info('\rprocessing trajectory: %s / %s' % (i, len(trajectories)))
        update_cluster_index = False
        prev_cluster = -1
        for point in trajectory:
            p_X.append(point.lon)
            p_Y.append(point.lat)
            # very first case: enter only once
            if len(clusters) == 0:
                # create a new cluster
                new_cluster = Cluster(cid=len(clusters), nb_points=1, last_seen=point.timestamp,
                                      lat=point.lat,
                                      lon=point.lon, angle=point.angle)
                clusters.append(new_cluster)
                _add_node(roadnet, new_cluster)
                prev_cluster = new_cluster.cid  # all I need is the index of the new cluster
                # recompute the cluster index
                cluster_kdtree = cKDTree([c.get_lonlat() for c in clusters])
                continue
            # if there's a cluster within x meters and y angle: add to. Else: create new cluster
            close_clusters_indices = [clu_index for clu_index in
                                      cluster_kdtree.query_ball_point(x=point.get_lonlat(),
                                                                      r=RADIUS_DEGREE, p=2)
                                      if math.fabs(
                    diffangles(point.angle, clusters[clu_index].angle)) <= heading_angle_tolerance]

            if len(close_clusters_indices) == 0:
                # create a new cluster
                new_cluster = Cluster(cid=len(clusters), nb_points=1, last_seen=point.timestamp,
                                      lat=point.lat, lon=point.lon, angle=point.angle)
                clusters.append(new_cluster)
                _add_node(roadnet, new_cluster)
                current_cluster = new_cluster.cid
                # recompute the cluster index
                update_cluster_index = True
            else:
                # add the point to the cluster
                pt = geopy.Point(point.get_coordinates())
                close_clusters_distances = [geopy.distance.geodesic(pt, geopy.Point(
                    clusters[clu_index].get_coordinates())).meters
                                            for clu_index in close_clusters_indices]
                closest_cluster_indx = close_clusters_indices[
                    close_clusters_distances.index(min(close_clusters_distances))]
                clusters[closest_cluster_indx].add(point)
                current_cluster = closest_cluster_indx
            # Adding the edge:
            if prev_cluster == -1:
                prev_cluster = current_cluster
                continue

            edge = [clusters[prev_cluster], clusters[current_cluster]]
            intermediate_clusters = partition_edge(edge, distance_interval=sampling_distance)
            # intermediate_clusters = []

            # Check if the newly created points belong to any existing cluster:
            intermediate_cluster_ids = []
            for pt in intermediate_clusters:
                close_clusters_indices = [clu_index for clu_index in
                                          cluster_kdtree.query_ball_point(x=pt.get_lonlat(),
                                                                          r=RADIUS_DEGREE, p=2)
                                          if math.fabs(
                        diffangles(pt.angle, clusters[clu_index].angle)) <= heading_angle_tolerance]

                if len(close_clusters_indices) == 0:
                    intermediate_cluster_ids.append(-1)
                    continue
                else:
                    # identify the cluster to which the intermediate cluster belongs
                    PT = geopy.Point(pt.get_coordinates())
                    close_clusters_distances = [
                        geopy.distance.distance(PT, geopy.Point(
                            clusters[clu_index].get_coordinates())).meters for clu_index
                        in close_clusters_indices]
                    closest_cluster_indx = close_clusters_indices[
                        close_clusters_distances.index(min(close_clusters_distances))]
                    intermediate_cluster_ids.append(closest_cluster_indx)

            # For each element is segment: if ==-1 create new cluster and link to it, else link to the corresponding cluster
            prev_path_point = prev_cluster
            for idx, inter_clus_id in enumerate(intermediate_cluster_ids):
                if inter_clus_id == -1:
                    n_cluster_point = intermediate_clusters[idx]
                    # create a new cluster
                    new_cluster = Cluster(cid=len(clusters), nb_points=1, last_seen=point.timestamp,
                                          lat=n_cluster_point.lat,
                                          lon=n_cluster_point.lon, angle=n_cluster_point.angle)
                    clusters.append(new_cluster)
                    _add_node(roadnet, new_cluster)
                    # recompute the cluster index
                    update_cluster_index = True
                    # create the actual edge:
                    if math.fabs(diffangles(clusters[prev_path_point].angle,
                                            new_cluster.angle)) > heading_angle_tolerance \
                        or math.fabs(diffangles(
                        vector_direction_re_north(clusters[prev_path_point], new_cluster),
                        clusters[prev_path_point].angle)) > heading_angle_tolerance:
                        prev_path_point = new_cluster.cid
                        continue
                    # if satisfy_path_condition_distance(prev_path_point, new_cluster.cid, roadnet, clusters, alpha=1.2):
                    roadnet.add_edge(prev_path_point, new_cluster.cid)
                    prev_path_point = new_cluster.cid
                else:
                    roadnet.add_edge(prev_path_point, inter_clus_id)
                    prev_path_point = inter_clus_id
                    clusters[inter_clus_id].add(intermediate_clusters[idx])
            if len(intermediate_cluster_ids) == 0 or intermediate_cluster_ids[-1] != current_cluster:
                roadnet.add_edge(prev_path_point, current_cluster)
            prev_cluster = current_cluster
        if update_cluster_index:
            cluster_kdtree = cKDTree([c.get_lonlat() for c in clusters])

    return roadnet, cluster_kdtree


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(reconstruct_road(pd.read_csv('../data/data_for_kharita.csv')))
