from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN, KMeans
from sklearn.base import ClusterMixin

import argparse
import math
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--iter', dest='iter', default=None)
parser.add_argument('--attacker_at', dest='attacker_at', default=5)
# parser.add_argument('n_workers', dest='n_workers', default=100)
# parser.add_argument('n_mali', dest='n_mali', default=20)
# parser.add_argument('n_clusters', dest='n_clusters', default=10)
# parser.add_argument('n_range', dest='n_range', default=3)
parser.add_argument('--date_string', dest='date_string', default='')

args = parser.parse_args()

iter = None if args.iter is None else int(args.iter)
iter_start = 0
iter_end = 30
if iter is not None:
    iter_start = iter
    iter_end = iter + 1
n_workers = 100
n_mali = 20
n_clusters = 10
n_range = 8
attacker_at = int(args.attacker_at)
date_string = args.date_string
detailed_run = False if iter is None else True

clustering_methods = [
    {
        'name': 'Spectral, cosine similarity',
        'short_name': 'spec',
        'fn': lambda x, k: SpectralClustering(n_clusters=k, affinity='cosine').fit(x),
        'costs': [None for _ in range(n_range * 2)]
    },
    {
        'name': 'Agglomerative, cosine similarity',
        'short_name': 'aggl',
        'fn': lambda x, k: AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='complete').fit(x),
        'costs': [None for _ in range(n_range * 2)]
    },
    # {
    #     'name': 'K-Means, Euclidean',
    #     'short_name': 'km',
    #     'fn': lambda x, k: KMeans(n_clusters=k).fit(x),
    #     'costs': [None for _ in range(n_range * 2)]
    # },
]

def cluster_cost(cluster):
    centroid = np.mean(cluster, axis=0)
    cosine_cost = np.sum(cosine_distances(cluster, [centroid]))
    euclidean_cost = np.sum(euclidean_distances(cluster, [centroid]))
    return cosine_cost, euclidean_cost, centroid

def get_validation_score(candidate, cluster):
    centroid = np.mean(cluster, axis=0)
    return np.mean(euclidean_distances([candidate, centroid]))
    return np.mean(cosine_distances([candidate, centroid]))
    return np.mean(cosine_distances([candidate], cluster))


def experiment():
    for method_num, clustering_method in enumerate(clustering_methods):
        for n_iter in range(iter_start, iter_end):
            nets = np.loadtxt('out/nets/' + date_string + '/net-' + str(n_iter) + '.txt', delimiter=',')
            labels = (np.loadtxt('out/nets/' + date_string + '/labels-' + str(n_iter) + '.txt', delimiter=',')).astype(int)
            print('Iteration ' + str(n_iter) + ', ' + clustering_method['name'])
            print('')
            cosine_elbow = []
            euclidean_elbow = []
            cluster_fidelity_scores = []
            mal_detection_scores = []
            for k in range(n_clusters - n_range, n_clusters + n_range):
                clustering = clustering_method['fn'](nets, k)

                label_sizes = dict(zip(set((labels)), [(0, 0) for _ in range(len(set(labels)))]))

                clusters = [[] for _ in range(k)]
                for i, label in enumerate(clustering.labels_.tolist()):
                    clusters[label].append(i)
                for cluster in clusters:
                    cluster.sort()
                clusters.sort(key = lambda cluster: len(cluster), reverse = True)

                cosine_costs = []
                euclidean_costs = []
                mal_worker_avg_rankings = []
                majorities = []
                centroids = []
                grads_for_clusters = []
                for cluster in clusters:
                    grads = [nets[i] for i in cluster]
                    grads_for_clusters.append(grads)
                    cosine_cost, euclidean_cost, centroid = cluster_cost(grads)
                    cosine_costs.append(cosine_cost)
                    euclidean_costs.append(euclidean_cost)
                    centroids.append(centroid)
                    
                cosine_tertile = (((np.max(cosine_costs) - np.min(cosine_costs)) / 3) + np.min(cosine_costs), ((2 *(np.max(cosine_costs) - np.min(cosine_costs))) / 3) + np.min(cosine_costs))
                euclidean_tertile = (((np.max(euclidean_costs) - np.min(euclidean_costs)) / 3) + np.min(euclidean_costs), ((2 *(np.max(euclidean_costs) - np.min(euclidean_costs))) / 3) + np.min(euclidean_costs))

                if detailed_run: print('\t' + (colored('k = ' + str(k), attrs=['underline']) if k == n_clusters else ('k = ' + str(k))) + '\tcos\teuclid\tclustering')
                if detailed_run: print('\t\t=============================================')

                for i, cluster in enumerate(clusters):
                    cluster.sort(key = lambda x: get_validation_score(nets[x], grads_for_clusters[i]))
                    cluster_labels = [labels[x] for x in cluster]

                    majority = (-1, -math.inf)
                    label_count = dict(zip(cluster_labels, map(cluster_labels.count, cluster_labels)))
                    for label in label_count:
                        if label_count[label] > majority[1]:
                            majority = (label, label_count[label])
                    if majority[1] > label_sizes[majority[0]][0]:
                        label_sizes[majority[0]] = (majority[1], majority[1] / len(cluster))
                    majorities.append((majority[1] / len(cluster)))

                    mal_worker_ranking = []
                    cosine_cost = cosine_costs[i]
                    euclidean_cost = euclidean_costs[i]
                    display_string = colored(str(round(cosine_cost, 3)), 'red' if cosine_cost > cosine_tertile[1] else 'green' if cosine_cost <= cosine_tertile[0] else 'yellow') + '\t' + colored(str(round(euclidean_cost, 3)), 'red' if euclidean_cost > euclidean_tertile[1] else 'green' if euclidean_cost <= euclidean_tertile[0] else 'yellow')

                    display_string += '\t['
                    for j, x in enumerate(cluster):
                        if j > 80:
                            display_string += ' ...'
                            break
                        display_string += ' '
                        if x > (n_workers - n_mali):
                            if x <= (n_workers - n_mali + attacker_at):
                                display_string += colored(int(labels[x]), 'white', 'on_red')
                            else:
                                display_string += colored(int(labels[x]), 'red')
                            mal_worker_ranking.append(j / len(cluster))
                        else:
                            display_string += colored(int(labels[x]), 'blue')
                        # display_string += ' (' + str(round(get_validation_score(nets[x], grads_for_clusters[i]) * 100, 0)) + ') '
                    if len(mal_worker_ranking) > 0: mal_worker_avg_rankings.append(np.mean(mal_worker_ranking))
                    display_string += ' ]'
                    if k < n_clusters:
                        if np.max(euclidean_costs) == euclidean_cost:
                            display_string += colored(' split up (euclid)', 'magenta')
                        if np.max(cosine_costs) == cosine_cost:
                            display_string += colored(' split up (cosine)', 'cyan')
                    elif k > n_clusters:
                        if np.min(euclidean_costs) == euclidean_cost:
                            display_string += colored(' merge together (euclid)', 'magenta')
                        if np.min(cosine_costs) == cosine_cost:
                            display_string += colored(' merge together (cosine)', 'cyan')
                        if np.partition(euclidean_costs, 2)[2] == euclidean_cost:
                            display_string += colored(' merge together (euclid)', 'magenta')
                        if np.partition(cosine_costs, 2)[2] == cosine_cost:
                            display_string += colored(' merge together (cosine)', 'cyan')

                    if detailed_run: print('\t\t' + display_string)
                if detailed_run: print('')

                cosine_elbow.append(round(np.sum(cosine_costs), 3))
                euclidean_elbow.append(round(np.sum(euclidean_costs), 3))

                # print('\tClustering fidelity scoring A: ' + str(round(np.mean(majorities), 2)))
                cluster_fidelity_score = np.sum(list(filter(lambda x: x > 0, map(lambda x: x[1], label_sizes.values())))) / len(clusters)
                cluster_fidelity_scores.append(cluster_fidelity_score)
                if detailed_run: print('\tClustering fidelity score: ' + str(round(cluster_fidelity_score, 2)))
                mal_detection_score = np.mean(mal_worker_avg_rankings)
                mal_detection_scores.append(mal_detection_score)
                if detailed_run: print('\tMalicious detection score: ' + str(round(mal_detection_score, 2)))
                if detailed_run: print('')
                    
            # plt.figure(method_num)
            # plt.title(clustering_method['name'] + ' cost elbow curves')
            # plt.ylabel('Costs (cosine and Euclidean)')
            # plt.xlabel('k')
            # plt.plot(range(n_clusters - n_range, n_clusters + n_range), cosine_elbow, label='Cosine similarity')
            # plt.plot(range(n_clusters - n_range, n_clusters + n_range), euclidean_elbow, label='Euclidean')
            # plt.legend()
            # plt.savefig(fname='out/clustering/attacker_at-' + str(attacker_at) + '-n_iter-' + str(n_iter) + '-' + clustering_method['short_name'] + '-elbows.png')

            print('Average clustering fidelity score: ' + str(round(np.mean(cluster_fidelity_scores), 2)))
            print('Average malicious detection score: ' + str(round(np.mean(mal_detection_scores), 2)))
            print('')

def combo_approach():
    print('=========================================================================================\n')
    print('Remove all malicious via agglomerative, then recluster via spectral.\n')
    # print('Reference agglomerative clusters to remove skew malicious group via spectral clustering.\n')
    print('=========================================================================================')
    for k in range(n_clusters - n_range, n_clusters + n_range):

        aggl_clusters = [[] for _ in range(k)]
        for i, label in enumerate((AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='complete').fit(nets)).labels_.tolist()):
            aggl_clusters[label].append(i)
        for cluster in aggl_clusters:
            cluster.sort()
        aggl_clusters.sort(key = lambda cluster: len(cluster), reverse = True)

        grads = []
        centroids = []
        cosine_costs = []
        euclidean_costs = []
        for cluster in aggl_clusters:
            g = [nets[i] for i in cluster]
            grads.append(g)
            cosine_cost, euclidean_cost, centroid = cluster_cost(g)
            cosine_costs.append(cosine_cost)
            euclidean_costs.append(euclidean_cost)
            centroids.append(centroid)

        mal_ref_cluster_idx = np.argmax(euclidean_costs)
        mal_ref_cluster = aggl_clusters[mal_ref_cluster_idx]

        filtered_nets = list(filter(lambda x: x is not None, [(None if idx in mal_ref_cluster else x) for idx, x in enumerate(nets)]))
        filtered_labels = list(filter(lambda x: x is not None, [(None if idx in mal_ref_cluster else x) for idx, x in enumerate(labels)]))

        adjustments = [0 for _ in range(len(nets))]
        running_adjustment = 0
        for idx, x in enumerate(nets):
            if idx in mal_ref_cluster:
                running_adjustment += 1
                adjustments[idx] += running_adjustment

        spec_clusters_1 = [[] for _ in range(k)]
        f = (SpectralClustering(n_clusters=k, affinity='cosine').fit(filtered_nets)).labels_.tolist()
        for i, label in enumerate(f):
            spec_clusters_1[label].append(i)
        for cluster in spec_clusters_1:
            cluster.sort()
        spec_clusters_1.sort(key = lambda cluster: len(cluster), reverse = True)

        print((colored('k = ' + str(k), attrs=['underline']) if k == n_clusters else ('k = ' + str(k))) + '\tclusters\trand score: ' + str(adjusted_rand_score(filtered_labels, f)))
        print('\t=============================================')
        for cluster in spec_clusters_1:
            display_string = '\t['
            for j, x in enumerate(cluster):
                adjusted_x = x + adjustments[x]
                if j > 80:
                    display_string += ' ...'
                    break
                display_string += ' '
                if adjusted_x > (n_workers - n_mali):
                    if adjusted_x <= (n_workers - n_mali + attacker_at):
                        display_string += colored(int(labels[adjusted_x]), 'white', 'on_red')
                    else:
                        display_string += colored(int(labels[adjusted_x]), 'red')
                else:
                    display_string += colored(int(labels[adjusted_x]), 'blue')
            display_string += ' ]'

            print('' + display_string)

        spec_clusters_2 = [[] for _ in range(k)]
        f2 = (SpectralClustering(n_clusters=k, affinity='cosine').fit(nets)).labels_.tolist()
        for i, label in enumerate(f2):
            spec_clusters_2[label].append(i)
        for cluster in spec_clusters_2:
            cluster.sort()
        spec_clusters_2.sort(key = lambda cluster: len(cluster), reverse = True)
        
        # print((colored('k = ' + str(k), attrs=['underline']) if k == n_clusters else ('k = ' + str(k))) + '\tclusters\trand score: ' + str(adjusted_rand_score(labels, f2)))
        # print('\t=============================================')
        # for idx, cluster in enumerate(spec_clusters_2):
        #     mal_skew_cluster = None
        #     cluster_labels = [labels[x] for x in cluster]
        #     label_counts = list(sorted(zip(cluster_labels, map(cluster_labels.count, cluster_labels)), key = lambda x: x[1], reverse = True))
        #     dominant_label = label_counts[0][0]
        #     matches = sum([1 if (dominant_label == labels[x] and x in mal_ref_cluster) else 0 for x in cluster])
        #     if (matches / len(cluster)) >= 0.9:
        #         mal_skew_cluster = cluster

        #     display_string = '\t['
        #     for j, x in enumerate(cluster):
        #         if j > 80:
        #             display_string += ' ...'
        #             break
        #         display_string += ' '
        #         if x > (n_workers - n_mali):
        #             if x <= (n_workers - n_mali + attacker_at):
        #                 display_string += colored(int(labels[x]), 'white', 'on_red')
        #             else:
        #                 display_string += colored(int(labels[x]), 'red')
        #         else:
        #             display_string += colored(int(labels[x]), 'blue')
        #     display_string += ' ]'

        #     if mal_skew_cluster is not None:
        #         display_string += colored('\tidentified as skewed malicious cluster', 'magenta')

        #     print('' + display_string)
        
        # print('')

experiment()
# combo_approach()

# start validation at iteration 7, using labels there

# use average of all pairwise cosine similarity scores across past five iterations

# compare with taking cosine similarities average across iterations

# keep splitting or merging until threshold, or until clusters have similar scores

# each cluster should have similar cost, and similar training data distribution

# what happens when cluster sizes are different? try this with uneven worker groupings

# find another scoring metric within a cluster to elect validation workers, some way to avoid malicious outliers within a cluster
# try with cosine and euclidean

# vary malicious per group: 2, 4, 6

# combine spectral and agglomerative? use agg as refernce guide for malicious workers, ensure validation workers don't belong to agg maclicious cluster, find spectral cluster of malicious workers as the cluster for which all workers are in the aggl malicious cluster
# use aggl to get rid of all malicious workers, then spectral clustering on clean data

# find out why aggl and spec lead to those results

# share experiment configs

# how does iteration affect clusteirng fidelity and how far malicious workers are from their cnetroids

# compare aggl and spec on k = 10 under rand score after removing all malicious clients