from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances, cosine_similarity
from sklearn.cluster import AgglomerativeClustering, SpectralClustering, DBSCAN, KMeans
from sklearn.base import ClusterMixin

import argparse
from termcolor import colored

parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', dest='n_iter', default=10)
parser.add_argument('--attacker_at', dest='attacker_at', default=5)
# parser.add_argument('n_workers', dest='n_workers', default=100)
# parser.add_argument('n_mali', dest='n_mali', default=20)
# parser.add_argument('n_clusters', dest='n_clusters', default=10)
# parser.add_argument('n_range', dest='n_range', default=3)
parser.add_argument('--date_string', dest='date_string', default='')

args = parser.parse_args()

n_iter = int(args.n_iter)
n_workers = 100
n_mali = 20
n_clusters = 10
n_validation_workers = 3
n_range = 9
attacker_at = int(args.attacker_at)
date_string = args.date_string

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
    {
        'name': 'K-Means, Euclidean',
        'short_name': 'km',
        'fn': lambda x, k: KMeans(n_clusters=k).fit(x),
        'costs': [None for _ in range(n_range * 2)]
    },
]

def clustering_cost(labels, data, n_clusters):
    clusters = [[] for _ in range(n_clusters)]
    cosine_costs = []
    euclidean_costs = []
    for i, label in enumerate(labels):
        clusters[label].append(data[i])
    for cluster in clusters:
        centroid = np.mean(cluster, axis=0)
        cosine_cost = np.mean(cosine_distances(cluster, [centroid]))
        euclidean_cost = np.mean(euclidean_distances(cluster, [centroid]))
        cosine_costs.append(cosine_cost)
        euclidean_costs.append(euclidean_cost)
    return cosine_costs, euclidean_costs

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

print('>> Iteration ' + str(n_iter))
nets = np.loadtxt('out/nets/' + date_string + '/net-' + str(n_iter) + '.txt', delimiter=',')
labels = (np.loadtxt('out/nets/' + date_string + '/labels-' + str(n_iter) + '.txt', delimiter=',')).astype(int)

for method_num, clustering_method in enumerate(clustering_methods):
    print('\t' + clustering_method['name'])
    print('')
    cosine_elbow = []
    euclidean_elbow = []
    for k in range(n_clusters - n_range, n_clusters + n_range):
        clustering = clustering_method['fn'](nets, k)
        # print('\t\t\tLabels: ', str(clustering.labels_.tolist()))

        label_sizes = dict(zip(set((labels)), [0 for _ in range(len(set(labels)))]))

        clusters = [[] for _ in range(k)]
        for i, label in enumerate(clustering.labels_.tolist()):
            clusters[label].append(i)
        for cluster in clusters:
            cluster.sort()
        clusters.sort(key = lambda cluster: len(cluster), reverse = True)

        cosine_costs = []
        euclidean_costs = []
        validators_are_malicious = [0 for _ in clusters]
        centroids = []
        grads_for_clusters = []
        for cluster in clusters:
            grads = [nets[i] for i in cluster]
            grads_for_clusters.append(grads)
            cosine_cost, euclidean_cost, centroid = cluster_cost(grads)
            cosine_costs.append(cosine_cost)
            euclidean_costs.append(euclidean_cost)
            centroids.append(centroid)

        # print('\t\t\tCosine similarity costs for clusters: ' + str(cosine_costs))
        # print('\t\t\tCosine similarity average cost: ' + str(np.mean(cosine_costs)))
        # print('\t\t\tEuclidean costs for clusters: ' + str(euclidean_costs))
        # print('\t\t\tEuclidean average cost: ' + str(np.mean(euclidean_costs)))
            
        cosine_tertile = (((np.max(cosine_costs) - np.min(cosine_costs)) / 3) + np.min(cosine_costs), ((2 *(np.max(cosine_costs) - np.min(cosine_costs))) / 3) + np.min(cosine_costs))
        euclidean_tertile = (((np.max(euclidean_costs) - np.min(euclidean_costs)) / 3) + np.min(euclidean_costs), ((2 *(np.max(euclidean_costs) - np.min(euclidean_costs))) / 3) + np.min(euclidean_costs))

        print('\t\t' + (colored('k = ' + str(k), attrs=['underline']) if k == n_clusters else ('k = ' + str(k))) + '\tcos\teuclid\tclustering')
        print('\t\t\t=============================================')

        for i, cluster in enumerate(clusters):
            cluster.sort(key = lambda x: get_validation_score(nets[x], grads_for_clusters[i]))
            cluster_labels = [labels[x] for x in cluster]

            label_count = dict(zip(cluster_labels, map(cluster_labels.count, cluster_labels)))
            for label in label_count:
                if label_count[label] > label_sizes[label]:
                    label_sizes[label] = label_count[label]

            cosine_cost = cosine_costs[i]
            euclidean_cost = euclidean_costs[i]
            display_string = colored(str(round(cosine_cost, 3)), 'red' if cosine_cost > cosine_tertile[1] else 'green' if cosine_cost <= cosine_tertile[0] else 'yellow') + '\t' + colored(str(round(euclidean_cost, 3)), 'red' if euclidean_cost > euclidean_tertile[1] else 'green' if euclidean_cost <= euclidean_tertile[0] else 'yellow')

            display_string += '\t['
            for j, x in enumerate(cluster):
                display_string += ' '
                if x > (n_workers - n_mali):
                    if x <= (n_workers - n_mali + attacker_at):
                        display_string += colored(int(labels[x]), 'white', 'on_red')
                    else:
                        display_string += colored(int(labels[x]), 'red')
                    if j < n_validation_workers:
                        validators_are_malicious[i] += 1
                else:
                    display_string += colored(int(labels[x]), 'blue')
                # display_string += ' (' + str(round(get_validation_score(nets[x], grads_for_clusters[i]) * 100, 0)) + ') '
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

            print('\t\t\t' + display_string)
        print('')

        cosine_elbow.append(round(np.sum(cosine_costs), 3))
        euclidean_elbow.append(round(np.sum(euclidean_costs), 3))

        clustering_score = np.mean(list(label_sizes.values()))
        print('\t\tClustering scoring: ' + str(round(clustering_score, 2)))
        print('\t\tMalicious workers selected as validators: ' + str(round(np.mean(validators_are_malicious), 2)))
        print('')
            
    plt.figure(method_num)
    plt.title(clustering_method['name'] + ' cost elbow curves')
    plt.ylabel('Costs (cosine and Euclidean)')
    plt.xlabel('k')
    plt.plot(range(n_clusters - n_range, n_clusters + n_range), cosine_elbow, label='Cosine similarity')
    plt.plot(range(n_clusters - n_range, n_clusters + n_range), euclidean_elbow, label='Euclidean')
    plt.legend()
    # plt.savefig(fname='clustering/attacker_at-' + str(attacker_at) + '-n_iter-' + str(n_iter) + '-' + clustering_method['short_name'] + '-elbows.png')

# start validation at iteration 7, using labels there

# use average of all pairwise cosine similarity scores across past five iterations

# compare with taking cosine similarities average across iterations

# keep splitting or merging until threshold, or until clusters have similar scores

# each cluster should have similar cost, and similar training data distribution

# what happens when cluster sizes are different? try this with uneven worker groupings

# find another scoring metric within a cluster to elect validation workers, some way to avoid malicious outliers within a cluster
# try with cosine and euclidean