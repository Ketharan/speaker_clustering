import numpy as np
from numpy.linalg import norm
from sklearn import preprocessing
from sklearn.metrics import pairwise as pw
from scipy.spatial import distance
import scipy.optimize as kuhn


class DominantSetClustering:
    def __init__(self, feature_vectors, speaker_ids, metric,
                 epsilon=1.0e-6, cutoff=1.0e-6, reassignment='noise',
                 dominant_search=False):
        """
        :param metric:
            'cosine'
            'euclidean'
        :param cutoff
            <0: relative cutoff at '-cutoff * max(x)'
            >0: cutoff value. usual value 1e-6
        :param epsilon
            float: minimum distance after which to stop dynamics (tipicaly 1e-6)
        :param reassignment
            'noise': reassign each point to the closest cluster
            'whole': create a new cluster with all remaining points
            'single': create a singleton cluster for each point left
        :param dominant_search
            bool: decide label of clusters through max method.
            (if False Hungarian method will be applied)
        """
        self.feature_vectors = feature_vectors
        self.cutoff = cutoff
        self.adj_matrix = None
        #self.unique_ids = np.unique(speaker_ids)
        self.speaker_ids = speaker_ids
        self.mutable_speaker_ids = np.array(range(len(self.feature_vectors)))
        #self.le = preprocessing.LabelEncoder().fit(self.speaker_ids)
        self.ds_result = np.zeros(shape=len(self.feature_vectors), dtype=int) - 1
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> =================== 8888888888888888888888888")
        print("speaker id shape  : ", len(self.feature_vectors))
        print("embeddings size : ", len(self.feature_vectors))
        self.ds_vals = np.zeros(shape=len(self.feature_vectors))
        self.cluster_counter = 0
        self.metric = metric
        self.reassignment = reassignment
        self.dominant_search = dominant_search
        self.epsilon = epsilon
        self.k = 0

    def update_cluster(self, idx, values):
        # in results vector (self.ds_result) assign cluster number to elements of idx
        # values: are partecipating values of carateristic vector of DS
        self.ds_result[self.mutable_speaker_ids[idx]] = self.cluster_counter
        self.ds_vals[self.mutable_speaker_ids[idx]] = values[idx]
        self.mutable_speaker_ids = self.mutable_speaker_ids[idx == False]
        self.cluster_counter += 1

    def get_n_clusters(self):
        return np.max(self.ds_result)

    # similarity matrix: high value = highly similar
    def get_adj_matrix(self):
        if self.metric == 'euclidean':
            dist_mat = distance.pdist(self.feature_vectors, metric=self.metric)
            dist_mat = distance.squareform(dist_mat)
        else:  # cosine distance
            dist_mat = pw.cosine_similarity(self.feature_vectors)
            dist_mat = np.arccos(dist_mat)
            dist_mat[np.eye(dist_mat.shape[0]) > 0] = 0
            dist_mat /= np.pi

        # the following heuristic is derived from Perona 2005 (Self-tuning spectral clustering)
        # with adaption from Zemene and Pelillo 2016 (Interactive image segmentation using
        # constrained dominant sets)
        sigmas = np.sort(dist_mat, axis=1)[:, 1:8]
        sigmas = np.mean(sigmas, axis=1)
        sigmas = np.dot(sigmas[:, np.newaxis], sigmas[np.newaxis, :])
        dist_mat /= -sigmas
        self.adj_matrix = np.exp(dist_mat)

        # zeros in main diagonal needed for dominant sets
        self.adj_matrix = self.adj_matrix * (1. - np.identity(self.adj_matrix.shape[0]))
        return self.adj_matrix

    def reassign(self, A, x):
        if self.reassignment == 'noise':
            for id in range(0, A.shape[0]):
                mask = id == np.arange(0, A.shape[0])
                features = self.feature_vectors[self.mutable_speaker_ids[mask]][0]
                nearest = 0.
                cluster_id = 0
                for i in range(0, np.max(self.ds_result)):
                    cluster_elements = self.feature_vectors[self.ds_result == i]
                    if len(cluster_elements) > 0:
                        cluster_vls = self.ds_vals[self.ds_result == i]
                        dominant_element = cluster_elements[cluster_vls == np.max(cluster_vls)][0]
                        temp_features = self.feature_vectors
                        self.feature_vectors = np.asmatrix([features, dominant_element])
                        # call to get_adj_matrix will give the similarity matrix for the selected 2 elements
                        dista = self.get_adj_matrix()[0, 1]
                        self.feature_vectors = temp_features

                        if dista > nearest:
                            cluster_id = i
                            nearest = dista
                self.ds_result[self.mutable_speaker_ids[mask]] = cluster_id
                self.ds_vals[self.mutable_speaker_ids[mask]] = 0.
        if self.reassignment == 'whole':
            self.update_cluster(np.asarray(x) >= 0., np.zeros(shape=len(x)))
        if self.reassignment == 'single':
            x = np.ones(x.shape[0])
            while np.count_nonzero(x) > 0:
                temp = np.zeros(shape=x.shape[0], dtype=bool)
                temp[0] = True
                self.update_cluster(temp, np.zeros(shape=x.shape[0]))
                x = x[1:]

    def apply_clustering(self):

        self.get_adj_matrix()  # calculate similarity matrix based on metric

        counter = 0
        A = self.adj_matrix
        x = np.ones(A.shape[0]) / float(A.shape[0])  # initialize x (carateristic vector)
        while x.size > 1:  # repeat until all objects have been clustered
            dist = self.epsilon * 2
            while dist > self.epsilon and A.sum() > 0:  # repeat until convergence (dist < epsilon means convergence)
                x_old = x.copy()
                counter += 1

                x = x * A.dot(x)  # apply replicator dynamics
                x = x / x.sum() if x.sum() > 0. else x
                dist = norm(x - x_old)  # calculate distance

            temp_cutoff = self.cutoff
            if self.cutoff < 0.:  # relative cutoff
                temp_cutoff = np.abs(self.cutoff) * np.max(x)

            # in case of elements not belonging to any cluster at the end,
            # we assign each of them based on self.reassignment preference
            if A.sum() == 0 or sum(x >= temp_cutoff) == 0:
                print("leaving out:" + str(x.size))
                self.reassign(A, x)
                return

            counter = 0
            idx = x < temp_cutoff
            # those elements whose value is >= temp_cutoff are the ones belonging to the cluster just found
            # on x are their partecipating values (carateristic vector)
            self.update_cluster(x >= temp_cutoff, x)

            A = A[idx, :][:, idx]  # remove elements of cluster just found, from matrix
            x = np.ones(A.shape[0]) / float(A.shape[0])  # re-initialize x (carateristic vector)
        if x.size > 0:  # in case of 1 remaining element, put him on a single cluster
            self.update_cluster(x >= 0., x)

        print("111111111111111111111----------------------->>>>>>>>>>>>>>>>>>>555555555555555")
        print("Inside apply clustering ds vals:", self.ds_vals)
        print("Inside apply clustering ds result:", self.ds_result)
        return self.ds_result

#
#     def evaluate(self):
#         self.k = np.max(self.ds_result) + 1
#
#         # Missclassification Rate
#         # assignment of clusters label
#         if self.dominant_search:
#             v = get_most_partecipating(labels=self.ds_result, ground_truth=self.speaker_ids, x=self.ds_vals)
#         else:
#             v = get_hungarian(labels=self.ds_result, ground_truth=self.speaker_ids)
#
#         print("--------------------------->>>>>>>>>>>>>>>>>>>>>")
#         print(v)
#         print(self.speaker_ids)
#
#         return v
#
# def get_hungarian(labels, ground_truth):
#     le = preprocessing.LabelEncoder().fit(ground_truth)
#     ground_truth = le.transform(ground_truth)
#     unique_truth = np.unique(ground_truth)
#     # Matrix for hungarian people
#     oc_mat = np.zeros(shape=(np.max(labels) + 1, unique_truth.shape[0]))
#
#     for i in range(0, np.max(labels) + 1):
#         cluster_elements = ground_truth[labels == i]
#         if len(cluster_elements) > 0:
#             for id in cluster_elements:
#                 oc_mat[i, id] += 1
#     assign = kuhn.linear_sum_assignment(np.max(oc_mat) - oc_mat)
#
#     v = np.zeros_like(ground_truth)
#     for i in range(0, assign[0].shape[0]):
#         v[labels == assign[0][i]] = unique_truth[assign[1][i]]
#
#     return v
#
#
# def get_most_partecipating(labels, ground_truth, x):
#     le = preprocessing.LabelEncoder().fit(ground_truth)
#     result_labels = np.zeros_like(labels)
#     for i in range(0, np.max(labels) + 1):
#         cluster_elements = ground_truth[labels == i]
#         cluster_vals = x[labels == i]
#         dominant_element = cluster_elements[cluster_vals == np.max(cluster_vals)][0]
#         mask1 = labels == i
#         mask2 = ground_truth == dominant_element
#         result_labels[mask1 & mask2] = le.transform([dominant_element])[0]
#         result_labels[mask1 & np.logical_not(mask2)] = -1
#
#     return result_labels
