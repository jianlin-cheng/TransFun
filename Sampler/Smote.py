import torch
from random import randint
import random


class SMOTE(object):
    def __init__(self, distance='euclidian', dims=512, k=5):
        super(SMOTE, self).__init__()
        self.newindex = 0
        self.k = k
        self.dims = dims
        self.distance_measure = distance

    def populate(self, N, i, nnarray, min_samples, k):
        while N:
            nn = randint(0, k - 2)

            diff = min_samples[nnarray[nn]] - min_samples[i]
            gap = random.uniform(0, 1)

            self.synthetic_arr[self.newindex, :] = min_samples[i] + gap * diff

            self.newindex += 1

            N -= 1

    def k_neighbors(self, euclid_distance, k):
        nearest_idx = torch.zeros((euclid_distance.shape[0], euclid_distance.shape[0]), dtype=torch.int64)

        idxs = torch.argsort(euclid_distance, dim=1)
        nearest_idx[:, :] = idxs

        return nearest_idx[:, 1:k]

    def find_k(self, X, k):
        euclid_distance = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32)

        for i in range(len(X)):
            dif = (X - X[i]) ** 2
            dist = torch.sqrt(dif.sum(axis=1))
            euclid_distance[i] = dist

        return self.k_neighbors(euclid_distance, k)

    def generate(self, min_samples, N, k):
        """
            Returns (N/100) * n_minority_samples synthetic minority samples.
    		Parameters
    		----------
    		min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
    		    Holds the minority samples
    		N : percetange of new synthetic samples:
    		    n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    		k : int. Number of nearest neighbours.
    		Returns
    		-------
    		S : Synthetic samples. array,
    		    shape = [(N/100) * n_minority_samples, n_features].
    	"""
        T = min_samples.shape[0]
        self.synthetic_arr = torch.zeros(int(N / 100) * T, self.dims)
        N = int(N / 100)
        if self.distance_measure == 'euclidian':
            indices = self.find_k(min_samples, k)
        for i in range(indices.shape[0]):
            self.populate(N, i, indices[i], min_samples, k)
        self.newindex = 0
        return self.synthetic_arr

    def fit_generate(self, X, y):
        # get occurence of each class
        occ = torch.eye(int(y.max() + 1), int(y.max() + 1))[y].sum(axis=0)
        # get the dominant class
        dominant_class = torch.argmax(occ)
        # get occurence of the dominant class
        n_occ = int(occ[dominant_class].item())
        for i in range(len(occ)):
            if i != dominant_class:
                # calculate the amount of synthetic data_bp to generate
                N = (n_occ - occ[i]) * 100 / occ[i]
                candidates = X[y == i]
                xs = self.generate(candidates, N, self.k)
                X = torch.cat((X, xs))
                ys = torch.ones(xs.shape[0]) * i
                y = torch.cat((y, ys))
        return X, y