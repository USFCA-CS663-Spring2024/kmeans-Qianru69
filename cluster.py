import numpy as np
import random

class cluster:

    def __init__(self, k=5, max_iterations=100):
        self.k = k   # the target number of cluster centroids
        self.max_iterations = max_iterations    # maximum number of times to execute the convergence attempt

    def fit(self, X):
        # place k centroids (𝜇1,𝜇2,...,𝜇k∈ Rn) randomly
        selected_indices = random.sample(range(len(X)), self.k)
        centroids = [X[i] for i in selected_indices]
        # repeat to convergence:
        for _ in range(self.max_iterations):
            cluster_labels = []
            # foreach x ∈ test_x:
            for x in X:
                distances = []
                for c in centroids:
                    dist = self.calc_euclidian(x, c)  # Calculate the Euclidean distance
                    distances.append(dist)
                # c(i) = index of closest centroid to x
                closest_centroid_index = self.find_min_distance(distances)
                cluster_labels.append(closest_centroid_index)

            # Step 2b: Update each centroid to be the mean of the points assigned to it
            new_centroids = []
            # foreach k ∈ centroids:
            for ki in range(self.k):
                assigned_points = []
                # 𝜇k = mean c(i) | index(c(i)) == k
                for xi, point in enumerate(X):
                    if cluster_labels[xi] == ki:  # If the point is assigned to the current centroid
                        assigned_points.append(point)
                # Calculate the mean of points assigned to this centroid
                if assigned_points:
                    centroid_mean = self.find_centroid(assigned_points)
                else:
                    # Handle case where no points are assigned to this centroid (edge case)
                    centroid_mean = centroids[ki]  # use the previous centroid
                new_centroids.append(centroid_mean)
            # update centroids
            centroids = new_centroids

        return cluster_labels, centroids

    def calc_euclidian(self, x, c):
        dist = 0
        for i in range(len(x)):
            dist += (x[i] - c[i]) ** 2
        return dist ** 0.5

    def find_min_distance(self, distances):
        closest_c = 0
        min_distance = distances[0]

        for i in range(1, len(distances)):
            if distances[i] < min_distance:
                min_distance = distances[i]
                closest_c = i
        return closest_c

    def find_centroid(self, points):
        sums = [0] * (len(points[0]))
        for p in points:
            for i in range(len(p)):
                sums[i] += p[i]
        return [total / len(points) for total in sums]


X = [[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]]
cluster_predict = cluster(2)
print(cluster_predict.fit(X))







