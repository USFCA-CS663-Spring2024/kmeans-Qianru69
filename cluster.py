import random
random.seed(12345678)

class cluster:

    def __init__(self, k=5, max_iterations=100, balanced=False):
        self.k = k   # the target number of cluster centroids
        self.max_iterations = max_iterations    # maximum number of times to execute the convergence attempt
        self.balanced = balanced

    def fit(self, X):
        # place k centroids (𝜇1,𝜇2,...,𝜇k∈ Rn) randomly
        selected_indices = random.sample(range(len(X)), self.k)
        centroids = [X[i] for i in selected_indices]
        cluster_labels = []
        # repeat to convergence:
        for i in range(self.max_iterations):
            new_cluster_labels = []
            # foreach x ∈ test_x:
            for x in X:
                distances = []
                for c in centroids:
                    dist = self.calc_euclidian(x, c)  # Calculate the Euclidean distance
                    distances.append(dist)
                # c(i) = index of closest centroid to x
                closest_centroid_index = self.find_min_distance(distances)
                new_cluster_labels.append(closest_centroid_index)

            # if the cluster labels doesn't change, the break from the loop earlier
            if cluster_labels == new_cluster_labels:
                break
            else:
                cluster_labels = new_cluster_labels

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

        if self.balanced:
            cluster_labels, centroids = self.balance(cluster_labels, centroids, X)

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

    def balance(self, cluster_labels, centroids, X):
        max_i_for_balancing = 10
        for i in range(max_i_for_balancing):
            # Calculate cluster sizes
            cluster_sizes = [0] * self.k
            for label in cluster_labels:
                cluster_sizes[label] += 1

            # Identify the largest and smallest clusters
            largest_cluster_index = cluster_sizes.index(max(cluster_sizes))
            smallest_cluster_index = cluster_sizes.index(min(cluster_sizes))

            # Check if clusters are already balanced within a threshold
            difference = max(cluster_sizes) - min(cluster_sizes)

            if difference <= 1 or difference < 0.01 * (len(X) / self.k):
                break  # Clusters are balanced

            # Re-balance clusters
            for j, label in enumerate(cluster_labels):
                if label == largest_cluster_index:
                    # Calculate distances to other centroids
                    distances = [float('inf')] * self.k
                    for k in range(self.k) :
                        if k == largest_cluster_index:
                            continue
                        distances[k] = self.calc_euclidian(X[k], centroids[k])  # Calculate the Euclidean distance
                    new_cluster_index = self.find_min_distance(distances)

                    # Check if moving this point to the new cluster improves balance
                    if new_cluster_index == smallest_cluster_index:
                        # Update the point's cluster label
                        cluster_labels[j] = new_cluster_index
                        # Update cluster sizes
                        cluster_sizes[largest_cluster_index] -= 1
                        cluster_sizes[smallest_cluster_index] += 1

                        # Update centroids
                        # For simplicity, we'll recalculate from scratch for affected clusters
                        for ki in [largest_cluster_index, smallest_cluster_index]:
                            assigned_points = []
                            # 𝜇k = mean c(i) | index(c(i)) == k
                            for xi, point in enumerate(X):
                                if cluster_labels[xi] == ki:  # If the point is assigned to the current centroid
                                    assigned_points.append(point)

                            if assigned_points:  # Avoid division by zero
                                centroids[ki] = self.find_centroid(assigned_points)
                            else:
                                # Edge case: cluster has no points, keep the old centroid
                                pass

                        # Recalculate the largest and smallest after the move
                        largest_cluster_index = cluster_sizes.index(max(cluster_sizes))
                        smallest_cluster_index = cluster_sizes.index(min(cluster_sizes))
        return cluster_labels, centroids

#X = [[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]]
#cluster_predict = cluster(2, 1, True)
#print(cluster_predict.fit(X))
