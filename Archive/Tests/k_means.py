import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(p1, p2):
    return sum((a - b) ** 2 for a, b in zip(p1, p2)) ** (1 / 2)


def compute_distances(data, centroids):
    return [[euclidean_distance(point, centroid) for centroid in centroids] for point in data]


def argmin(arr):
    return min(range(len(arr)), key=lambda i: arr[i])


def k_means(data, k, n, max_iter=10 ** 4):
    np.random.seed(0)
    centroids = data[np.random.choice(n, k)]

    for _ in range(max_iter):
        distances = compute_distances(data, centroids)
        labels = [argmin(row) for row in distances]
        new_centroids = np.array([data[np.array(labels) == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


n = len(data)
clusters, centroids = k_means(data, 3, n)

for i in range(3):
    print(f"cluster {i + 1}:")
    print("\n".join(f"- text {j + 1}" for j, cluster in enumerate(clusters) if cluster == i))

plt.figure(figsize=(15, 8))
plt.scatter(data[:, 0], data[:, 1], c=clusters, label='texts')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label='centroids')
plt.xlabel('distance to centroid 1')
plt.ylabel('distance to centroid 2')
plt.title('K_means clustering')
plt.legend()
plt.show()
