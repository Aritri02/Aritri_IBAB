import numpy as np
import matplotlib.pyplot as plt

def manual_kmeans():
    # (a) Observations
    X = np.array([
        [1, 4],
        [1, 3],
        [0, 4],
        [5, 1],
        [6, 2],
        [4, 0]
    ])

    n_clusters = 2
    n_obs = X.shape[0]
    max_iter = 100  # Prevent infinite loops

    # (b) Random initial cluster labels
    labels = np.random.choice(n_clusters, size=n_obs)

    # 📊 Initial Cluster Visualization
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100)
    for i in range(n_obs):
        plt.text(X[i, 0]+0.1, X[i, 1], f"{i+1}", fontsize=12)
    plt.title("Initial Random Cluster Assignments")
    plt.grid(True)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    def compute_centroids(X, labels):
        centroids = []
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) == 0:
                # Reinitialize centroid randomly if no points assigned
                centroid = X[np.random.choice(n_obs)]
            else:
                centroid = cluster_points.mean(axis=0)
            centroids.append(centroid)
        return np.array(centroids)

    def assign_clusters(X, centroids):
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1)

    iteration = 0
    while iteration < max_iter:
        iteration += 1
        centroids = compute_centroids(X, labels)
        new_labels = assign_clusters(X, centroids)

        print(f"Iteration {iteration}: Labels = {new_labels}")

        if np.array_equal(new_labels, labels):
            print("Converged.")
            break
        labels = new_labels
    else:
        print("Max iterations reached. Might not have fully converged.")

    # (f) Final plot
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=100)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    for i in range(n_obs):
        plt.text(X[i, 0]+0.1, X[i, 1], f"{i+1}", fontsize=12)
    plt.title("Final K-Means Clustering Result")
    plt.legend()
    plt.grid(True)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def main():
    manual_kmeans()

if __name__ == "__main__":
    main()
