import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import accuracy_score

from ISLP import load_data

def load_nci_data():
    NCI60 = load_data('NCI60')
    X = NCI60['data']
    y = NCI60['labels'].values.ravel()
    return X, y

def reduce_features_pca(X, n_components=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca

def reduce_features_hclust(X, n_clusters=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    linkage_matrix = linkage(X_scaled.T, method='ward')
    cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    clustered_features = []
    for cluster_num in range(1, n_clusters + 1):
        cluster_columns = np.where(cluster_assignments == cluster_num)[0]
        cluster_mean = X_scaled[:, cluster_columns].mean(axis=1)
        clustered_features.append(cluster_mean)

    X_hclust = np.vstack(clustered_features).T
    return X_hclust

from sklearn.model_selection import train_test_split

def evaluate_model(X, y):
    model = SVC(kernel='rbf', C=1.0, gamma='scale')
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    return accuracy


def main():
    # Load data
    X, y = load_nci_data()

    # PCA-based feature reduction
    X_pca = reduce_features_pca(X, n_components=5)
    pca_accuracy = evaluate_model(X_pca, y)
    print(f'PCA-based classification accuracy (SVM): {pca_accuracy:.4f}')

    # Hierarchical clustering-based feature reduction
    X_hclust = reduce_features_hclust(X, n_clusters=5)
    hclust_accuracy = evaluate_model(X_hclust, y)
    print(f'Hierarchical clustering-based classification accuracy (SVM): {hclust_accuracy:.4f}')

    # Visualize results
    results_df = pd.DataFrame({
        'Method': ['PCA (5 PCs)', 'Hierarchical Clustering (5 clusters)'],
        'Accuracy': [pca_accuracy, hclust_accuracy]
    })

    sns.barplot(data=results_df, x='Method', y='Accuracy')
    plt.ylim(0, 1)
    plt.title('SVM Classification Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.show()

if __name__ == "__main__":
    main()
