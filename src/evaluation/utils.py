import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from src.evaluation.projection import FrozenRandomProjection

def get_prototypes(X_train, y_train):
    class_mean_vectors = {}
    for diagnosis_class in np.unique(y_train):
        class_features = X_train[y_train == diagnosis_class]  # Feature columns
        class_mean_vectors[diagnosis_class] = np.mean(class_features, axis=0)  # Compute mean vector
   
    return class_mean_vectors

def predict_nearest_mean(test_sample, class_prototypes):
    distances = {label: euclidean(test_sample, mean_vec) for label, mean_vec in class_prototypes.items()}
    return min(distances, key=distances.get)  


def get_projected_datasets(X_train, X_test, in_dim, out_dim):
    proj_layer = FrozenRandomProjection(input_dim=in_dim, output_dim=out_dim)
    X_train_proj = proj_layer.transform(X_train)
    X_test_proj = proj_layer.transform(X_test)
    return X_train_proj, X_test_proj

def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm

def apply_LDA(X_train ,y_train , X_test):
    lda = LinearDiscriminantAnalysis(n_components=None)  # max = num_classes - 1
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    return X_train_lda, X_test_lda

def apply_PCA(X_train, X_test):
  pca = PCA(n_components=0.95)
  X_train_pca = pca.fit_transform(X_train)
  X_test_pca = pca.transform(X_test)
  return X_train_pca, X_test_pca

def get_task_data(X, y, task_classes):
    mask = np.isin(y, task_classes)
    return X[mask], y[mask]
