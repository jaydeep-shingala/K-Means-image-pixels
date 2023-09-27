import numpy as np

class KMeans:
    def __init__(self, k: int, epsilon: float = 1e-6) -> None:
        self.num_clusters = k
        self.cluster_centers = None
        self.epsilon = epsilon
        self.final_assignment = None
    
    def fit(self, X: np.ndarray, max_iter: int = 100) -> None:
        # Initialize cluster centers (need to be careful with the initialization,
        # otherwise you might see that none of the pixels are assigned to some
        # of the clusters, which will result in a division by zero error)
        self.cluster_centers = X[np.random.choice(X.shape[0], size=self.num_clusters, replace=False)]
        old_Assignment = np.random.randint(self.num_clusters, size=X.shape[0])

        for p in range(0,max_iter):
            new_Assignment = self.predict(X, old_Assignment)
            for j in range(self.num_clusters):
                self.cluster_centers[j] = X[new_Assignment == j].mean(axis=0)
            diff = np.linalg.norm(new_Assignment - old_Assignment)
            if diff < self.epsilon:
                break
            old_Assignment = new_Assignment
        self.final_assignment = new_Assignment

    
    def predict(self, X: np.ndarray, new_Assignment) -> np.ndarray:
        # Predicts the index of the closest cluster center for each data point
        # raise NotImplementedError
        # new_Assignment = np.random.randint(self.num_clusters, size=X.shape[0])
        for i in range(0,X.shape[0]):
                mindist = float('inf')
                mindistk = 0
                for j in range(0,self.num_clusters):
                    dist = np.linalg.norm(X[i] - self.cluster_centers[j])
                    if(dist < mindist):
                        mindist = dist
                        mindistk = j
                new_Assignment[i] = mindistk
        return new_Assignment
        # pass
    
    def fit_predict(self, X: np.ndarray, max_iter: int = 100) -> np.ndarray:
        self.fit(X, max_iter)
        return self.predict(X)
    
    def replace_with_cluster_centers(self, X: np.ndarray) -> np.ndarray:
        # Returns an ndarray of the same shape as X
        # Each row of the output is the cluster center closest to the corresponding row in X
        for i in range(0, X.shape[0]):
            X[i] = self.cluster_centers[self.final_assignment[i]]
        return X
        # raise NotImplementedError