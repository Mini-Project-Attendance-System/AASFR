from math import sqrt
from pickle import dump

from sklearn import neighbors

from .HOGTrainerTemplate import HOGTrainerTemplate

class KNNTrainer(HOGTrainerTemplate):

    def __init__(self, classifier = None, pixels_per_cell = 16, cells_per_block = 4):

        super().__init__(classifier, pixels_per_cell, cells_per_block)

    def trainKNN(self, n_neighbors = None, save_path = None):

        if n_neighbors is None:

            n_neighbors = int(round(sqrt(len(self.images))))

        self.classifier = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, algorithm="ball_tree", weights="distance")

        self.classifier.fit(self.hog_features, self.classes)

        self.classifier.max_dim1 = self.max_dim1
        self.classifier.max_dim2 = self.max_dim2

        if save_path is not None:

            with open(save_path, "wb+") as knn_model:

                dump(self.classifier, knn_model)
