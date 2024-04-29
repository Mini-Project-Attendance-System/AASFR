from math import sqrt
from pickle import dump
from time import perf_counter

from sklearn import neighbors

from .HOGTrainerTemplate import HOGTrainerTemplate

class KNNTrainer(HOGTrainerTemplate):

    def __init__(self, classifier = None, pixels_per_cell = 16, cells_per_block = 4):

        super().__init__(classifier, pixels_per_cell, cells_per_block, logger_name="KNNTrainer")

    def trainKNN(self, n_neighbors = None, save_path = None):

        self.logger.info(f"n_neighbors set to {n_neighbors}")

        if n_neighbors is None:

            n_neighbors = int(round(sqrt(len(self.images))))

            self.logger.info(f"n_neighbors was None. Setting new value : {n_neighbors}")

        self.classifier = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, algorithm="ball_tree", weights="distance")

        self.logger.info("Training KNN model with HoG features and Classification labels")

        start_time = perf_counter()

        self.classifier.fit(self.hog_features, self.classes)
        self.logger.info(f"KNN model trained in {perf_counter() - start_time}s")

        self.classifier.resize_dim1 = self.resize_dim1
        self.classifier.resize_dim2 = self.resize_dim2

        if save_path is not None:

            with open(save_path, "wb+") as knn_model:

                self.logger.info(f"Saving trained KNN model as : {save_path}")
                dump(self.classifier, knn_model)
