from pickle import dump
from time import perf_counter

from sklearn.svm import SVC

from .HOGTrainerTemplate import HOGTrainerTemplate

class SVMTrainer(HOGTrainerTemplate):

    def __init__(self, classifier = None, pixels_per_cell = 16, cells_per_block = 4):

        super().__init__(classifier, pixels_per_cell, cells_per_block, logger_name="SVMTrainer")

    def trainSVM(self, save_path = None):

        self.classifier = SVC(kernel="linear")

        self.logger.info("Training SVM model with HoG features and Classification labels")

        start_time = perf_counter()

        self.classifier.fit(self.hog_features, self.classes)
        self.logger.info(f"SVM model trained in {perf_counter() - start_time}s")

        self.classifier.max_dim1 = self.max_dim1
        self.classifier.max_dim2 = self.max_dim2

        if save_path is not None:

            with open(save_path, "wb+") as svm_model:

                self.logger.info(f"Saving trained SVM model as : {save_path}")
                dump(self.classifier, svm_model)
