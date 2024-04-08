from pickle import dump

from sklearn.svm import SVC

from .HOGTrainerTemplate import HOGTrainerTemplate

class SVMTrainer(HOGTrainerTemplate):

    def __init__(self, classifier = None, pixels_per_cell = 16, cells_per_block = 4):

        super().__init__(classifier, pixels_per_cell, cells_per_block)

    def trainSVM(self, save_path = None):

        self.classifier = SVC(kernel="linear")

        self.classifier.fit(self.hog_features, self.classes)

        self.classifier.max_dim1 = self.max_dim1
        self.classifier.max_dim2 = self.max_dim2

        if save_path is not None:

            with open(save_path, "wb+") as svm_model:

                dump(self.classifier, svm_model)
