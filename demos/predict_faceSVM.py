from os import listdir, path

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from source import SVMTrainer, Logger

if __name__ == "__main__":

    predictor = SVMTrainer()

    logger = Logger("SVMPrediction")
    logger.toggle_logger(True)
    logger = logger.logger

    root = "testing_images"

    predictions = []
    labels = []

    for directory in listdir(root):

        if (directory == "_group_pics"):

            continue

        for image in listdir(path.join(root, directory)):

            labels.append(directory)

            prediction = predictor.predictFace(path.join(root, directory, image), "trained_models/vakibcSVM.clf")

            predictions.append(prediction)

    logger.info(f"Accuracy score : {accuracy_score(labels, predictions)}")
    logger.info(f"Classification report :\n{classification_report(labels, predictions)}")

    matrix = confusion_matrix(labels, predictions)

    logger.info(f"Confusion matrix :\n{matrix}")
