from os import listdir

from source import SVMTrainer

if __name__ == "__main__":

    predictor = SVMTrainer()

    for image in listdir("testing_images"):

        try:

            prediction = predictor.predictFace(f"testing_images/{image}", "trained_models/vakibcSVM.clf")
            print(prediction)

        except Exception as e:

            print(f"{e}\nOccured for image : {image}")

            continue
