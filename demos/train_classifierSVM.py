from source import SVMTrainer

if __name__ == "__main__":

    trainer = SVMTrainer()

    trainer.loadImagesHOG("training_images/")
    trainer.trainSVM(save_path="trained_models/vakibcSVM.clf")
