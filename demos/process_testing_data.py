import os

from source import ImageSplitter

if __name__ == "__main__":

    splitter = ImageSplitter()
    root = "./preprocessed_testing_images"
    save = "./testing_images"

    for directory in os.listdir(root):

        print(f"Processing data in {directory}")

        for image in os.listdir(os.path.join(root, directory)):

            splitter.findFaces(os.path.join(root, directory, image), os.path.join(save, directory))
