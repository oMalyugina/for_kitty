import os
import codecs
from PIL import Image, ImageOps
import math
import numpy as np
from sklearn.model_selection import train_test_split


class Label:
    size_for_cutting = 64

    def __init__(self, label_str: str):
        label_str = label_str.split(" ")
        self.object_type = label_str[0]
        self.truncated = float(label_str[1])
        self.occluded = int(label_str[2])
        self.x_left = float(label_str[4])
        self.y_top = float(label_str[5])
        self.x_right = float(label_str[6])
        self.y_bottom = float(label_str[7])

    def make_borders_square_with_ideal_size_or_bigger(self):
        x_c = (self.x_right + self.x_left) / 2
        y_c = (self.y_bottom + self.y_top) / 2
        width = self.x_right - self.x_left
        height = self.y_bottom - self.y_top
        width = max(width, height, self.size_for_cutting)
        height = max(width, height, self.size_for_cutting)
        self.x_left = math.floor(x_c - width / 2)
        self.x_right = math.ceil(x_c + width / 2)
        self.y_top = math.floor(y_c - height / 2)
        self.y_bottom = math.ceil(y_c + height / 2)
        # print("new size = x: {} {} y: {} {}".format(self.x_left, self.x_right, self.y_top, self.y_bottom))


class DataGenerator:

    def __init__(self, folder_with_images: str, folder_with_labels: str):
        self.object2number = {word: i for i, word in enumerate(
            ['DontCare', 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc'])}
        self.number2object = {i: word for word, i in self.object2number.items()}
        self.X = None
        self.Y = None
        self.Xtest = None
        self.Xtrain = None
        self.Ytest = None
        self.Ytrain = None

        if not os.path.exists(folder_with_images):
            print("error - {} doesn't exist".format(folder_with_images))
        if not os.path.isdir(folder_with_images):
            print("error - {} is not a directory".format(folder_with_images))
        self.folder_with_images = folder_with_images

        if not os.path.exists(folder_with_labels):
            print("error - {} doesn't exist".format(folder_with_labels))
        if not os.path.isdir(folder_with_labels):
            print("error - {} is not a directory".format(folder_with_labels))
        self.folder_with_labels = folder_with_labels

    @staticmethod
    def read_labels(path_to_label: str) -> list:
        labels = []
        if not os.path.isfile(path_to_label):
            print("wrong path {}".format(path_to_label))
            return labels
        with codecs.open(path_to_label) as f:
            labels_str = f.read().strip()

        label_str = labels_str.split("\n")
        for label_str in label_str:
            labels.append(Label(label_str))
        return labels

    @classmethod
    def cut_one_image(cls, im, label):
        delta = [0, 0, 0, 0]
        delta[0] = -min(label.x_left, 0)
        label.x_left += delta[0]
        label.x_right += delta[0]
        delta[1] = -min(label.y_top, 0)
        label.y_top += delta[1]
        label.y_bottom += delta[1]
        delta[2] = max(label.x_right - im.size[0] + 1, 0)
        delta[3] = max(label.y_bottom - im.size[1] + 1, 0)
        new_width = delta[0] + delta[2] + im.size[0]
        new_height = delta[1] + delta[3] + im.size[1]
        new_im = Image.new(im.mode, (new_width, new_height))
        new_im.paste(im, (delta[0], delta[1]))
        new_im = new_im.crop((label.x_left, label.y_top, label.x_right, label.y_bottom))
        new_im = new_im.resize([Label.size_for_cutting, Label.size_for_cutting], Image.ANTIALIAS)

        # new_im.show()
        return np.asarray(new_im)

    @classmethod
    def cut_images(cls, path_to_image: str, labels: list):
        images = []
        if not os.path.isfile(path_to_image):
            print("wrong path {}".format(path_to_image))
            return images
        im = Image.open(path_to_image)
        for label in labels:
            label.make_borders_square_with_ideal_size_or_bigger()
            images.append(cls.cut_one_image(im, label))

        return images

    def read_data(self):
        self.X = []
        self.Y = []
        images_names = sorted(os.listdir(self.folder_with_images))

        for image_name in images_names:
            label_name = image_name[:-3] + 'txt'
            labels = self.read_labels(self.folder_with_labels + label_name)
            y_for_labels = [self.object2number[label.object_type] for label in labels]
            images = self.cut_images(self.folder_with_images + image_name, labels)
            self.X = self.X + images
            self.Y = self.Y + y_for_labels

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def create_train_test(self, ratio=0.2):
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(self.X, self.Y, test_size=ratio,
                                                                            random_state=0)


if __name__ == "__main__":
    test = 2
    if test == 1:
        im = Image.open("/home/olga/my/work/aid/data/images/training/image_2/000000.png")
        im.show()
        label = Label("Car 0 0 0 0 10 1240 370")
        label.make_borders_square_with_ideal_size_or_bigger()
        new_img = DataGenerator.cut_one_image(im, label)
        new_img.show()
        print(new_img.size)

    if test == 2:
        data_generator = DataGenerator("/home/olga/my/work/aid/data/for_debug/images/",
                                       "/home/olga/my/work/aid/data/for_debug/labels/")
        data_generator.read_data()
        data_generator.shuffle()
        print(len(data_generator.Xtrain))
        print(len(data_generator.Ytrain))
