import os
import numpy as np
import tensorflow as tf
import random

class RoeDeerDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, data_path, set_name, batch_size, num_classes, balance_classes = False):
        self.data_path = data_path
        self.set_name = set_name
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.image_directory = os.path.join(self.data_path, "images", self.set_name)
        self.image_filenames = os.listdir(self.image_directory)

        self.label_directory = os.path.join(self.data_path, "labels", self.set_name) 

        if balance_classes:
            self.__balance_classes()

    def __balance_classes(self):
        samples_per_class = {}
        for filename in self.image_filenames:
            filename = filename.split(".")[0]
            label, _ = self.__load_label(filename)
            
            if not label in samples_per_class:
                samples_per_class[label] = []
            samples_per_class[label].append(filename)

        max_num_per_class = 0
        for key in samples_per_class:
            if len(samples_per_class[key]) > max_num_per_class:
                max_num_per_class = len(samples_per_class[key])

        for key in samples_per_class:
            num_samples = len(samples_per_class[key])
            to_pick = max_num_per_class - num_samples
            picked = random.choices(samples_per_class[key], k = to_pick)
            self.image_filenames.extend(picked)

        random.shuffle(self.image_filenames)

    def __load_image(self, name):
        image_directory = os.path.join(self.data_path, "images", self.set_name)
        image_path = os.path.join(image_directory, "{}.jpg".format(name))
        image = tf.keras.preprocessing.image.load_img(image_path)
        image_arr = tf.keras.preprocessing.image.img_to_array(image)
        image_arr = image_arr/255.
        return image_arr

    def __load_label(self, name):
        label_path = os.path.join(self.label_directory,"{}.txt".format(name))
        with open(label_path, "r") as label_file:
            label = int(label_file.read())
        return label, tf.keras.utils.to_categorical(label, num_classes = self.num_classes)

    def __load_month(self, name):
        basename = name.split("_")[0]
        month_directory = os.path.join(self.data_path, "months", self.set_name)
        month_path = os.path.join(month_directory,"{}.month".format(basename))
        with open(month_path, "r") as month_file:
            month = int(month_file.read())

        month_array = np.zeros((1))
        month_array[0] = month

        return month_array
    
    def on_epoch_end(self):
        random.shuffle(self.image_filenames)
    
    def __getitem__(self, index):
        batch_filenames = self.image_filenames[index * self.batch_size : (index + 1) * self.batch_size]
       
        images = []
        labels = []
        months = []
        for filename in batch_filenames:
            filename = filename.split(".")[0]
            
            images.append(self.__load_image(filename))
            labels.append(self.__load_label(filename)[1])
            months.append(self.__load_month(filename))

        image_out = np.stack(images)
        label_out = np.stack(labels)
        month_out = np.stack(months)

        return tuple([image_out, month_out]), label_out

    def __len__(self):
        return len(self.image_filenames) // self.batch_size

