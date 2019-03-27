from keras import backend as K
from keras.preprocessing import image
from random import shuffle
import numpy as np
import numpy.random as rng
import os


class SiameseDirectoryIterator(image.DirectoryIterator):
    def __init__(self, directory, image_data_generator,
                 bounding_boxes: dict = None, landmark_info: dict = None, attr_info: dict = None,
                 num_landmarks=26, num_attrs=463,
                 target_size=(256, 256), color_mode: str = 'rgb',
                 classes=None, class_mode: str = 'categorical',
                 batch_size: int = 32, shuffle: bool = True, seed=None, data_format=None,
                 follow_links: bool = False, testing: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, follow_links)
        self.testing = testing
        self.bounding_boxes = bounding_boxes
        self.landmark_info = landmark_info
        self.attr_info = attr_info
        self.num_landmarks = num_landmarks
        self.num_attrs = num_attrs
        self.num_bbox = 4

    def next(self):
        """
        # Returns
            The next batch.
        """

        batch_x = np.zeros((self.batch_size,) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (self.num_bbox,), dtype=K.floatx())
        landmarks = np.zeros((len(batch_x),) + (self.num_landmarks,), dtype=K.floatx())
        attributes = np.zeros((len(batch_x),) + (self.num_attrs,), dtype=K.floatx())

        # initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((self.batch_size, self.target_size[0], self.target_size[1], 3)) for _ in range(2)]
        # initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((self.batch_size,))

        if not self.testing:
            targets[self.batch_size // 2:] = 1
        else:
            targets[-1] = 1
        idx_1 = rng.randint(0, self.samples)

        for i in range(self.batch_size):
            if not self.testing:
                idx_1 = rng.randint(0, self.samples)
            fname_1 = self.filenames[idx_1]
            print("\nPairs:")
            print("Category: " + str(self.classes[idx_1]) + ", Filename: " + str(fname_1))
            img_1 = image.load_img(os.path.join(self.directory, fname_1),
                                   grayscale=self.color_mode == 'grayscale',
                                   target_size=self.target_size)
            img_1 = image.img_to_array(img_1, data_format=self.data_format)
            img_1 = self.image_data_generator.random_transform(img_1)
            img_1 = self.image_data_generator.standardize(img_1)

            pairs[0][i, :, :, :] = img_1

            idx_2 = rng.randint(0, self.samples)
            # pick images of same class for 1st half, different for 2nd
            if not self.testing:
                if i >= self.batch_size // 2:
                    print("Same class")
                    while self.classes[idx_2] != self.classes[idx_1]:
                        idx_2 = rng.randint(0, self.samples)
                    # category_2 = category
                else:
                    print("Different class")
                    # add a random number to the category modulo n classes to ensure 2nd image has
                    # ..different category
                    while self.classes[idx_2] == self.classes[idx_1]:
                        idx_2 = rng.randint(0, self.samples)
                # category_2 = (category + rng.randint(1, self.num_classes)) % self.num_classes
            else:
                if i == self.batch_size-1:
                    print("Same class")
                    while self.classes[idx_2] != self.classes[idx_1]:
                        idx_2 = rng.randint(0, self.samples)
                else:
                    print("Different class")
                    while self.classes[idx_2] == self.classes[idx_1]:
                        idx_2 = rng.randint(0, self.samples)

            fname_2 = self.filenames[idx_2]
            print("Category: " + str(self.classes[idx_2]) + ", Filename: " + str(fname_2) + "\n")
            img_2 = image.load_img(os.path.join(self.directory, fname_2),
                                   grayscale=self.color_mode == 'grayscale',
                                   target_size=self.target_size)
            img_2 = image.img_to_array(img_2, data_format=self.data_format)
            img_2 = self.image_data_generator.random_transform(img_2)
            img_2 = self.image_data_generator.standardize(img_2)
            pairs[1][i, :, :, :] = img_2

            if self.bounding_boxes is not None:
                locations[i] = (self.get_bbox(fname_1), self.get_bbox(fname_2))

            if self.landmark_info is not None:
                landmarks[i] = (self.get_landmark_info(fname_1), self.get_landmark_info(fname_1))

            if self.attr_info is not None:
                attr_info_lst_1 = self.attr_info[fname_1]
                attr_info_lst_2 = self.attr_info[fname_2]
                attributes[i] = (np.asarray(attr_info_lst_1), np.asarray(attr_info_lst_2))

        y = [targets, locations, landmarks, attributes]
        statements = [True, self.bounding_boxes is not None,
                      self.landmark_info is not None, self.attr_info is not None]

        y = np.asarray([x for x, y in zip(y, statements) if y], dtype=K.floatx()).reshape((self.batch_size,))

        if self.shuffle:
            negs = pairs[0]
            poss = pairs[1]
            tmp = list(zip(negs, poss, y))
            shuffle(tmp)
            negs, poss, y = zip(*tmp)

            pairs[0] = np.asarray(negs)
            pairs[1] = np.asarray(poss)

        return np.asarray(pairs), np.asarray(y)

    def get_bbox(self, fname):
        bbox = self.bounding_boxes[fname]
        return np.asarray([bbox['origin']['x'], bbox['origin']['y'], bbox['width'], bbox['height']], dtype=K.floatx())

    def get_landmark_info(self, fname):
        landmark_info = self.landmark_info[fname]
        return np.asarray([landmark_info["clothes_type"], landmark_info["variation_type"],
                           landmark_info['1']['visibility'], landmark_info['1']['x'],
                           landmark_info['1']['y'],
                           landmark_info['2']['visibility'], landmark_info['2']['x'],
                           landmark_info['2']['y'],
                           landmark_info['3']['visibility'], landmark_info['3']['x'],
                           landmark_info['3']['y'],
                           landmark_info['4']['visibility'], landmark_info['4']['x'],
                           landmark_info['4']['y'],
                           landmark_info['5']['visibility'], landmark_info['5']['x'],
                           landmark_info['5']['y'],
                           landmark_info['6']['visibility'], landmark_info['6']['x'],
                           landmark_info['6']['y'],
                           landmark_info['7']['visibility'], landmark_info['7']['x'],
                           landmark_info['7']['y'],
                           landmark_info['8']['visibility'], landmark_info['8']['x'],
                           landmark_info['8']['y']], dtype=K.floatx())

