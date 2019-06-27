import os
import re
import shutil
from colorama import Fore
from tqdm import tqdm
import csv
from PIL import Image
import glob
import numpy as np
from keras.preprocessing import image

splitter = re.compile("\s+")
base_path = "./data_c2s/"  # "./data/"


def eval_partioner():
        # Read the relevant annotation file and preprocess it
        # Assumed that the annotation files are under '<project folder>/data/anno' path
        with open(os.path.join(base_path, 'Eval/list_eval_partition.txt'), 'r') as eval_partition_file:
            list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
            list_eval_partition = [splitter.split(line) for line in list_eval_partition]
            print(list_eval_partition)
            list_all = [(v[0], v[0].split('/')[2], v[1], v[2], v[0].split('/')[1]) for v in list_eval_partition]
            print(list_all)

        new_path = os.path.join(os.path.join(base_path, "img"), "BOTH")

        if not os.path.exists(new_path):
            os.mkdir(new_path)

        # Put each image into the relevant folder in train/test/validation folder
        for element in list_all:
            if not os.path.exists(os.path.join(new_path, element[3])):
                os.mkdir(os.path.join(new_path, element[3]))
            if not os.path.exists(os.path.join(os.path.join(new_path, element[3]), element[4]+"_"+element[1])):
                os.mkdir(os.path.join(os.path.join(new_path, element[3]), element[4]+"_"+element[1]))
            if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(new_path, element[3]),
                                                                         element[4] + "_" + element[1])),
                                  element[2])):
                os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(new_path, element[3]),
                                                                element[4]+"_"+element[1])),
                         element[2]))
            shutil.move(os.path.join(base_path, element[0]),
                        os.path.join(os.path.join(os.path.join(new_path, element[3]), element[4]+"_"+element[1]),
                                     element[2]))


def eval_partioner_c2s():
    data_path = "/home/birdortyedi/Downloads/"

    # Read the relevant annotation file and preprocess it
    # Assumed that the annotation files are under '<project folder>/data/anno' path
    with open(os.path.join(base_path, 'Eval/list_eval_partition.txt'), 'r') as eval_partition_file:
        list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
        list_eval_partition = [splitter.split(line) for line in list_eval_partition]
        # print(list_eval_partition)
        list_train = [(os.path.join(os.path.join(line[0].split('/')[0], line[3]), "/".join(line[0].split('/')[1:])),
                       line[0], line[0].split('/')[4],
                       os.path.join(os.path.join(line[1].split('/')[0], line[3]), "/".join(line[1].split('/')[1:])),
                       line[1], line[1].split('/')[4], line[2], line[3])
                      for line in list_eval_partition if line[3] == "train"]
        # print(list_train)

        list_test = [(os.path.join(os.path.join(line[0].split('/')[0], line[3]), "/".join(line[0].split('/')[1:])),
                      line[0], line[0].split('/')[4],
                      os.path.join(os.path.join(line[1].split('/')[0], line[3]), "/".join(line[1].split('/')[1:])),
                      line[1], line[1].split('/')[4], line[2], line[3])
                     for line in list_eval_partition if line[3] == "test"]
        # print(list_test)

    new_path = os.path.join(base_path, "img")
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for element in list_test:
        if not os.path.exists(os.path.join(new_path, "test")):
            os.mkdir(os.path.join(new_path, "test"))

        top_cat_1 = element[1].split('/')[1]
        bot_cat_1 = element[1].split('/')[2]
        idx_1 = element[1].split('/')[3]
        top_cat_2 = element[4].split('/')[1]
        bot_cat_2 = element[4].split('/')[2]
        idx_2 = element[4].split('/')[3]

        if not os.path.exists(os.path.join(os.path.join(new_path, "test"), top_cat_1)):
            os.mkdir(os.path.join(os.path.join(new_path, "test"), top_cat_1))

        if not os.path.exists(os.path.join(os.path.join(new_path, "test"), top_cat_2)):
            os.mkdir(os.path.join(os.path.join(new_path, "test"), top_cat_2))

        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                                     top_cat_1)),
                                           bot_cat_1)):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                            top_cat_1)),
                                  bot_cat_1))

        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                                     top_cat_2)),
                                           bot_cat_2)):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                            top_cat_2)),
                                  bot_cat_2))

        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                                                  top_cat_1)),
                                                        bot_cat_1),
                                           idx_1)):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                                         top_cat_1)),
                                               bot_cat_1),
                                  idx_1))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                                                  top_cat_2)),
                                                        bot_cat_2),
                                           idx_2)):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(os.path.join(new_path, "test"),
                                                                         top_cat_2)),
                                               bot_cat_2),
                                  idx_2))

        if not os.path.isfile(os.path.join(base_path, element[0])):
            shutil.copy(os.path.join(data_path, element[1]), os.path.join(base_path, element[0]))

        if not os.path.isfile(os.path.join(base_path, element[3])):
            shutil.copy(os.path.join(data_path, element[4]), os.path.join(base_path, element[3]))


def extract_neg_hard_pairs():
    path = os.path.join(os.path.join(os.path.join(base_path, "img"), "BOTH"), "train")

    datagen_anchor = image.ImageDataGenerator(rescale=1 / 255.)
    datagen_negative = image.ImageDataGenerator(rescale=1 / 255.)

    iterator_anchor = image.DirectoryIterator(directory=path, image_data_generator=datagen_anchor,
                                              batch_size=1, target_size=(64, 64))
    iterator_negative = image.DirectoryIterator(directory=path, image_data_generator=datagen_negative,
                                                batch_size=1, target_size=(64, 64))

    def euclidean_dist(a, n):
        return np.sum(np.square((a - n))) / 64

    for i in tqdm(range(len(iterator_anchor)), ncols=100, desc="Training",
                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        info_anchor = iterator_anchor.filenames[i].split("/")
        cls_idx_anchor = info_anchor[0]
        item_idx_anchor = info_anchor[1]
        item_name_anchor = info_anchor[2]

        img_anchor, y_anchor = next(iterator_anchor)
        closest_dist = 1e9
        closest_cls_idx = -1
        closest_item_idx = -1
        closest_item_name = ""
        dist = 0
        for j in range(len(iterator_negative)):
            info_negative = iterator_negative.filenames[j].split("/")
            cls_idx_negative = info_negative[0]
            item_idx_negative = info_negative[1]
            item_name_negative = info_negative[2]

            img_negative, y_negative = next(iterator_negative)
            # dist = euclidean_dist(img_anchor, img_negative)

            if cls_idx_anchor != cls_idx_negative and closest_dist > dist:
                closest_dist = dist
                closest_cls_idx = cls_idx_negative
                closest_item_idx = item_idx_negative
                closest_item_name = item_name_negative

        pair_dict = {"c_idx_anc": cls_idx_anchor,
                     "c_idx_neg": closest_cls_idx,
                     "i_idx_anc": item_idx_anchor,
                     "i_idx_neg": closest_item_idx,
                     "i_name_anc": item_name_anchor,
                     "i_name_neg": closest_item_name,
                     "distance": closest_dist}

        print(pair_dict)

        with open('./data/Anno/hard_neg_pairs.csv', 'w') as output_file:
            writer = csv.writer(output_file)
            for key, value in pair_dict.items():
                writer.writerow([key, value])


# eval_partioner()
eval_partioner_c2s()
# extract_neg_hard_pairs()