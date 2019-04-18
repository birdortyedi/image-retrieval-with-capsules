import os
import re
import shutil
splitter = re.compile("\s+")
base_path = "./data/"


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


eval_partioner()

