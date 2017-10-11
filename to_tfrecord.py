from random import shuffle
from os import listdir
import json

shuffle_data = True
dataset_path = '../../datasets/street2shop/'

meta_path = dataset_path + 'meta/json/'
images_path = dataset_path + '/images/'

meta_files = listdir(meta_path)

for filename in meta_files:
    if 'train' in filename:
        fin = open(meta_path + filename, 'r')
        train_data = json.loads(fin.read())
        fin.close()

        print len(train_data),train_data[0]

        break