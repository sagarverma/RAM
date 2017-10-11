from random import shuffle
from os import listdir
import json
import cv2
import tensorflow as tf 
import sys 

class_map= {'bags':0,'belts':1,'dresses':2,'eyewear':3,'footwear':4,
    'hats':5,'leggings':6,'outerwear':7,'pants':8,'skirts':9,'tops':10}


shuffle_data = True
dataset_path = '../../datasets/street2shop/'

meta_path = dataset_path + 'meta/json/'
images_path = dataset_path + '/images/'

meta_files = listdir(meta_path)

train_images = []
train_labels = []

for filename in meta_files:
    if 'train' in filename:
        fin = open(meta_path + filename, 'r')
        train_data = json.loads(fin.read())
        fin.close()

        for dat in train_data:
            train_images.append(dat['photo'])
            train_images.append(dat['product'])
            train_labels.append(class_map[filename[12:-5]])
            train_labels.append(class_map[filename[12:-5]])

if shuffle_data:
    c = list(zip(train_images, train_labels))
    shuffle(c)
    train_images, train_labels =zip(*c)

test_images = []
test_labels = []

for filename in meta_files:
    if 'test' in filename:
        fin = open(meta_path + filename, 'r')
        test_data = json.loads(fin.read())
        fin.close()

        for dat in test_data:
            test_images.append(dat['photo'])
            test_images.append(dat['product'])
            test_labels.append(class_map[filename[11:-5]])
            test_labels.append(class_map[filename[11:-5]])


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = dataset_path + 'street2shop_train.tfrecords'

writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_images)):
    if not i % 1000:
        print 'Train data: {}/{}'.format(i, len(train_images))
        sys.stdout.flush()

    img = load_image(images_path + train_images[i] + '.jpg')
    label = train_labels[i]

    # Create a feature
    feature = {'train/label': _int64_feature(label),
               'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()  

test_filename = dataset_path + 'street2shop_test.tfrecords'  # address to save the TFRecords file

writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(test_images)):
    # print how many images are saved every 1000 images
    if not i % 1000:
        print 'Test data: {}/{}'.format(i, len(test_images))
        sys.stdout.flush()
    # Load the image
    img = load_image(test_images[i])
    label = test_labels[i]
    # Create a feature
    feature = {'test/label': _int64_feature(label),
               'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
writer.close()
sys.stdout.flush()

