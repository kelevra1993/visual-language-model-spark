# Variables for benchmarking
image_height = 1024
image_width = 1024
keep_ratio = False
center = False
num_classes = 1
label_dictionary = {0: "data"}

import itertools
import json
import os
import shutil
import sys
import time
from random import shuffle

import cv2
from tqdm import tqdm

import numpy as np

# Confusion matrix imports
# plot a pretty confusion matrix with seaborn
# @author: Wagner Cipriano - wagnerbhbr - gmail - CEFETMG / MMC
# Link :https://github.com/wcipriano/pretty-print-confusion-matrix
import tensorflow as tf

# Disable eager mode
tf.compat.v1.disable_eager_execution()


def preprocess(image, height, width, keep_ratio=keep_ratio, center=center):
    """
    :param image: a numpy array of an image in BLUE, GREEN, RED opened by OpenCV
    :param height: desired output height
    :param width: desired output width
    :param keep_ratio: boolean that decides whether or not we keep the aspect ratio of an image
                    by default keep_ratio is False.
                    In the case where we decide to keep the aspect ratio, black bars will either be appended
                    on the right hand side of the images or on the bottom of the image to fit our desired height and width
    :return: processed_image: numpy array of desired height and width
    """
    if keep_ratio:
        image_height, image_width, _ = image.shape
        image_ratio = image_height / image_width

        if center:

            maximum_size = max([image_height, image_width])
            processed_image = np.zeros(
                (maximum_size, maximum_size, 3), dtype=image.dtype
            )

            start_x = int((maximum_size - image_width) / 2)
            start_y = int((maximum_size - image_height) / 2)

            processed_image[
                start_y: start_y + image_height, start_x: start_x + image_width, :
            ] = image[:image_height, :image_width, :]
            processed_image = cv2.resize(processed_image, (width, height))

        else:
            if image_ratio > 1:
                new_width = int(height / image_ratio)
                image = cv2.resize(image, (new_width, height))
                stack = np.zeros((height, (width - new_width), 3), dtype=np.uint8)
                processed_image = np.hstack((image, stack))

            if image_ratio < 1:
                new_height = int(width * image_ratio)
                image = cv2.resize(image, (width, new_height))
                stack = np.zeros(((height - new_height), width, 3), dtype=np.uint8)
                processed_image = np.vstack((image, stack))

            # To shrink an image, it will generally look best with CV_INTER_AREA interpolation
            # Whereas to enlarge an image, it will generally look best with CV_INTER_CUBIC (slow)
            # by default it is CV_INTER_LINEAR (faster but still looks OK)

            if image_ratio == 1:
                processed_image = cv2.resize(image, (width, height))
    else:
        processed_image = cv2.resize(image, (width, height))

    return processed_image


def convert_to_bytes(value):
    """
    :param value: any type of value
    :return: bytes_value: value converted to bytes
    """
    try:
        bytes_value = str.encode(value)
    except:
        bytes_value = value.tobytes()

    return bytes_value


def _bytes_feature(value):
    """
    :param value: input value of type bytes
    :return: a feature
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def decode_png(input):
    """
    :param input: input value of type bytes or string
    :return: decoded input(string) as an image
    """
    return tf.image.decode_png(input)


def get_paths(folder_path, label_dictionary):
    """
    :param folder_path: path to a training or validation folder
    :param label_dictionary:  Dictionary that defines Neural Network labels
            example :label_dictionary = {0: "FRONT", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    :return: image_files: paths to image files for a given folder path
    """
    image_files = []

    for k, v in label_dictionary.items():
        buf_fol = os.path.join(folder_path, v)
        buf_fil = [os.path.join(buf_fol, f) for f in os.listdir(buf_fol)]
        for element in buf_fil:
            if (
                    element.endswith((".jpg", ".png", ".jpeg", ".tiff"))
                    and os.stat(element).st_size != 0
            ):
                image_files.append(element)
    shuffle(image_files)

    return image_files


def create_tfrecord(
        folder_path, output_tfrecord, height, width, label_dictionary, keep_ratio=keep_ratio
):
    """
    :param folder_path: path of training folder or validation folder
    :param output_tfrecord: name of desired tfrecord file
    :param height: Neural Network Input Height
    :param width: Neural Network Input Width
    :param label_dictionary: Dictionary that defines Neural Network labels
            example :label_dictionary = {0: "FRONT", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    :param keep_ratio: boolean that decides whether or not we keep the aspect ratio of an image
                    by default keep_ratio is False.
    :return: storage of a tfrecord file
    """

    image_files = get_paths(folder_path=folder_path, label_dictionary=label_dictionary)

    # inverting label dictionary
    label_indices = {}
    n_classes = len(label_dictionary)
    for k, v in label_dictionary.items():
        buffer_label = np.zeros(n_classes)
        np.put(buffer_label, k, 1)
        label_indices[v] = buffer_label

    print(
        "The Tensorflow Record %s will have %d images\n"
        % (output_tfrecord.split("/")[-1], len(image_files))
    )

    buffer = "buffer-" + output_tfrecord.split("/")[-1]
    buffer_record = os.path.join(
        output_tfrecord.strip(output_tfrecord.split("/")[-1]), buffer
    )

    # First we get rid of an eventual buffer record  that was not completed
    if os.path.exists(buffer_record):
        print("\nErasing a previous tfrecord that was not completed")
        os.remove(buffer_record)

    destination_writer = tf.io.TFRecordWriter(buffer_record)

    time.sleep(1)

    for image_path in tqdm(image_files, desc='Creation of TFRecord File for %s ' % folder_path.split("/")[-1]):
        image = cv2.imread(image_path)

        image = preprocess(image, height, width, keep_ratio=keep_ratio, center=center)
        processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoded_image_string = cv2.imencode(".png", processed_image)[1].tobytes()

        image_label = os.path.basename(os.path.dirname(image_path))
        image_name = os.path.basename(image_path)
        label = label_indices[image_label]

        name_raw = convert_to_bytes(image_name)
        label_raw = convert_to_bytes(label)

        # example stored in tf record
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "input_name": _bytes_feature(name_raw),
                    "input": _bytes_feature(encoded_image_string),
                    "label": _bytes_feature(label_raw),
                }
            )
        )

        destination_writer.write(example.SerializeToString())
    # Rename the buffer tfrecord once serialization is complete
    os.rename(buffer_record, output_tfrecord)
    os.setxattr(
        output_tfrecord,
        "user.number_of_sequences",
        (len(image_files)).to_bytes(16, byteorder="big"),
    )


def get_generator(paths):
    """
    :param paths: list of paths of data used for training or validation
    :return: a generator used for tf.data.Dataset
    """

    def generator():
        for element in paths:
            yield (element.encode("utf-8"))

    return generator


def parse_function(
        element,
        height=image_height,
        width=image_width,
        label_dictionary=label_dictionary,
        keep_ratio=keep_ratio,
):
    """
    :param element: string tensor that holds path to an element of training or validation
    :param height: desired resized height
    :param width: desired resized width
    :param label_dictionary: Dictionary that defines Neural Network labels
            example :label_dictionary = {0: "FRONT", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    :param keep_ratio: boolean that defines whether we keep ratio for image training
    :return: (image_name,image,label) : name of a given image, the image and the corresponding label
    """

    image_string = element.decode("utf-8")
    image = cv2.imread(image_string)

    label_indices = {}
    n_classes = len(label_dictionary)

    for k, v in label_dictionary.items():
        buffer_label = np.zeros(n_classes)
        np.put(buffer_label, k, 1)
        label_indices[v] = buffer_label

    image = preprocess(image, height, width, keep_ratio=keep_ratio, center=center)
    image_label = image_string.split("/")[-2]

    image_name = image_string.split("/")[-1]

    label = label_indices[image_label]

    return (image_name, image, label)


def create_dataset(folder_path, label_dictionary, batch_size=1, prefetch=True):
    """
    :param folder_path: path of training/validation folder
    :param label_dictionary: Dictionary that defines Neural Network labels
            example :label_dictionary = {0: "FRONT", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}
    :param batch_size: desired batch size
    :param prefetch: choice to prefetch batch-size data in order to gain time
    :return:
    """

    paths = get_paths(folder_path=folder_path, label_dictionary=label_dictionary)

    generator = get_generator(paths=paths)

    # If one chooses to use a generator rather than a list (no perfomance difference noticed)
    dataset = tf.data.Dataset.from_generator(
        generator=generator, output_types=(tf.string)
    )
    # dataset = tf.data.Dataset.from_tensor_slices(paths)

    dataset = dataset.map(
        lambda element: tf.numpy_function(
            parse_function, [element], [tf.string, tf.uint8, tf.float64]
        )
    )

    # here we put repeat and batch size
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size=batch_size)

    if prefetch:
        dataset = dataset.prefetch(buffer_size=5)

    # Create an iterator
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

    # Get the next value
    value = iterator.get_next()

    # Will be called automatically, but won't need to if we use dataset.repeat
    init = iterator.initializer

    return init, value


def count_records(session, record):
    """
    Function that counts number of element in a tfrecord
    :param session:
    :param record:
    :return:
    """
    example_dataset = tf.data.TFRecordDataset([record])
    example_iterator = tf.compat.v1.data.make_initializable_iterator(example_dataset)

    example_element = example_iterator.get_next()
    example_initializer = example_iterator.initializer

    session.run(example_initializer)

    record_count = 0
    while True:
        try:
            _ = session.run(example_element)
            record_count += 1
        except tf.errors.OutOfRangeError:
            break

    return record_count


def read_and_decode(
        serialized_data, height=image_height, width=image_width, num_classes=num_classes
):
    """
    :param serialized_data: serialized data from a tfrecord file
    :param height:neural network input image height
    :param width: neural network input image width
    :param num_classes: Number of classes for the classification task
    :return:
    """

    features = tf.io.parse_single_example(
        serialized_data,
        features={
            "input_name": tf.io.FixedLenFeature([], tf.string),
            "input": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.string),
        },
    )

    # Decode it from Bytes(Raw information) to the suitable type
    image_name = features["input_name"]
    image = decode_png(features["input"])
    label = tf.io.decode_raw(features["label"], tf.float64)

    resized_image = tf.reshape(image, [height, width, 3])
    resized_label = tf.reshape(label, [num_classes])

    return image_name, resized_image, resized_label


if __name__ == "__main__":
    tfrecord_path = "test.tfrecord"
    train_dir = "/home/robert_kelevra/Projects/visual-language-model-spark/datasets/coco-2017/train"
    
    # # 1. Create TFRecord
    # create_tfrecord(
    #     folder_path=train_dir,
    #     output_tfrecord=tfrecord_path,
    #     height=image_height,
    #     width=image_width,
    #     label_dictionary=label_dictionary
    # )
    
    # 2. Read TFRecord
    dataset = tf.data.TFRecordDataset([tfrecord_path])
    dataset = dataset.map(lambda x: read_and_decode(x, height=image_height, width=image_width, num_classes=num_classes))
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    next_element = iterator.get_next()
    
    total_images = int.from_bytes(os.getxattr(tfrecord_path, "user.number_of_sequences"), byteorder="big")
    
    show_images = True
    
    with tf.compat.v1.Session() as sess:
        sess.run(iterator.initializer)
        for _ in tqdm(range(total_images), desc="Reading TFRecord"):
            try:
                image_name, image, label = sess.run(next_element)
                
                # if show_images:
                #     # Convert back to BGR for OpenCV display
                #     cv2.imshow("TFRecord Viewer", image)
                #     # Press 'q' to quit early
                #     if cv2.waitKey(0) & 0xFF == ord('q'):
                #         break

            except tf.errors.OutOfRangeError:
                break
        
        if show_images:
            cv2.destroyAllWindows()
