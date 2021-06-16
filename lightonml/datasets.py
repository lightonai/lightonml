# -*- coding: utf8
"""This module contains functions to load some common datasets. All datasets return tuples of train and test examples
and labels. Grayscale images have shape (height, width), RGB images have shape (3, height, width).

All functions use the data location provided by `lightonml.utils.get_ml_data_dir_path`.

This location can be defined from the following sources (listed in decreasing priority):

 * `LIGHTONML_DATA_DIR` environment variable
 * `lightonml.set_ml_data_dir()` function
 * `~/.lighton.json` file
 * `/etc/lighton.json` file
 * `/etc/lighton/host.json` file
"""

import gzip
import itertools
import os
import pickle
import tarfile

import numpy as np

from lightonml.utils import get_ml_data_dir_path, download


def MNIST():
    """Data loader for the MNIST dataset.

    Returns
    -------
    (X_train, y_train) : tuple of np.ndarray of np.uint8, of shape (60000, 28, 28) and (60000,)
        train flattened MNIST images and labels.
    (X_test, y_test) : tuple of np.ndarray of np.uint8, of shape (10000, 28, 28) and (10000,)
        test flattened MNIST images and labels.
    """
    urls = [
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
        'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz'
    ]

    data_home = get_ml_data_dir_path()

    if not (data_home/'MNIST').is_dir():
        os.mkdir(str(data_home/'MNIST'))
        for url in urls:
            print('Downloading {}'.format(url))
            download(url, str(data_home / 'MNIST'))

    paths = []
    for url in urls:
        paths.append(str(data_home / 'MNIST' / url.split('/')[-1]))
    y_train = np.frombuffer(gzip.open(paths[1], 'rb').read(), np.uint8, offset=8)
    X_train = np.frombuffer(gzip.open(paths[0], 'rb').read(),
                            np.uint8, offset=16).reshape(-1, 28, 28)
    y_test = np.frombuffer(gzip.open(paths[3], 'rb').read(), np.uint8, offset=8)
    X_test = np.frombuffer(gzip.open(paths[2], 'rb').read(),
                           np.uint8, offset=16).reshape(-1, 28, 28)
    return (X_train, y_train), (X_test, y_test)


def FashionMNIST():
    """Data Loader for the FashionMNIST dataset.

    Returns
    -------
    (X_train, y_train) : tuple of np.ndarray of np.uint8, of shape (60000, 28, 28) and (60000,)
        train flattened FashionMNIST images and labels.
    (X_test, y_test) : tuple of np.ndarray of np.uint8, of shape (10000, 28, 28) and (10000,)
        test flattened FashionMNIST images and labels.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]

    data_home = get_ml_data_dir_path()

    if not (data_home/'FashionMNIST').is_dir():
        os.mkdir(str(data_home/'FashionMNIST'))
        for url in urls:
            print('Downloading {}'.format(url))
            download(url, str(data_home / 'FashionMNIST'))

    paths = []
    for url in urls:
        paths.append(str(data_home / 'FashionMNIST' / url.split('/')[-1]))
    y_train = np.frombuffer(gzip.open(paths[1], 'rb').read(), np.uint8, offset=8)
    X_train = np.frombuffer(gzip.open(paths[0], 'rb').read(),
                            np.uint8, offset=16).reshape(-1, 28, 28)
    y_test = np.frombuffer(gzip.open(paths[3], 'rb').read(), np.uint8, offset=8)
    X_test = np.frombuffer(gzip.open(paths[2], 'rb').read(),
                           np.uint8, offset=16).reshape(-1, 28, 28)
    return (X_train, y_train), (X_test, y_test)


def SignMNIST():
    """Data Loader for the SignMNIST dataset. Each training and test case represents
    a label (0-25) as a one-to-one map for each alphabetic letter A-Z.

    https://www.kaggle.com/datamunge/sign-language-mnist/home

    Returns
    -------
    (X_train, y_train) : tuple of np.ndarray of np.uint8, of shape (27455, 784) and (27455,)
        train flattened SignMNIST images and labels.
    (X_test, y_test) : tuple of np.ndarray of np.uint8, of shape (7172, 784) and (7172,)
        test flattened SignMNIST images and labels.
    """
    csvfiles = ['sign_mnist_train.csv', 'sign_mnist_test.csv']
    data_home = get_ml_data_dir_path()

    if not (data_home / 'SignMNIST').is_dir():
        raise FileNotFoundError('Download and unzip the dataset from {} in a folder SignMNIST inside {}'.format(
            'https://www.kaggle.com/datamunge/sign-language-mnist/home', data_home
        ))

    data = np.genfromtxt(str(data_home / 'SignMNIST' / csvfiles[0]),
                         dtype=np.uint8, skip_header=True, delimiter=',')
    X_train = data[:, 1:]
    y_train = data[:, 0]
    data = np.genfromtxt(str(data_home / 'SignMNIST' / csvfiles[1]),
                         dtype=np.uint8, skip_header=True, delimiter=',')
    X_test = data[:, 1:]
    y_test = data[:, 0]

    return (X_train, y_train), (X_test, y_test)


def STL10(unlabeled=False):
    """Data Loader for the STL10 dataset.

    Parameters
    ----------
    unlabeled: bool, default to False,
        if `True` returns also the unlabeled part of the dataset

    Returns
    -------
    (X_train, y_train) : tuple of np.ndarray of np.uint8, of shape (5000, 3, 96, 96) and (5000,)
        train STL10 images and labels.
    (X_test, y_test) : tuple of np.ndarray of np.uint8, of shape (8000, 3, 96, 96) and (8000,)
        test STL10 images and labels.
    X_unlabeled: np.ndarray of np.uint8, of shape (100000, 3, 96, 96),
        unlabeled images from STL10.
    """
    url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

    data_home = get_ml_data_dir_path()
    file_name = url.split('/')[-1]
    file_path = data_home / 'STL10' / file_name

    if not (data_home / 'STL10').is_dir():
        os.mkdir(str(data_home / 'STL10'))
        print('Downloading {}'.format(url))
        download(url, str(data_home / 'STL10'))
        tarfile.open(str(file_path), 'r:gz').extractall(str(data_home / 'STL10'))

    binary_data_path = data_home / 'STL10/stl10_binary'
    train_images_path = str(binary_data_path / 'train_X.bin')
    test_images_path = str(binary_data_path / 'test_X.bin')
    train_labels_path = str(binary_data_path / 'train_y.bin')
    test_labels_path = str(binary_data_path / 'test_y.bin')

    with open(train_images_path, 'rb') as f:
        train_images = np.fromfile(f, dtype=np.uint8)
        X_train = np.reshape(train_images, (-1, 3, 96, 96))
    with open(test_images_path, 'rb') as f:
        test_images = np.fromfile(f, dtype=np.uint8)
        X_test = np.reshape(test_images, (-1, 3, 96, 96))
    with open(train_labels_path, 'rb') as f:
        y_train = np.fromfile(f, dtype=np.uint8)
    with open(test_labels_path, 'rb') as f:
        y_test = np.fromfile(f, dtype=np.uint8)

    if unlabeled:
        unlabeled_images_path = str(binary_data_path / 'unlabeled_X.bin')
        with open(unlabeled_images_path, 'rb') as f:
            unlabeled_images = np.fromfile(f, dtype=np.uint8)
            X_unlabeled = np.reshape(unlabeled_images, (-1, 3, 96, 96))
        return (X_train, y_train), (X_test, y_test), X_unlabeled

    return (X_train, y_train), (X_test, y_test)


def CIFAR10():
    """Data Loader for the CIFAR10 dataset.

    Returns
    -------
    (X_train, y_train) : tuple of np.ndarray of np.uint8, of shape (50000, 3, 96, 96) and (50000,)
        train CIFAR10 images and labels.
    (X_test, y_test) : tuple of np.ndarray of np.uint8, of shape (10000, 3, 96, 96) and (10000,)
        test CIFAR10 images and labels.
    """

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    data_home = get_ml_data_dir_path()
    file_name = url.split('/')[-1]
    file_path = data_home / 'CIFAR10' / file_name

    if not (data_home / 'CIFAR10').is_dir():
        os.mkdir(str(data_home / 'CIFAR10'))
        print('Downloading {}'.format(url))
        download(url, str(data_home / 'CIFAR10'))
        tarfile.open(str(file_path), 'r:gz').extractall(str(data_home / 'CIFAR10'))

    binary_data_path = data_home / 'CIFAR10/cifar-10-batches-py/'
    batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

    train_images = []
    train_labels = []
    for batch_file in batches:
        fo = open(str(binary_data_path / batch_file), 'rb')
        batch = pickle.load(fo, encoding='latin1')
        train_images.append(batch['data'])
        train_labels.append(batch['labels'])

    test_fo = open(str(binary_data_path / 'test_batch'), 'rb')
    test_batch = pickle.load(test_fo, encoding='latin1')
    test_images = test_batch['data']
    test_labels = test_batch['labels']

    # noinspection PyArgumentList
    X_train = np.concatenate(train_images, axis=0).reshape(-1, 3, 32, 32)
    X_test = test_images.reshape(-1, 3, 32, 32)

    y_train = np.concatenate(train_labels)
    y_test = np.asarray(test_labels)

    return (X_train, y_train), (X_test, y_test)


def CIFAR100():
    """Data Loader for the CIFAR100 dataset.

    Returns
    -------
    (X_train, y_train) : tuple of np.ndarray of np.uint8, of shape (50000, 3, 96, 96) and (50000,)
        train CIFAR100 images and labels.
    (X_test, y_test) : tuple of np.ndarray of np.uint8, of shape (10000, 3, 96, 96) and (10000,)
        test CIFAR100 images and labels.
    """

    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    data_home = get_ml_data_dir_path()
    file_name = url.split('/')[-1]
    file_path = data_home / 'CIFAR100' / file_name

    if not (data_home / 'CIFAR100').is_dir():
        os.mkdir(str(data_home / 'CIFAR100'))
        print('Downloading {}'.format(url))
        download(url, str(data_home / 'CIFAR100'))
        tarfile.open(str(file_path), 'r:gz').extractall(str(data_home / 'CIFAR100'))

    binary_data_paths = [str(data_home / 'CIFAR100/cifar-100-python/train'),
                         str(data_home / 'CIFAR100/cifar-100-python/test')]

    fo = open(binary_data_paths[0], 'rb')
    train_data = pickle.load(fo, encoding='latin1')
    train_images = train_data['data']
    train_labels = train_data['fine_labels']

    fo = open(binary_data_paths[1], 'rb')
    test_data = pickle.load(fo, encoding='latin1')
    test_images = test_data['data']
    test_labels = test_data['fine_labels']

    X_train = train_images.reshape(-1, 3, 32, 32)
    X_test = test_images.reshape(-1, 3, 32, 32)

    y_train = np.asarray(train_labels)
    y_test = np.asarray(test_labels)

    return (X_train, y_train), (X_test, y_test)


def parse_line(line, sep=','):
    line = line.split(sep)
    fields = [element.strip() for element in line]
    if len(fields) > 3:
        fields = fields[:3]
    return fields


def movielens100k(processed=False):
    """Data Loader for the Movielens-100k dataset. It consists of 100,000 ratings (1-5) from 943 users on 1682 movies.

    F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on
    Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
    DOI=http://dx.doi.org/10.1145/2827872

    Parameters
    ----------
    processed: bool, default False,
        if False, returns the raw data in a list of lists. If True, the user-item ratings matrix of shape (943, 1682)

    Returns
    -------
    ratings: depending on the value of `processed`, a list of lists or user-item ratings matrix (np.ndarray)
    """
    data_home = get_ml_data_dir_path()
    if processed:
        data_path = data_home / 'movielens-100k' / 'ratings_mat.npy'
        ratings = np.load(str(data_path))
    else:
        data_path = data_home / 'movielens-100k' / 'u.data'
        with open(str(data_path)) as f:
            ratings = [parse_line(line, sep='\t') for line in itertools.islice(f, 0, None)]
        header = ['userId', 'movieId', 'rating']
        ratings.insert(0, header)
    return ratings


def movielens20m(processed=False, id_to_movie=False):
    """Data Loader for the Movielens-20m dataset. It consists of 20000263 ratings (1-5) from 138493 users on 27278
    movies.

    F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on
    Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
    DOI=http://dx.doi.org/10.1145/2827872

    Parameters
    ----------
    processed: bool, default False,
        if False, returns the raw data in a list of lists. If True, the user-item ratings matrix of shape
        (138493, 27278)
    id_to_movie: bool, default False,
        if True returns also the mapping from movieId to movie name.

    Returns
    -------
    ratings: depending on the value of `processed`, a list of lists or user-item ratings matrix (np.ndarray)
    id_to_movie_mapping: None or list of lists,
        mapping between movieId and movie name.
    """
    data_home = get_ml_data_dir_path()
    if processed:
        data_path = data_home / 'movielens-20m' / 'ratings_mat.npy'
        ratings = np.load(str(data_path))
    else:
        data_path = data_home / 'movielens-20m' / 'ratings.csv'
        with open(str(data_path)) as f:
            ratings = [parse_line(line) for line in itertools.islice(f, 0, None)]
    id_to_movie_mapping = None
    if id_to_movie:
        data_path = data_home / 'movielens-20m' / 'movies.csv'
        with open(str(data_path)) as f:
            id_to_movie_mapping = [parse_line(line) for line in itertools.islice(f, 0, None)]
    return ratings, id_to_movie_mapping
