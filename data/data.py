import itertools

import numpy as np
import pandas as pd
import tensorflow as tf


def _convert_to_float32(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    return x, y


def get_mnist_data_as_numpy():
    """Returns numpy arrays of images and labels"""
    mnist = tf.keras.datasets.mnist
    (img_train, label_train), (img_test, label_test) = mnist.load_data()
    img_train, img_test = img_train / 255.0, img_test / 255.0
    img_train = img_train[..., tf.newaxis]
    img_test = img_test[..., tf.newaxis]
    return img_train, label_train, img_test, label_test


def get_mnist_dataset(
        count_train,
        count_test,
        buffer_size,
        batch_size):
    """Returns tf.data.Dataset instance for the mnist datasets.
    Iterating over it, we get (image,label) batches.
    """
    if count_train > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for training." % count_train)
    if count_test > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for testing." % count_test)
    img_train, label_train, img_test, label_test = get_mnist_data_as_numpy()
    ds_train = tf.data.Dataset.from_tensor_slices((img_train, label_train)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices((img_test, label_test)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)
    return ds_train, ds_test


def get_mnist_op_dataset(
        count_train,
        count_test,
        buffer_size,
        batch_size,
        n_operands=2,
        op=lambda args: args[0] + args[1]):
    """Returns tf.data.Dataset instance for an operation with the numbers of the mnist dataset.
    Iterating over it, we get (image_x1,...,image_xn,label) batches
    such that op(image_x1,...,image_xn)= label.

    Args:
        n_operands: The number of sets of images to return,
            that is the number of operands to the operation.
        op: Operation used to produce the label.
            The lambda arguments must be a list from which we can index each operand.
            Example: lambda args: args[0] + args[1]
    """
    if count_train * n_operands > 60000:
        raise ValueError("The MNIST dataset comes with 60000 training examples. \
            Cannot fetch %i examples for each %i operands for training." % (count_train, n_operands))
    if count_test * n_operands > 10000:
        raise ValueError("The MNIST dataset comes with 10000 test examples. \
            Cannot fetch %i examples for each %i operands for testing." % (count_test, n_operands))

    img_train, label_train, img_test, label_test = get_mnist_data_as_numpy()

    img_per_operand_train = [img_train[i * count_train:i * count_train + count_train] for i in range(n_operands)]
    label_per_operand_train = [label_train[i * count_train:i * count_train + count_train] for i in range(n_operands)]
    label_result_train = np.apply_along_axis(op, 0, label_per_operand_train)
    img_per_operand_test = [img_test[i * count_test:i * count_test + count_test] for i in range(n_operands)]
    label_per_operand_test = [label_test[i * count_test:i * count_test + count_test] for i in range(n_operands)]
    label_result_test = np.apply_along_axis(op, 0, label_per_operand_test)

    ds_train = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_train) + (label_result_train,)) \
        .take(count_train).shuffle(buffer_size).batch(batch_size)
    ds_test = tf.data.Dataset.from_tensor_slices(tuple(img_per_operand_test) + (label_result_test,)) \
        .take(count_test).shuffle(buffer_size).batch(batch_size)

    return ds_train, ds_test


def get_data(dataset, info_data=None):
    """
    Get data for the specified dataset.

    Parameters:
    - dataset (str): The name of the dataset to retrieve.
    - info_data (dict): Additional information required for data retrieval.

    Returns:
    - dict: A dictionary containing the retrieved data.
    """
    returned_data = {}
    if dataset == "binaryc":
        rng = np.random.default_rng(info_data["data_seed"])
        # # Data
        # Sample data from [0,1]^2.
        # The groundtruth positive is data close to the center (.5,.5) (given a threshold)
        # All the other data is considered as negative examples
        nr_samples = 100
        batch_size = info_data["batch_size"]
        data = rng.uniform([0, 0], [1, 1], (nr_samples, 2))
        labels = np.sum(np.square(data - [.5, .5]), axis=1) < .09

        # 50 examples for training; 50 examples for testing
        ds_train = tf.data.Dataset.from_tensor_slices((data[:50], labels[:50])).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((data[50:], labels[50:])).batch(batch_size)

        returned_data["ds_train"] = ds_train
        returned_data["ds_test"] = ds_test

    elif dataset == "clustering":
        rng = np.random.default_rng(info_data["data_seed"])

        nr_of_clusters = 4
        nr_of_points_x_cluster = 50
        clst_ids = range(nr_of_clusters)

        margin = .2
        mean = [rng.uniform([-1 + margin, -1 + margin], [0 - margin, 0 - margin], 2),
                 rng.uniform([0 + margin, -1 + margin], [1 - margin, 0 - margin], 2),
                 rng.uniform([-1 + margin, 0 + margin], [0 - margin, 1 - margin], 2),
                 rng.uniform([0 + margin, 0 + margin], [1 - margin, 1 - margin], 2)]

        cov = np.array([[[.01, 0], [0, .01]]] * nr_of_clusters)

        cluster_data = {}
        for i in clst_ids:
            cluster_data[i] = rng.multivariate_normal(mean=mean[i], cov=cov[i], size=nr_of_points_x_cluster)

        returned_data["clst_ids"] = clst_ids
        returned_data["cluster_data"] = cluster_data
    elif dataset == "mnist_singled":
        ds_train, ds_test = get_mnist_op_dataset(
            count_train=3000, count_test=1000, buffer_size=3000, batch_size=16, n_operands=2, op=lambda args: args[0] + args[1])

        returned_data["ds_train"] = ds_train
        returned_data["ds_test"] = ds_test
    elif dataset == "mnist_multid":
        ds_train, ds_test = get_mnist_op_dataset(
            count_train=3000, count_test=1000, buffer_size=3000, batch_size=16, n_operands=4, op=lambda args: 10 * args[0] + args[1] + 10 * args[2] + args[3])

        returned_data["ds_train"] = ds_train
        returned_data["ds_test"] = ds_test
    elif dataset == "multiclass":

        df_train = pd.read_csv("./data/datasets/iris/iris_training.csv")
        df_test = pd.read_csv("./data/datasets/iris/iris_test.csv")

        labels_train = df_train.pop("species")
        labels_test = df_test.pop("species")

        batch_size = info_data["batch_size"]
        ds_train = tf.data.Dataset.from_tensor_slices((df_train, labels_train)).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((df_test, labels_test)).batch(batch_size)

        returned_data["ds_train"] = ds_train
        returned_data["ds_test"] = ds_test
    elif dataset == "multilabel":
        df = pd.read_csv("./data/datasets/crabs.dat", sep=" ", skipinitialspace=True)
        df = df.sample(frac=1, random_state=info_data["data_seed"])  # shuffle

        features = df[['FL', 'RW', 'CL', 'CW', 'BD']]
        labels_sex = df['sex']
        labels_color = df['sp']

        # 160 samples for training and 40 samples for testing.

        batch_size = info_data["batch_size"]
        ds_train = tf.data.Dataset.from_tensor_slices((features[:160], labels_sex[:160], labels_color[:160])).batch(
            batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((features[160:], labels_sex[160:], labels_color[160:])).batch(
            batch_size)

        returned_data["ds_train"] = ds_train
        returned_data["ds_test"] = ds_test
    elif dataset == "parent_ancestor":
        entities = ["sue", "diana", "john", "edna", "paul", "francis", "john2",
                    "john3", "john4", "joe", "jennifer", "juliet", "janice",
                    "joey", "tom", "bonnie", "katie"]

        parents = [
            ("sue", "diana"),
            ("john", "diana"),
            ("sue", "bonnie"),
            ("john", "bonnie"),
            ("sue", "tom"),
            ("john", "tom"),
            ("diana", "katie"),
            ("paul", "katie"),
            ("edna", "sue"),
            ("john2", "sue"),
            ("edna", "john3"),
            ("john2", "john3"),
            ("francis", "john"),
            ("john4", "john"),
            ("francis", "janice"),
            ("john4", "janice"),
            ("janice", "jennifer"),
            ("joe", "jennifer"),
            ("janice", "juliet"),
            ("joe", "juliet"),
            ("janice", "joey"),
            ("joe", "joey")]

        all_relationships = list(itertools.product(entities, repeat=2))
        not_parents = [item for item in all_relationships if item not in parents]

        returned_data = {"entities": entities, "parents": parents, "not_parents": not_parents, "all_relationship": all_relationships}

    elif dataset == "regression":
        batch_size = info_data["batch_size"]

        path_dataset = "./data/datasets/real-estate.csv"
        df = pd.read_csv(path_dataset)
        df = df.sample(frac=1, random_state=info_data["data_seed"])  # shuffle

        x = df[['X1 transaction date', 'X2 house age',
                'X3 distance to the nearest MRT station',
                'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
        y = df[['Y house price of unit area']]
        ds_train = tf.data.Dataset.from_tensor_slices((x[:330], y[:330])).batch(batch_size)
        ds_test = tf.data.Dataset.from_tensor_slices((x[330:], y[330:])).batch(batch_size)

        # Convert datasets to float32
        ds_train = ds_train.map(_convert_to_float32)
        ds_test = ds_test.map(_convert_to_float32)

        returned_data["ds_train"] = ds_train
        returned_data["ds_test"] = ds_test

    elif dataset == "smokes_friends_cancer":

        friends = [('a', 'b'), ('a', 'e'), ('a', 'f'), ('a', 'g'), ('b', 'c'), ('c', 'd'), ('e', 'f'), ('g', 'h'),
                   ('i', 'j'), ('j', 'm'), ('k', 'l'), ('m', 'n')]
        smokes = ['a', 'e', 'f', 'g', 'j', 'n']
        cancer = ['a', 'e']

        returned_data["friends"] = friends
        returned_data["smokes"] = smokes
        returned_data["cancer"] = cancer


    return returned_data