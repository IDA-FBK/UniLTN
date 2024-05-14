import argparse
import random

import ltn
import numpy as np
import os
import tensorflow as tf

import matplotlib.pyplot as plt
from data.data import get_data
from experiments.configurations.configurations import get_configuration
from experiments.operators import get_operators
from experiments.plots import (
    plot_clusters,
    plot_clustering_results,
    plot_parents,
    plot_ancestor,
)
from models.axioms import (
    axioms_binaryc,
    axioms_clustering,
    axioms_mnist_singled,
    axioms_mnist_multid,
    axioms_multiclass,
    axioms_multilabel,
    sat_phi1,
    sat_phi2,
    sat_phi3,
    axioms_parent_ancestor,
    axioms_propositional_variables,
    axioms_regression,
    axioms_smokes_friends_cancer,
)
from models.commons import train, train_model_cpps
from models.evaluate import (
    evalute_additional_queries_parent_ancestors,
    evaluate_smokes_friend_cancer,
)
from models.models import get_model
from models.utils import get_optimizer
from models.steps import (
    train_step_binaryc,
    test_step_binaryc,
    train_step_mnist_singled,
    test_step_mnist_singled,
    train_step_mnist_multid,
    test_step_mnist_multid,
    train_step_multiclass,
    test_step_multiclass,
    train_step_multilabel,
    test_step_multilabel,
    train_step_regression,
    test_step_regression,
)
from utils import create_dir, get_scheduled_parameters_mnist_addition


def _set_seed(seed):
    # set seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    # When running on the CuDNN backend, two further options must be set
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-example",
        type=str,
        default="smokes_friends_cancer",
        help="specify the dataset to use",
    )
    parser.add_argument(
        "-path_to_conf",
        type=str,
        default="./experiments/configurations/smokes_friends_cancer/ltn/conf-00.json",
        help="configuration file for the current experiment",
    )
    parser.add_argument(
        "-path_to_results",
        type=str,
        default="./experiments/results/smokes_friends_cancer/uniltn/",
        help="directory where to store the results",
    )

    args = parser.parse_args()
    example = args.example
    path_to_conf = args.path_to_conf
    path_to_results = args.path_to_results
    conf = get_configuration(path_to_conf)

    exp_seed = conf["exp_seed"]
    epochs = conf["epochs"]
    lr = conf["learning_rate"]
    track_metrics = conf["track_metrics"]

    operators = get_operators(conf["operators"])
    if "not" in operators:
        Not = operators["not"]
    if "and" in operators:
        And = operators["and"]
    if "or" in operators:
        Or = operators["or"]
    if "implies" in operators:
        Implies = operators["implies"]
    if "forall" in operators:
        Forall = operators["forall"]
    if "exists" in operators:
        Exists = operators["exists"]
    if "equiv" in operators:
        Equiv = operators["equiv"]
    if "fagg" in operators:
        formula_aggregator = operators["fagg"]

    _set_seed(exp_seed)
    path_to_results += f"seed-{exp_seed}_epochs{epochs}_lr{lr}/"
    create_dir(path_to_results)

    if example == "binaryc":
        current_data = get_data(example, conf)
        ds_train, ds_test = current_data["ds_train"], current_data["ds_test"]

        models = get_model(example)

        A = models["A"]

        # Define the metrics
        metrics_dict = {
            "train_sat": tf.keras.metrics.Mean(name="train_sat"),
            "test_sat": tf.keras.metrics.Mean(name="test_sat"),
            "train_accuracy": tf.keras.metrics.BinaryAccuracy(
                name="train_accuracy", threshold=0.5
            ),
            "test_accuracy": tf.keras.metrics.BinaryAccuracy(
                name="test_accuracy", threshold=0.5
            ),
        }

        axioms_info = {"A": A, "not": Not, "forall": Forall, "fagg": formula_aggregator}

        # Initialize all layers and the static graph.
        for _data, _labels in ds_test:
            print(
                "Initial sat level %.5f" % axioms_binaryc(_data, _labels, axioms_info)
            )
            break

        optimizer = get_optimizer(conf)
        step_info = {
            "axioms": axioms_binaryc,
            "axioms_info": axioms_info,
            "optimizer": optimizer,
            "metrics_dict": metrics_dict,
        }

        train(
            epochs,
            metrics_dict,
            ds_train,
            ds_test,
            train_step_binaryc,
            test_step_binaryc,
            step_info,
            csv_path=path_to_results,
            track_metrics=track_metrics,
        )
    elif example == "clustering":
        plt.rcParams["font.size"] = 12
        plt.rcParams["axes.linewidth"] = 1

        current_data = get_data(example, conf)
        clst_ids, cluster_data = current_data["clst_ids"], current_data["cluster_data"]

        nr_of_clusters = len(clst_ids)
        plot_clusters(clst_ids, cluster_data, path_to_results)
        data = np.concatenate([cluster_data[i] for i in clst_ids])

        close_threshold = 0.2
        distant_threshold = 1.0

        models = get_model(example)
        c_model = models["clustering"]
        C = ltn.Predicate(c_model(nr_of_clusters, single_label=True))
        cluster = ltn.Variable("cluster", clst_ids)

        x = ltn.Variable("x", data)
        y = ltn.Variable("y", data)

        eucl_dist = ltn.Function.Lambda(
            lambda inputs: tf.expand_dims(
                tf.norm(inputs[0] - inputs[1], axis=1), axis=1
            )
        )
        is_greater_than = ltn.Predicate.Lambda(lambda inputs: inputs[0] > inputs[1])

        close_thr = ltn.Constant(close_threshold, trainable=False)
        distant_thr = ltn.Constant(distant_threshold, trainable=False)

        axioms_info = {
            "C": C,
            "not": Not,
            "and": And,
            "forall": Forall,
            "exists": Exists,
            "equiv": Equiv,
            "fagg": formula_aggregator,
            "x": x,
            "y": y,
            "cluster": cluster,
            "close_thr": close_thr,
            "distant_thr": distant_thr,
            "eucl_dist": eucl_dist,
            "is_greater_than": is_greater_than,
        }

        p_exists = 6
        # first call to build the graph
        axioms_clustering(axioms_info, p_exists)

        optimizer = get_optimizer(conf)
        train_model_cpps(
            example,
            epochs,
            axioms_clustering,
            axioms_info,
            C.trainable_variables,
            optimizer,
            path_to_results,
            track_metrics,
        )
        plot_clustering_results(
            data, nr_of_clusters, clst_ids, cluster_data, C, path_to_results
        )

    elif example == "mnist_singled":
        current_data = get_data(example, conf)
        ds_train, ds_test = current_data["ds_train"], current_data["ds_test"]

        models = get_model(example)
        logits_model = models["logits_model"]
        Digit = models["Digit"]

        # mask
        add = ltn.Function.Lambda(lambda inputs: inputs[0] + inputs[1])
        equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])

        d1 = ltn.Variable("digits1", range(10))
        d2 = ltn.Variable("digits2", range(10))

        axioms_info = {
            "Digit": Digit,
            "d1": d1,
            "d2": d2,
            "and": And,
            "forall": Forall,
            "exists": Exists,
            "add": add,
            "equals": equals,
        }

        images_x, images_y, labels_z = next(ds_train.as_numpy_iterator())
        axioms_mnist_singled(
            images_x, images_y, labels_z, axioms_info, tf.constant(2.0)
        )

        optimizer = get_optimizer(conf)

        metrics_dict = {
            "train_loss": tf.keras.metrics.Mean(name="train_loss"),
            "train_accuracy": tf.keras.metrics.Mean(name="train_accuracy"),
            "test_loss": tf.keras.metrics.Mean(name="test_loss"),
            "test_accuracy": tf.keras.metrics.Mean(name="test_accuracy"),
        }

        step_info = {
            "axioms": axioms_mnist_singled,
            "axioms_info": axioms_info,
            "logits_model": logits_model,
            "optimizer": optimizer,
            "metrics_dict": metrics_dict,
        }

        scheduled_parameters = get_scheduled_parameters_mnist_addition(epochs)
        train(
            epochs,
            metrics_dict,
            ds_train,
            ds_test,
            train_step_mnist_singled,
            test_step_mnist_singled,
            step_info,
            csv_path=path_to_results,
            track_metrics=track_metrics,
            scheduled_parameters=scheduled_parameters,
        )

    elif example == "mnist_multid":
        current_data = get_data(example, conf)
        ds_train, ds_test = current_data["ds_train"], current_data["ds_test"]

        models = get_model(example)
        logits_model = models["logits_model"]
        Digit = models["Digit"]

        ### Variables
        d1 = ltn.Variable("digits1", range(10))
        d2 = ltn.Variable("digits2", range(10))
        d3 = ltn.Variable("digits3", range(10))
        d4 = ltn.Variable("digits4", range(10))

        # mask
        add = ltn.Function.Lambda(lambda inputs: inputs[0] + inputs[1])
        times = ltn.Function.Lambda(lambda inputs: inputs[0] * inputs[1])
        ten = ltn.Constant(10, trainable=False)
        equals = ltn.Predicate.Lambda(lambda inputs: inputs[0] == inputs[1])
        two_digit_number = lambda inputs: add([times([ten, inputs[0]]), inputs[1]])

        axioms_info = {
            "Digit": Digit,
            "d1": d1,
            "d2": d2,
            "d3": d3,
            "d4": d4,
            "and": And,
            "forall": Forall,
            "exists": Exists,
            "add": add,
            "equals": equals,
            "two_digit_number": two_digit_number,
        }

        x1, x2, y1, y2, z = next(ds_train.as_numpy_iterator())
        axioms_mnist_multid(x1, x2, y1, y2, z, axioms_info, tf.constant(2.0))

        optimizer = get_optimizer(conf)

        metrics_dict = {
            "train_loss": tf.keras.metrics.Mean(name="train_loss"),
            "train_accuracy": tf.keras.metrics.Mean(name="train_accuracy"),
            "test_loss": tf.keras.metrics.Mean(name="test_loss"),
            "test_accuracy": tf.keras.metrics.Mean(name="test_accuracy"),
        }

        step_info = {
            "axioms": axioms_mnist_multid,
            "axioms_info": axioms_info,
            "logits_model": logits_model,
            "optimizer": optimizer,
            "metrics_dict": metrics_dict,
        }

        scheduled_parameters = get_scheduled_parameters_mnist_addition(epochs)
        train(
            epochs,
            metrics_dict,
            ds_train,
            ds_test,
            train_step_mnist_multid,
            test_step_mnist_multid,
            step_info,
            csv_path=path_to_results,
            track_metrics=track_metrics,
            scheduled_parameters=scheduled_parameters,
        )
    elif example == "multiclass":
        current_data = get_data(example, conf)
        ds_train, ds_test = current_data["ds_train"], current_data["ds_test"]

        models = get_model(example)
        logits_model = models["logits_model"]
        p = models["p"]

        # Constants to index/iterate on the classes
        class_A = ltn.Constant(0, trainable=False)
        class_B = ltn.Constant(1, trainable=False)
        class_C = ltn.Constant(2, trainable=False)

        axioms_info = {
            "class_A": class_A,
            "class_B": class_B,
            "class_C": class_C,
            "p": p,
            "forall": Forall,
            "fagg": formula_aggregator,
            "training": False,
        }
        # Initialize all layers and the static graph
        for features, labels in ds_test:
            print(
                "Initial sat level %.5f"
                % axioms_multiclass(features, labels, axioms_info)
            )
            break

        metrics_dict = {
            "train_sat_kb": tf.keras.metrics.Mean(name="train_sat_kb"),
            "test_sat_kb": tf.keras.metrics.Mean(name="test_sat_kb"),
            "train_accuracy": tf.keras.metrics.CategoricalAccuracy(
                name="train_accuracy"
            ),
            "test_accuracy": tf.keras.metrics.CategoricalAccuracy(name="test_accuracy"),
        }
        optimizer = get_optimizer(conf)
        step_info = {
            "axioms": axioms_multiclass,
            "axioms_info": axioms_info,
            "logits_model": logits_model,
            "optimizer": optimizer,
            "metrics_dict": metrics_dict,
        }
        train(
            epochs,
            metrics_dict,
            ds_train,
            ds_test,
            train_step_multiclass,
            test_step_multiclass,
            step_info,
            csv_path=path_to_results,
            track_metrics=track_metrics,
        )

    elif example == "multilabel":
        current_data = get_data(example, conf)
        ds_train, ds_test = current_data["ds_train"], current_data["ds_test"]

        models = get_model(example)
        logits_model = models["logits_model"]
        p = models["p"]

        class_male = ltn.Constant(0, trainable=False)
        class_female = ltn.Constant(1, trainable=False)
        class_blue = ltn.Constant(2, trainable=False)
        class_orange = ltn.Constant(3, trainable=False)

        axioms_info = {
            "class_male": class_male,
            "class_female": class_female,
            "class_blue": class_blue,
            "class_orange": class_orange,
            "p": p,
            "and": And,
            "not": Not,
            "implies": Implies,
            "forall": Forall,
            "fagg": formula_aggregator,
        }

        for features, labels_sex, labels_color in ds_train:
            print(
                "Initial sat level %.5f"
                % axioms_multilabel(features, labels_sex, labels_color, axioms_info)
            )
            break

        metrics_dict = {
            "train_sat_kb": tf.keras.metrics.Mean(name="train_sat_kb"),
            "test_sat_kb": tf.keras.metrics.Mean(name="test_sat_kb"),
            "train_accuracy": tf.keras.metrics.Mean(name="train_accuracy"),
            "test_accuracy": tf.keras.metrics.Mean(name="test_accuracy"),
            "test_sat_phi1": tf.keras.metrics.Mean(name="test_sat_phi1"),
            "test_sat_phi2": tf.keras.metrics.Mean(name="test_sat_phi2"),
            "test_sat_phi3": tf.keras.metrics.Mean(name="test_sat_phi3"),
        }

        optimizer = get_optimizer(conf)
        step_info = {
            "axioms": axioms_multilabel,
            "axioms_info": axioms_info,
            "logits_model": logits_model,
            "sat_phi1": sat_phi1,
            "sat_phi2": sat_phi2,
            "sat_phi3": sat_phi3,
            "optimizer": optimizer,
            "metrics_dict": metrics_dict,
        }
        train(
            epochs,
            metrics_dict,
            ds_train,
            ds_test,
            train_step_multilabel,
            test_step_multilabel,
            step_info,
            csv_path=path_to_results,
            track_metrics=track_metrics,
        )

    elif example == "parent_ancestor":
        data = get_data(example, conf)
        entities, parents, not_parents = (
            data["entities"],
            data["parents"],
            data["not_parents"],
        )
        plot_parents(parents, path_to_results)
        plot_ancestor(parents, entities, path_to_results)

        models = get_model(example)
        Parent = models["Parent"]
        Ancestor = models["Ancestor"]

        g_e = {
            l: ltn.Constant(
                np.random.uniform(low=0.0, high=1.0, size=4), trainable=True
            )
            for l in entities
        }

        axioms_info = {
            "g_e": g_e,
            "parent_predicate": Parent,
            "ancestor_predicate": Ancestor,
            "parents": parents,
            "not_parents": not_parents,
            "not": Not,
            "and": And,
            "or": Or,
            "forall": Forall,
            "exists": Exists,
            "implies": Implies,
            "fagg": formula_aggregator,
        }

        print("Initial sat level %.5f" % axioms_parent_ancestor(axioms_info))

        trainable_variables = (
            Parent.trainable_variables
            + Ancestor.trainable_variables
            + ltn.as_tensors(list(g_e.values()))
        )

        optimizer = get_optimizer(conf)
        # simple_train(epochs, axioms_parent_ancestor, axioms_info, trainable_variables, optimizer,
        #               path_to_results, track_metrics)
        train_model_cpps(
            example,
            epochs,
            axioms_parent_ancestor,
            axioms_info,
            trainable_variables,
            optimizer,
            path_to_results,
            track_metrics,
        )
        evalute_additional_queries_parent_ancestors(axioms_info, path_to_results)

    elif example == "propositional_variables":
        a = ltn.Proposition(0.2, trainable=True)
        b = ltn.Proposition(0.5, trainable=True)
        c = ltn.Proposition(0.5, trainable=True)
        d = ltn.Proposition(0.3, trainable=False)
        e = ltn.Proposition(0.9, trainable=False)

        x = ltn.Variable("x", np.array([[1, 2], [3, 4], [5, 6]]))
        models = get_model(example)
        P = models["P"]

        axioms_info = {
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "e": e,
            "P": P,
            "x": x,
            "not": Not,
            "and": And,
            "forall": Forall,
            "exists": Exists,
            "implies": Implies,
            "fagg": formula_aggregator,
        }

        print("Initial sat level %.5f" % axioms_propositional_variables(axioms_info))

        trainable_variables = ltn.as_tensors([a, b, c])
        optimizer = get_optimizer(conf)

        train_model_cpps(
            example,
            epochs,
            axioms_propositional_variables,
            axioms_info,
            trainable_variables,
            optimizer,
            path_to_results,
            track_metrics,
        )
        # simple_train(epochs, axioms_propositional_variables, axioms_info, trainable_variables, optimizer, path_to_results, 100)
    elif example == "regression":
        # TODO: problem with accuracy calculation & still ongoing
        data = get_data(example, conf)
        ds_train, ds_test = data["ds_train"], data["ds_test"]

        models = get_model(example)
        f = models["f"]

        # Equality Predicate
        eq = ltn.Predicate.Lambda(
            # lambda args: tf.exp(-0.05*tf.sqrt(tf.reduce_sum(tf.square(args[0]-args[1]),axis=1)))
            lambda args: 1
            / (1 + 0.5 * tf.sqrt(tf.reduce_sum(tf.square(args[0] - args[1]), axis=1)))
        )

        # # Training
        #
        # Define the metrics. While training, we measure:
        # 1. The level of satisfiability of the Knowledge Base of the training data.
        # 1. The level of satisfiability of the Knowledge Base of the test data.
        # 3. The training accuracy.
        # 4. The test accuracy.
        metrics_dict = {
            "train_sat": tf.keras.metrics.Mean(name="train_sat"),
            "test_sat": tf.keras.metrics.Mean(name="test_sat"),
            "train_accuracy": tf.keras.metrics.RootMeanSquaredError(
                name="train_accuracy"
            ),
            "test_accuracy": tf.keras.metrics.RootMeanSquaredError(
                name="test_accuracy"
            ),
        }

        track_metrics = conf["track_metrics"]

        axioms_info = {"f": f, "forall": Forall, "eq": eq}

        # Initialize all layers and the static graph
        for x, y in ds_test:
            print("Initial sat level %.5f" % axioms_regression(x, y, axioms_info))
            break

        optimizer = get_optimizer(conf)

        step_info = {
            "axioms": axioms_regression,
            "axioms_info": axioms_info,
            "optimizer": optimizer,
            "metrics_dict": metrics_dict,
        }
        train(
            epochs,
            metrics_dict,
            ds_train,
            ds_test,
            train_step_regression,
            test_step_regression,
            step_info,
            csv_path=path_to_results,
            track_metrics=track_metrics,
        )
    elif example == "smokes_friends_cancer":
        current_data = get_data(example)

        friends = current_data["friends"]
        smokes = current_data["smokes"]
        cancer = current_data["cancer"]

        embedding_size = [5]

        g1 = {
            l: ltn.Constant(
                np.random.uniform(low=0.0, high=1.0, size=embedding_size),
                trainable=True,
            )
            for l in "abcdefgh"
        }
        g2 = {
            l: ltn.Constant(
                np.random.uniform(low=0.0, high=1.0, size=embedding_size),
                trainable=True,
            )
            for l in "ijklmn"
        }
        g = {**g1, **g2}

        models = get_model(example)
        Smokes = models["Smokes"]
        Friends = models["Friends"]
        Cancer = models["Cancer"]

        trainable_variables = (
            Smokes.trainable_variables
            + Friends.trainable_variables
            + Cancer.trainable_variables
            + ltn.as_tensors(list(g.values()))
        )

        optimizer = get_optimizer(conf)

        # evaluations
        axioms_info = {
            "g": g,
            "g1": g1,
            "g2": g2,
            "Friends": Friends,
            "Smokes": Smokes,
            "Cancer": Cancer,
            "smokes": smokes,
            "friends": friends,
            "cancer": cancer,
            "not": Not,
            "and": And,
            "or": Or,
            "implies": Implies,
            "forall": Forall,
            "exists": Exists,
            "fagg": formula_aggregator,
        }

        train_model_cpps(
            example,
            epochs,
            axioms_smokes_friends_cancer,
            axioms_info,
            trainable_variables,
            optimizer,
            path_to_results,
        )

        evaluate_smokes_friend_cancer(axioms_info, path_to_results)
