from collections import defaultdict

import tensorflow as tf

def train(
        epochs,
        metrics_dict,
        ds_train,
        ds_test,
        train_step,
        test_step,
        step_info,
        track_metrics=1,
        csv_path=None,
        scheduled_parameters=defaultdict(lambda: {})
):
    """
    Train a model - used for binary classification.

    Parameters:
    - epochs (int): Number of training epochs.
    - metrics_dict (dict): A dictionary containing metrics to track during training.
    - ds_train (iterable): Iterable dataset for training.
    - ds_test (iterable): Iterable dataset for testing.
    - train_step (callable): Function representing a single training step.
    - test_step (callable): Function representing a single testing step.
    - step_info (any): Additional information required for training/testing steps.
    - track_metrics (int): Frequency of printing/tracking metrics during training.
    - csv_path (str, optional): Path to save the metrics in a CSV file.
    - scheduled_parameters (dict, optional): A dictionary that returns keyword arguments for train_step

    Returns:
    - None
    """
    template = "Epoch {}"

    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label
    if csv_path is not None:
        csv_file = open(csv_path+"results.csv", "w+")
        headers = ",".join(["Epoch"] + list(metrics_dict.keys()))
        csv_template = ",".join(["{}" for _ in range(len(metrics_dict) + 1)])
        csv_file.write(headers + "\n")

    for epoch in range(epochs):
        for metrics in metrics_dict.values():
            metrics.reset_state()
        for batch_elements in ds_train:
            train_step(*batch_elements, step_info, **scheduled_parameters[epoch])
        for batch_elements in ds_test:
            test_step(*batch_elements, step_info, **scheduled_parameters[epoch])

        metrics_results = [metrics.result() for metrics in metrics_dict.values()]
        if epoch % track_metrics == 0:
            print(template.format(epoch, *metrics_results))
        if csv_path is not None:
            # Format metrics_results with desired number of decimals
            formatted_metrics = [f'{metric:.4f}' for metric in metrics_results]
            csv_file.write(csv_template.format(epoch, *formatted_metrics) + "\n")
            csv_file.flush()
    if csv_path is not None:
        csv_file.close()


def train_model_cpps(example, epochs, axioms, axioms_info, trainable_variables, optimizer, log_path, log_epochs=200):
    """
    Train a model based on axioms - used for clustering,  parent_ancestor, propositional variables, smokes_friend_cancer examples.

    Parameters:
    - example (str): name of the example (i.e, task).
    - epochs (int): Number of training epochs.
    - axioms (callable): A function representing the axioms to be satisfied.
    - axioms_info (dict): Information required for evaluating axioms.
    - trainable_variables (list): List of trainable variables.
    - optimizer (tf.keras.optimizers.Optimizer): The optimizer used for gradient descent optimization.
    - log_path (str): Path to save the training logs.
    - log_epochs (int): Frequency of logging training progress, in terms of epochs.

    Returns:
    - None
    """
    log_file = log_path + "log.txt"
    with open(log_file, 'w') as f:
        for epoch in range(epochs):
            p_exists = None
            if example == "clustering":
                p_exists = 1 if epoch <= 100 else 6
            elif example == "smokes_friends_cancer":
                p_exists = tf.constant(1.) if 0 <= epoch < 400 else tf.constant(6.)

            with tf.GradientTape() as tape:
                if p_exists is None:
                    loss_value = 1. - axioms(axioms_info)
                else:
                    loss_value = 1. - axioms(axioms_info, p_exists=p_exists)

            grads = tape.gradient(loss_value, trainable_variables)
            optimizer.apply_gradients(zip(grads, trainable_variables))

            if epoch % log_epochs == 0:
                if p_exists is None:
                    log_message = "Epoch %d: Sat Level %.3f\n" % (epoch, axioms(axioms_info))
                else:
                    log_message = "Epoch %d: Sat Level %.3f\n" % (epoch, axioms(axioms_info, p_exists=p_exists))
                print(log_message)
                f.write(log_message)

        if p_exists is None:
            final_log_message = "Training finished at Epoch %d with Sat Level %.3f\n" % (epoch, axioms(axioms_info))
        else:
            final_log_message = "Training finished at Epoch %d with Sat Level %.3f\n" % (epoch, axioms(axioms_info, p_exists=p_exists))

        print(final_log_message)
        f.write(final_log_message)

