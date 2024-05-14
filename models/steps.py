import tensorflow as tf


@tf.function
def train_step_binaryc(data, labels, train_info):
    """
    Perform a single training step for a binary classification model.

    Parameters:
    - data (tf.Tensor): Input data.
    - labels (tf.Tensor): Target labels.
    - train_info (dict): Information required for training.

    Returns:
    - None
    """
    axioms = train_info["axioms"]
    axioms_info = train_info["axioms_info"]
    A = axioms_info["A"]
    optimizer = train_info["optimizer"]
    metrics_dict = train_info["metrics_dict"]

    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(data, labels, axioms_info)
        loss = 1.-sat
    gradients = tape.gradient(loss, A.trainable_variables)
    optimizer.apply_gradients(zip(gradients, A.trainable_variables))

    metrics_dict['train_sat'](sat)
    # accuracy
    predictions = A.model(data)
    metrics_dict['train_accuracy'](labels, predictions)


@tf.function
def test_step_binaryc(data, labels, test_info):
    """
    Perform a single testing step for a binary classification model.

    Parameters:
    - data (tf.Tensor): Input data.
    - labels (tf.Tensor): Target labels.
    - test_info (dict): Information required for testing.

    Returns:
    - None
    """
    axioms = test_info["axioms"]
    axioms_info = test_info["axioms_info"]
    A = axioms_info["A"]
    metrics_dict = test_info["metrics_dict"]

    # sat and update
    sat = axioms(data, labels, axioms_info)
    metrics_dict['test_sat'](sat)
    # accuracy
    predictions = A.model(data)
    metrics_dict['test_accuracy'](labels, predictions)


@tf.function
def train_step_mnist_singled(images_x, images_y, labels_z, train_info, **parameters):
    """
    Perform a single step of training for mnist single digit addition.

    Parameters:
    - images_x (tf.Tensor): Input images for the first operand.
    - images_y (tf.Tensor): Input images for the second operand.
    - labels_z (tf.Tensor): Target labels.
    - test_info (dict): Dictionary containing testing information such as axioms and model.
    - **parameters: Additional keyword arguments.

    Returns:
    - None
    """
    axioms = train_info["axioms"]
    axioms_info = train_info["axioms_info"]
    logits_model = train_info["logits_model"]
    optimizer = train_info["optimizer"]
    metrics_dict = train_info["metrics_dict"]

    # loss
    with tf.GradientTape() as tape:
        loss = 1. - axioms(images_x, images_y, labels_z, axioms_info, **parameters)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]),axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]),axis=-1)
    predictions_z = predictions_x + predictions_y
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


@tf.function
def test_step_mnist_singled(images_x, images_y, labels_z, test_info, **parameters):
    """
    Perform a single step of testing for mnist single digit addition.

    Parameters:
    - images_x (tf.Tensor): Input images for the first operand.
    - images_y (tf.Tensor): Input images for the second operand.
    - labels_z (tf.Tensor): Target labels.
    - test_info (dict): Dictionary containing testing information such as axioms and model.
    - **parameters: Additional keyword arguments.

    Returns:
    - None
    """
    axioms = test_info["axioms"]
    axioms_info = test_info["axioms_info"]
    logits_model = test_info["logits_model"]
    metrics_dict = test_info["metrics_dict"]
    # loss
    loss = 1. - axioms(images_x, images_y, labels_z, axioms_info, **parameters)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x = tf.argmax(logits_model([images_x]),axis=-1)
    predictions_y = tf.argmax(logits_model([images_y]),axis=-1)
    predictions_z = predictions_x + predictions_y
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


@tf.function
def train_step_mnist_multid(images_x1, images_x2, images_y1, images_y2, labels_z, train_info, **kwargs):
    """
    Perform a single step of training for mnist multidigit addition.

    Parameters:
    - images_x1 (tf.Tensor): Input images for the first digit of the first operand.
    - images_x2 (tf.Tensor): Input images for the second digit of the first operand.
    - images_y1 (tf.Tensor): Input images for the first digit of the second operand.
    - images_y2 (tf.Tensor): Input images for the second digit of the second operand.
    - labels_z (tf.Tensor): Target labels.
    - test_info (dict): Dictionary containing testing information such as axioms and model.
    - **parameters: Additional keyword arguments.

    Returns:
    - None
    """
    axioms = train_info["axioms"]
    axioms_info = train_info["axioms_info"]
    logits_model = train_info["logits_model"]
    optimizer = train_info["optimizer"]
    metrics_dict = train_info["metrics_dict"]

    # loss
    with tf.GradientTape() as tape:
        loss = 1. - axioms(images_x1, images_x2, images_y1, images_y2, labels_z,  axioms_info, **kwargs)
    gradients = tape.gradient(loss, logits_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, logits_model.trainable_variables))
    metrics_dict['train_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]), axis=-1, output_type=tf.int32)
    predictions_x2 = tf.argmax(logits_model([images_x2]), axis=-1, output_type=tf.int32)
    predictions_y1 = tf.argmax(logits_model([images_y1]), axis=-1, output_type=tf.int32)
    predictions_y2 = tf.argmax(logits_model([images_y2]), axis=-1, output_type=tf.int32)
    predictions_z = 10 * predictions_x1 + predictions_x2 + 10 * predictions_y1 + predictions_y2
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['train_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


@tf.function
def test_step_mnist_multid(images_x1, images_x2, images_y1, images_y2, labels_z, test_info, **kwargs):
    """
    Perform a single step of testing for mnist multidigit addition.

    Parameters:
    - images_x1 (tf.Tensor): Input images for the first digit of the first operand.
    - images_x2 (tf.Tensor): Input images for the second digit of the first operand.
    - images_y1 (tf.Tensor): Input images for the first digit of the second operand.
    - images_y2 (tf.Tensor): Input images for the second digit of the second operand.
    - labels_z (tf.Tensor): Target labels.
    - test_info (dict): Dictionary containing testing information such as axioms and model.
    - **parameters: Additional keyword arguments.

    Returns:
    - None
    """
    axioms = test_info["axioms"]
    axioms_info = test_info["axioms_info"]
    logits_model = test_info["logits_model"]
    metrics_dict = test_info["metrics_dict"]
    # loss
    loss = 1. - axioms(images_x1, images_x2, images_y1, images_y2, labels_z,  axioms_info, **kwargs)
    metrics_dict['test_loss'](loss)
    # accuracy
    predictions_x1 = tf.argmax(logits_model([images_x1]), axis=-1, output_type=tf.int32)
    predictions_x2 = tf.argmax(logits_model([images_x2]), axis=-1, output_type=tf.int32)
    predictions_y1 = tf.argmax(logits_model([images_y1]), axis=-1, output_type=tf.int32)
    predictions_y2 = tf.argmax(logits_model([images_y2]), axis=-1, output_type=tf.int32)
    predictions_z = 10*predictions_x1+predictions_x2+10*predictions_y1+predictions_y2
    match = tf.equal(predictions_z, tf.cast(labels_z, predictions_z.dtype))
    metrics_dict['test_accuracy'](tf.reduce_mean(tf.cast(match, tf.float32)))


@tf.function
def train_step_multiclass(features, labels, train_info):
    """
    Perform a single training step for multiclass classification.

    Parameters:
    - features (tf.Tensor): Input features.
    - labels (tf.Tensor): Target labels.
    - train_info (dict): Dictionary containing training information including axioms, axioms_info,
                         optimizer, metrics_dict, and logits_model.

    Returns:
    - None
    """
    axioms = train_info["axioms"]
    axioms_info = train_info["axioms_info"]
    p = axioms_info["p"]
    optimizer = train_info["optimizer"]
    metrics_dict = train_info["metrics_dict"]
    logits_model = train_info["logits_model"]

    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(features, labels, axioms_info)
        loss = 1.-sat
    gradients = tape.gradient(loss, p.trainable_variables)
    optimizer.apply_gradients(zip(gradients, p.trainable_variables))
    sat = axioms(features, labels,  axioms_info) # compute sat without dropout
    metrics_dict['train_sat_kb'](sat)
    # accuracy
    predictions = logits_model([features])
    metrics_dict['train_accuracy'](tf.one_hot(labels,3),predictions)


@tf.function
def test_step_multiclass(features, labels, test_info):
    """
    Perform a single test step for multiclass classification.

    Parameters:
    - features (tf.Tensor): Input features.
    - labels (tf.Tensor): Target labels.
    - train_info (dict): Dictionary containing training information including axioms, axioms_info,
                         ometrics_dict, and logits_model.

    Returns:
    - None
    """
    axioms = test_info["axioms"]
    axioms_info = test_info["axioms_info"]
    metrics_dict = test_info["metrics_dict"]
    logits_model = test_info["logits_model"]

    # sat
    sat = axioms(features, labels, axioms_info)
    metrics_dict['test_sat_kb'](sat)
    # accuracy
    predictions = logits_model([features])
    metrics_dict['test_accuracy'](tf.one_hot(labels,3),predictions)


def multilabel_hamming_loss(y_true, y_pred, threshold=0.5, from_logits=False):
    """
    Compute the Hamming loss for multilabel classification.

    Parameters:
    - y_true (tf.Tensor): Ground truth labels.
    - y_pred (tf.Tensor): Predicted labels.
    - threshold (float): Threshold for binary classification (default is 0.5).
    - from_logits (bool): Whether the predictions are logits or probabilities (default is False).

    Returns:
    - tf.Tensor: Hamming loss.
    """
    if from_logits:
        y_pred = tf.math.sigmoid(y_pred)
    y_pred = y_pred > threshold
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.int32)
    nonzero = tf.cast(tf.math.count_nonzero(y_true-y_pred,axis=-1), tf.float32)
    return nonzero/y_true.get_shape()[-1]


@tf.function
def train_step_multilabel(features, labels_sex, labels_color, train_info):
    """
    Perform a single training step for multilabel classification.

    Parameters:
    - features (tf.Tensor): Input features.
    - labels_sex (tf.Tensor): Target labels for sex.
    - labels_color (tf.Tensor): Target labels for color.
    - train_info (dict): Dictionary containing training information including axioms, axioms_info,
                         optimizer, metrics_dict, and logits_model.

    Returns:
    - None
    """
    axioms = train_info["axioms"]
    axioms_info = train_info["axioms_info"]
    p = axioms_info["p"]
    optimizer = train_info["optimizer"]
    metrics_dict = train_info["metrics_dict"]
    logits_model = train_info["logits_model"]

    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(features, labels_sex, labels_color, axioms_info)
        loss = 1.-sat
    gradients = tape.gradient(loss, p.trainable_variables)
    optimizer.apply_gradients(zip(gradients, p.trainable_variables))
    metrics_dict['train_sat_kb'](sat)
    # accuracy
    predictions = logits_model([features])
    labels_male = (labels_sex == "M")
    labels_female = (labels_sex == "F")
    labels_blue = (labels_color == "B")
    labels_orange = (labels_color == "O")
    onehot = tf.stack([labels_male, labels_female, labels_blue, labels_orange], axis=-1)
    metrics_dict['train_accuracy'](1-multilabel_hamming_loss(onehot, predictions, from_logits=True))


@tf.function
def test_step_multilabel(features, labels_sex, labels_color, test_info):
    """
    Perform a single test step for multilabel classification.

    Parameters:
    - features (tf.Tensor): Input features.
    - labels_sex (tf.Tensor): Target labels for sex.
    - labels_color (tf.Tensor): Target labels for color.
    - test_info (dict): Dictionary containing test information including axioms, axioms_info,
                        metrics_dict, logits_model, sat_phi1, sat_phi2, and sat_phi3.

    Returns:
    - None
    """

    axioms = test_info["axioms"]
    axioms_info = test_info["axioms_info"]
    metrics_dict = test_info["metrics_dict"]
    logits_model = test_info["logits_model"]
    sat_phi1 = test_info["sat_phi1"]
    sat_phi2 = test_info["sat_phi2"]
    sat_phi3 = test_info["sat_phi3"]
    # sat
    sat_kb = axioms(features, labels_sex, labels_color, axioms_info)

    metrics_dict['test_sat_kb'](sat_kb)
    metrics_dict['test_sat_phi1'](sat_phi1(features, axioms_info))
    metrics_dict['test_sat_phi2'](sat_phi2(features, axioms_info))
    metrics_dict['test_sat_phi3'](sat_phi3(features, axioms_info))
    # accuracy
    predictions = logits_model([features])
    labels_male = (labels_sex == "M")
    labels_female = (labels_sex == "F")
    labels_blue = (labels_color == "B")
    labels_orange = (labels_color == "O")
    onehot = tf.stack([labels_male, labels_female, labels_blue, labels_orange], axis=-1)
    metrics_dict['test_accuracy'](1-multilabel_hamming_loss(onehot, predictions, from_logits=True))


@tf.function
def train_step_regression(x, y, train_info):
    """
    Perform a single training step for a regression model.

    Parameters:
    - x (tf.Tensor): Input data features.
    - y (tf.Tensor): Target values.
    - train_info (dict): Information required for training.

    Returns:
    - None
    """
    axioms = train_info["axioms"]
    axioms_info = train_info["axioms_info"]
    optimizer = train_info["optimizer"]
    f = axioms_info["f"]
    metrics_dict = train_info["metrics_dict"]

    # sat and update
    with tf.GradientTape() as tape:
        sat = axioms(x, y, axioms_info)
        loss = 1.-sat

    gradients = tape.gradient(loss, f.trainable_variables)
    optimizer.apply_gradients(zip(gradients, f.trainable_variables))

    sat = axioms(x, y, axioms_info)
    metrics_dict['train_sat'](sat)
    # accuracy
    metrics_dict['train_accuracy'](y, f.model(x))


@tf.function
def test_step_regression(x, y, test_info):
    """
    Perform a single testing step for a regression model.

    Parameters:
    - x (tf.Tensor): Input data features.
    - y (tf.Tensor): Target values.
    - test_info (dict): Information required for testing.

    Returns:
    - None
    """
    axioms = test_info["axioms"]
    axioms_info = test_info["axioms_info"]
    f = axioms_info["f"]

    metrics_dict = test_info["metrics_dict"]

    # sat
    sat = axioms(x, y, axioms_info)
    metrics_dict['test_sat'](sat)
    # accuracy
    metrics_dict['test_accuracy'](y, f.model(x))
