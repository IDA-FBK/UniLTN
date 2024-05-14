import ltn
import tensorflow as tf
from tensorflow.keras import layers


class MLP_clustering(tf.keras.Model):
    """ Model to call as P(x,class) """

    def __init__(self, n_classes, single_label, hidden_layer_sizes=(16, 16, 16)):
        super(MLP_clustering, self).__init__()
        self.denses = [layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = layers.Dense(n_classes)
        self.to_probs = tf.nn.softmax if single_label else tf.math.sigmoid

    def call(self, inputs):
        x, c = inputs[0], inputs[1]
        for dense in self.denses:
            x = dense(x)
        logits = self.dense_class(x)
        probs = self.to_probs(logits)
        indices = tf.cast(c, tf.int32)
        return tf.gather(probs, indices, batch_dims=1)

##### MNIST

class MNISTConv(tf.keras.Model):
    """CNN that returns linear embeddings for MNIST images.
    """

    def __init__(self, hidden_conv_filters=(6, 16), kernel_sizes=(5, 5), hidden_dense_sizes=(100,)):
        super(MNISTConv, self).__init__()
        self.convs = [layers.Conv2D(f, k, activation="elu") for f, k in
                      zip(hidden_conv_filters, kernel_sizes)]
        self.maxpool = layers.MaxPool2D((2, 2))
        self.flatten = layers.Flatten()
        self.denses = [layers.Dense(s, activation="elu") for s in hidden_dense_sizes]

    def call(self, x):
        for conv in self.convs:
            x = conv(x)
            x = self.maxpool(x)
        x = self.flatten(x)
        for dense in self.denses:
            x = dense(x)
        return x


class SingleDigit(tf.keras.Model):
    """Model classifying one digit image into 10 possible classes.
    """

    def __init__(self, hidden_dense_sizes=(84,), inputs_as_a_list=False):
        super(SingleDigit, self).__init__()
        self.mnistconv = MNISTConv()
        self.denses = [layers.Dense(s, activation="elu") for s in hidden_dense_sizes]
        self.dense_class = layers.Dense(10)
        self.inputs_as_a_list = inputs_as_a_list

    def call(self, inputs):
        x = inputs if not self.inputs_as_a_list else inputs[0]
        x = self.mnistconv(x)
        for dense in self.denses:
            x = dense(x)
        x = self.dense_class(x)
        return x

##### MNIST

class MultiDigits(tf.keras.Model):
    """Model classifying several digit images into n possible classes.
    """

    def __init__(self, n_classes, hidden_dense_sizes=(84,)):
        super(MultiDigits, self).__init__()
        self.mnistconv = MNISTConv()
        self.denses = [layers.Dense(s, activation="elu") for s in hidden_dense_sizes]
        self.dense_class = layers.Dense(n_classes)

    def call(self, inputs):
        x = [self.mnistconv(x) for x in inputs]
        x = tf.concat(x, axis=-1)
        for dense in self.denses:
            x = dense(x)
        x = self.dense_class(x)
        return x
class MLPMulticlass(tf.keras.Model):
    """Model that returns logits."""

    def __init__(self, n_classes, hidden_layer_sizes=(16, 16, 8)):
        super(MLPMulticlass, self).__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, inputs, training=False):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
            x = self.dropout(x, training=training)
        return self.dense_class(x)


class MLPMultilabel(tf.keras.Model):
    """Model that returns logits."""

    def __init__(self, n_classes, hidden_layer_sizes=(16, 16, 8)):
        super(MLPMultilabel, self).__init__()
        self.denses = [tf.keras.layers.Dense(s, activation="elu") for s in hidden_layer_sizes]
        self.dense_class = tf.keras.layers.Dense(n_classes)

    def call(self, inputs):
        x = inputs[0]
        for dense in self.denses:
            x = dense(x)
        return self.dense_class(x)


def MLP_regression(input_shapes,output_shape,hidden_layer_sizes = (16,16)):
    inputs = [tf.keras.Input(shape) for shape in input_shapes]
    flat_inputs = [layers.Flatten()(x) for x in inputs]
    hidden = layers.Concatenate()(flat_inputs) if len(flat_inputs) > 1 else flat_inputs[0]
    for units in hidden_layer_sizes:
        hidden = layers.Dense(units,activation=tf.nn.elu)(hidden)
    flat_outputs = layers.Dense(output_shape)(hidden)
    outputs = layers.Reshape([output_shape])(flat_outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def get_model(example):
    """
    Build models based on different examples.

    Parameters:
    - example (str): A string specifying the example type for which to build the model.

    Returns:
    - dict: A dictionary containing the built models. Keys are model names, and values are the corresponding models.
    """

    models = {}
    if example == "binaryc":
        models["A"] = ltn.Predicate.MLP([[2]], hidden_layer_sizes=(16, 16))
    elif example == "clustering":
        models["clustering"] = MLP_clustering
    elif example == "multiclass":
        logits_model = MLPMulticlass(3)
        p = ltn.Predicate.FromLogits(logits_model, activation_function="softmax", with_class_indexing=True)
        models["logits_model"] = logits_model
        models["p"] = p
    elif example == "mnist_singled":
        logits_model = SingleDigit(inputs_as_a_list=True)
        Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")
        models["logits_model"] = logits_model
        models["Digit"] = Digit
    elif example == "mnist_multid":
        logits_model = SingleDigit(inputs_as_a_list=True)
        Digit = ltn.Predicate.FromLogits(logits_model, activation_function="softmax")
        models["logits_model"] = logits_model
        models["Digit"] = Digit
    elif example == "multilabel":
        logits_model = MLPMultilabel(4)
        p = ltn.Predicate.FromLogits(logits_model, activation_function="sigmoid", with_class_indexing=True)
        models["logits_model"] = logits_model
        models["p"] = p
    elif example == "parent_ancestor":
        embedding_size = [4]
        models["Ancestor"] = ltn.Predicate.MLP([embedding_size, embedding_size], hidden_layer_sizes=(8, 8))
        models["Parent"] = ltn.Predicate.MLP([embedding_size, embedding_size], hidden_layer_sizes=(8, 8))
    elif example == "propositional_variables":
        models["P"] = ltn.Predicate.MLP(input_shapes=[[2]])
    elif example == "regression":
        models["f"] = ltn.Function.MLP(input_shapes=[[6]], output_shape=[1], hidden_layer_sizes=(8, 8))
    elif example == "smokes_friends_cancer":
        embedding_size = [5]
        models["Smokes"] = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(8, 8))
        models["Friends"] = ltn.Predicate.MLP([embedding_size, embedding_size], hidden_layer_sizes=(8, 8))
        models["Cancer"] = ltn.Predicate.MLP([embedding_size], hidden_layer_sizes=(8, 8))

    return models

