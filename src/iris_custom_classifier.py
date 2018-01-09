import os

import six.moves.urllib.request as request
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

# Check that we have correct TensorFlow version installed
tf_version = tf.__version__
print("TensorFlow version: {}".format(tf_version))
assert "1.4" <= tf_version, "TensorFlow r1.4 or later is needed"



# Windows users: You only need to change PATH, rest is platform independent
PATH = "/tmp/tf_iris_knowm_1"

# Fetch and store Training and Test dataset files
PATH_DATASET = PATH + os.sep + "dataset"
FILE_TRAIN = PATH_DATASET + os.sep + "iris_training.csv"
FILE_TEST = PATH_DATASET + os.sep + "iris_test.csv"
URL_TRAIN = "http://download.tensorflow.org/data/iris_training.csv"
URL_TEST = "http://download.tensorflow.org/data/iris_test.csv"


def downloadDataset(url, file):
    if not os.path.exists(PATH_DATASET):
        os.makedirs(PATH_DATASET)
    if not os.path.exists(file):
        data = request.urlopen(url).read()
        with open(file, "wb") as f:
            f.write(data)
            f.close()
downloadDataset(URL_TRAIN, FILE_TRAIN)
downloadDataset(URL_TEST, FILE_TEST)

tf.logging.set_verbosity(tf.logging.INFO)

# The CSV features in our training & test data
feature_names = [
    'SepalLength',
    'SepalWidth',
    'PetalLength',
    'PetalWidth']
























# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def my_input_fn(file_path, repeat_count=1, shuffle_count=1):
    def decode_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        label = parsed_line[-1]  # Last element is the label
        del parsed_line[-1]  # Delete last element
        features = parsed_line  # Everything but last elements are the features
        d = dict(zip(feature_names, features)), label
        return d

    dataset = (tf.data.TextLineDataset(file_path)  # Read text file
        .skip(1)  # Skip header row
        .map(decode_csv, num_parallel_calls=4)  # Decode each line
        .cache() # Warning: Caches entire dataset, can cause out of memory
        .shuffle(shuffle_count)  # Randomize elems (1 == no operation)
        .repeat(repeat_count)    # Repeats dataset this # times
        .batch(32)
        .prefetch(1)  # Make sure you always have 1 batch ready to serve
    )
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels



def my_model_fn(
    features, # This is batch_features from input_fn
    labels,   # This is batch_labels from input_fn
    mode):    # And instance of tf.estimator.ModeKeys, see below

    # All our inputs are feature columns of type numeric_column
    feature_columns = [
        tf.feature_column.numeric_column(feature_names[0]),
        tf.feature_column.numeric_column(feature_names[1]),
        tf.feature_column.numeric_column(feature_names[2]),
        tf.feature_column.numeric_column(feature_names[3])
    ]










    # Create the layer of input
    input_layer = tf.feature_column.input_layer(features, feature_columns)

    # Definition of hidden layer: h1
    # We implement it as a fully-connected layer (tf.layers.dense)
    # Has 10 neurons, and uses ReLU as the activation function
    # Takes input_layer as input
    h1 = tf.layers.Dense(10, activation=tf.nn.relu)(input_layer)

    # Definition of hidden layer: h2 (this is the logits layer)
    # Similar to h1, but takes h1 as input
    h2 = tf.layers.Dense(10, activation=tf.nn.relu)(h1)

    # Output 'logits' layer is three number = probability distribution
    # between Iris Setosa, Versicolor, and Viginica
    logits = tf.layers.Dense(3)(h2)

    # class_ids will be the model prediction for the class (Iris flower type)
    # The output node with the highest value is our prediction
    predictions = { 'class_ids': tf.argmax(input=logits, axis=1) }

    # 1. Prediction mode
    # Return our prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)



    # Evaluation and Training mode

    # Calculate the loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Calculate the accuracy between the true labels, and our predictions
    accuracy = tf.metrics.accuracy(labels, predictions['class_ids'])

    # 2. Evaluation mode
    # Return our loss (which is used to evaluate our model)
    # Set the TensorBoard scalar my_accurace to the accuracy
    # Obs: This function only sets value during mode == ModeKeys.EVAL
    # To set values during training, see tf.summary.scalar
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops={'my_accuracy': accuracy})

    # If mode is not PREDICT nor EVAL, then we must be in TRAIN
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"

    # 3. Training mode

    # Default optimizer for DNNClassifier: Adagrad with learning rate=0.05
    # Our objective (train_op) is to minimize loss
    # Provide global step counter (used to count gradient updates)
    optimizer = tf.train.AdagradOptimizer(0.05)
    train_op = optimizer.minimize(
        loss,
        global_step=tf.train.get_global_step())

    # Set the TensorBoard scalar my_accuracy to the accuracy
    # Obs: This function only sets the value during mode == ModeKeys.TRAIN
    # To set values during evaluation, see eval_metrics_ops
    tf.summary.scalar('my_accuracy', accuracy[1])

    # Return training operations: loss and train_op
    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=train_op)







# Create a custom estimator using my_model_fn to define the model
classifier = tf.estimator.Estimator(
    model_fn=my_model_fn,
    model_dir=PATH)  # Path to where checkpoints etc are stored













# Train our model, use the previously function my_input_fn
# Input to training is a file with training example
# Stop training after 8 iterations of train data (epochs)
classifier.train(
    input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))



# Evaluate our model using the examples contained in FILE_TEST
# Return value will contain evaluation_metrics such as: loss & average_loss
evaluate_result = classifier.evaluate(input_fn=lambda: my_input_fn(FILE_TEST, 1))
tf.logging.info("Evaluate results")
for key in evaluate_result:
    tf.logging.info("   {}, was: {}".format(key, evaluate_result[key]))





# Evaluate our model using the examples contained in FILE_TEST
# Evaluate accuracy.
accuracy_score = classifier.evaluate(input_fn=lambda: my_input_fn(FILE_TEST, 1))["my_accuracy"]
print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


