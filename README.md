## Intro

This project aims to be a collection of notes, links, code snippets and mini-guides to teach you how to get Tensorflow up and running on MacOS (CPU only), Windows 10 (CPU and GPU) and Linux (work in progress) with zero experience in Tensorflow and little or no background in Python. It also runs through some basic machine learning code and concepts and focuses on specific details of TensorFlow as they are seen for the first time. This project assumes you are familiar with the command line, git and the most common developer tools of your chosen operating system. The code samples can be run in the console as is the most common scenario, but I also show how to set up Eclipse and the PyDev developer environment for Python motivated by my need for a cross platform IDE with code highlighting and other helpful IDE tools.

Many of the code examples have been taken from other Internet resources and credit is always given. We welcome pull requests for corrections, updates or additional tips, etc. 

## Pre-requisites

Before walking through this README, take the time to read the `Dev4Windows.md` or `Dev4MacOS.md` to get things setup and installed and make sure you can successfully run `hellopy.py` and `hellotf.py` as well as some examples from the Tensorflow/models repo from the command line as well as from within Eclipse.

### MacOS

1. Dev4MacOS.md

### Windows 10

1. Dev4Windows.md

### Linux

TODO


## TF Walkthrough

The following 5 python (`*.py`) files are located in `src` and were adapted from  [blog.altoros.com](https://blog.altoros.com).

1. contants.py
1. contantsmatrix.py
1. variables.py
1. matrices.py
1. linear_regression.py

Linear Regression fits a line to a smattering of continuous X-Y values. At each iteration, the STG method is used along with the least squared error cost function. In this example the data is first normalized.

1. [Source for TF basics](https://blog.altoros.com/basic-concepts-and-manipulations-with-tensorflow.html)
1. [Source for linear regression](https://blog.altoros.com/using-linear-regression-in-tensorflow.html)
1. [Placeholders](https://www.tensorflow.org/versions/r0.12/api_docs/python/io_ops/placeholders)


You may need to install the matplotlib library:

```
pip3 install matplotlib
```


## Continuation

At this point we can leverage the project called [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples) to learn more M.L. in Tensorflow concepts. Go ahead and clone that repo and try running some of the examples either in the console or in Eclipse.

At this point, you could also use the Jupyter Notebook IDE for working with the examples, as it provides many nice conveniences although I'm quite happy with my Eclipse IDE setup.

```
python3 -m pip install jupyter
jupyter notebook
```

## Logistic Regression (logistic_regression.py)

Logistic Regression is like Linear Regression except it allows you to classify things rather than predict a continuous value. In logistic regression, the hypothesis function is the sigmoid function. The sigmoid function is bounded between 0 and 1, and produces a value that can be interpreted as a probability. This value can also be a yes / no answer with a cross-over, or decision boundary, at 0.5. The cost function is also changed from Linear Regression to be more compatible with gradient descent. The output is thresholded for a single class classifier and the softmax algo is used for a multi-class classifier.

1. [Logistic Regression](https://crsmithdev.com/blog/ml-logistic-regression/)
1. [Gentlest Intro to Tensorflow #4: Logistic Regression](https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-4-logistic-regression-2afd0cabc54)

### A note on calculating model accuracy

Here is the "testing" code from the above example:

```
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
```

Argmax
: Returns the index with the largest value across axes of a tensor.

axis (the `1` argument for the argmax function)
: A Tensor. Must be one of the following types: int32, int64. int32 or int64, must be in the range [-rank(input), rank(input)). Describes which axis of the input Tensor to reduce across. For vectors, use axis = 0.

In this case we have a matrix with rows representing the softmax score vector of the 10 digits for each test image. Axis = 0 refers to rows, axis = 1 refers to columns.

* [How do you use argmax and gather in Tensorflow?](http://www.michaelburge.us/2017/07/18/how-to-use-argmax-in-tensorflow.html)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
: a matrix (10,000 rows by 1 column) of booleans representing the correct guess of the predictor according to the truth labels

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
: this converts the booleans to floats and calculates the mean of all elements in the matrix (tensor)

## K-Means Clustering (kmeans.py)

For K-Means clustering, the concept is really simple. You pick `K` centroids and randomly place them. In a loop, you calculate the N closest datapoints to each centroid and calculate the average centroid location. Repeat until it converges. This converges to a local minimum so it is not at all perfect.

1. [Clustering With K-Means in Python](https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/)
1. [How would I implement k-means with TensorFlow?](https://stackoverflow.com/questions/33621643/how-would-i-implement-k-means-with-tensorflow)
1. [K-Means Clustering on Handwritten Digits](http://johnloeber.com/docs/kmeans.html)
1. [TF KMeans](https://www.tensorflow.org/api_docs/python/tf/contrib/factorization/KMeans)

import os os.environ["CUDA_VISIBLE_DEVICES"] = ""
: Ignore all GPUs, tf random forest does not benefit from it. I think I read somewhere it's even better to set it to -1. If you don't set this and there is a GPU on the system it will run it on the GPU. This forces it to use the CPU.

## Nearest Neighbor (nearest_neighbor.py)

In this example there is no training at all. You just give it a test image and all the "train" images and it sees which one it is closest to.

## Random Forest (random_forest.py)

1. [TensorForest](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tensor_forest)
1. [Wikipedia Random Forest](https://en.wikipedia.org/wiki/Random_forest#ExtraTrees)
1. [TensorForest on Iris](https://www.kaggle.com/thomascolthurst/tensorforest-on-iris/notebook)

This demo uses the `tensor_forest` from the tf contrib package. It forms 10 trees, using the individual pixels as features. Each branch in the tree looks at the pixel values and makes a decision: left or right. The number of nodes in the trees is sqrt(numFeatures), and each node gathers statistics about the data it sees in order to determine an "optimal" threshold value for the decision. The outputs of all 10 trees are averaged to determine the combined classification output of the entire forest. 

Again, here we ignore all GPUs, tf random forest does not benefit from it.

## Saving and Importing a Model (save_restore_model.py)

Here the model is saved to disk and then restored later for inference use. It looks like the saving needs to be implemented in the Model itself.

```
model_path = "/tmp/model.ckpt"
# 'Saver' op to save and restore all the variables
saver = tf.train.Saver() // saves all variables, you could specify which ones.
...
# Save model weights to disk
save_path = saver.save(sess, model_path)
...
# Restore model weights from previously saved model
saver.restore(sess, model_path)
```

1. [TensorFlow: Save and Restore Models](http://stackabuse.com/tensorflow-save-and-restore-models/)
1. [Official TF Docs: Saving and Restoring Variables](https://www.tensorflow.org/versions/r0.12/api_docs/python/state_ops/saving_and_restoring_variables)

When saving the model, you'll notice there are 4 types of files:

1. ".meta" files: containing the graph structure
1. ".data" files: containing the values of variables
1. ".index" files: identifying the checkpoint
1. "checkpoint" file: a protocol buffer with a list of recent checkpoints

Checkpoints are binary files in a proprietary format which map variable names to tensor values. Savers can automatically number checkpoint filenames with a provided counter. This lets you keep multiple checkpoints at different steps while training a model. 

A few other useful arguments of the Saver constructor, which enable control of the whole process, are:

* `max_to_keep`: maximum number of checkpoints to keep,
* `keep_checkpoint_every_n_hours`: a time interval for saving checkpoints


## Tensorboard Basic (tensorboard_basic.py)

Tensorboard allows you to view the graph as well as model parameters, updating live.

```
logs_path = '/tmp/tensorflow_logs/example/'
...
# op to write logs to Tensorboard
summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
...
# Write logs at every iteration
summary_writer.add_summary(summary, epoch * total_batch + i)
...
print("Run the command line:\n" \
      "--> tensorboard --logdir=/tmp/tensorflow_logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
```

1. [TensorBoard README on Github](https://github.com/tensorflow/tensorboard)
1. [TensorBoard: Graph Visualization](https://www.tensorflow.org/get_started/graph_viz)
1. [Visualizing TensorFlow Graphs with TensorBoard](https://blog.altoros.com/visualizing-tensorflow-graphs-with-tensorboard.html)

### Name scoping and nodes

Typical TensorFlow graphs can have many thousands of nodes--far too many to see easily all at once, or even to lay out using standard graph tools. To simplify, variable names can be scoped and the visualization uses this information to define a hierarchy on the nodes in the graph. By default, only the top of this hierarchy is shown. Here is an example that defines three operations under the hidden name scope using tf.name_scope:

```
import tensorflow as tf

with tf.name_scope('hidden') as scope:
  a = tf.constant(5, name='alpha')
  W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), name='weights')
  b = tf.Variable(tf.zeros([1]), name='biases')
```
  
Grouping nodes by name scopes is critical to making a legible graph. If you're building a model, name scopes give you control over the resulting visualization. The better your name scopes, the better your visualization.

### Tensor shape information

When the serialized GraphDef includes tensor shapes, the graph visualizer labels edges with tensor dimensions, and edge thickness reflects total tensor size. To include tensor shapes in the GraphDef pass the actual graph object (as in sess.graph) to the FileWriter when serializing the graph.

### Runtime statistics

Often it is useful to collect runtime metadata for a run, such as total memory usage, total compute time, and tensor shapes for nodes. The code example below is a snippet from the train and test section of a modification of the simple MNIST tutorial, in which we have recorded summaries and runtime statistics. See the Summaries Tutorial for details on how to record summaries. When you launch tensorboard and go to the Graph tab, you will now see options under "Session runs" which correspond to the steps where run metadata was added. Selecting one of these runs will show you the snapshot of the network at that step, fading out unused nodes. In the controls on the left hand side, you will be able to color the nodes by total memory or total compute time. Additionally, clicking on a node will display the exact total memory, compute time, and tensor output sizes. 

```
  # Train the model, and also write summaries.
  # Every 10th step, measure test-set accuracy, and write test summaries
  # All other steps, run train_step on training data, & add training summaries

  def feed_dict(train):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if train or FLAGS.fake_data:
      xs, ys = mnist.train.next_batch(100, fake_data=FLAGS.fake_data)
      k = FLAGS.dropout
    else:
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    return {x: xs, y_: ys, keep_prob: k}

  for i in range(FLAGS.max_steps):
    if i % 10 == 0:  # Record summaries and test-set accuracy
      summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
      test_writer.add_summary(summary, i)
      print('Accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      if i % 100 == 99:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict(True),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
        train_writer.add_summary(summary, i)
        print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)
 ```

This code will emit runtime statistics for every 100th step starting at step 99.

1. [Source and Code](https://www.tensorflow.org/get_started/graph_viz)

### A Video to Watch 

[Hands-on TensorBoard (TensorFlow Dev Summit 2017)](https://www.youtube.com/watch?v=eBbEDRsCmv4)

We can do some amazing data mining, insight and comparison with TensorBoard. Also, we should be able to debug stuff like Nans, Infs and Tensor shapes and data soon if not yet already.

## Advanced Examples from [TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)

1. `build_an_image_dataset.py` - create a custom image dataset.
1. `multigpu_basics.py` - how to assign different parts of the graph to different GPUs.

