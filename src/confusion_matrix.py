# This file demonstrates the [confusion_matrix](https://www.tensorflow.org/versions/master/api_docs/python/tf/confusion_matrix).

import tensorflow as tf

y_ = [0, 2, 2, 2]
y = [2, 1, 2, 2]

with tf.Session() as sess:
    confusion_matrix = tf.confusion_matrix(labels=y_, predictions=y, num_classes=4)
    confusion_matrix_to_Print = sess.run(confusion_matrix)
    print(confusion_matrix_to_Print)
