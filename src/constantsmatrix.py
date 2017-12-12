# Import TensorFlow.
import tensorflow as tf

# Perform the same operations on constant matrices

# Build a dataflow graph
c = tf.constant([[2.0, 2.0], [5.0, 4.0]])
d = tf.constant([[1.0, 2.0], [1.0, 2.0]])
e = tf.matmul(c, d)

# Construct a `Session` to execute the graph
sess = tf.Session()

# Execute the graph and print the resulting matrix
print(sess.run(e))

# Close the session
sess.close()