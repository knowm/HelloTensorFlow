# Import TensorFlow.
import tensorflow as tf

# Operations with variables as graph input flow
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Apply some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

# Construct a `Session` to execute the graph.
sess = tf.Session()

# Execute the graph with variable input
print("Addition input variables: %i" % sess.run(add, feed_dict={a: 5, b: 8}))
print("Multiplication input variables: %i" % sess.run(mul, feed_dict={a: 5, b: 8}))

# Close the session
sess.close()