# Import TensorFlow.
import tensorflow as tf

# Basic operations with some constants

# Create constants a and b

# Build a data flow graph
a = tf.constant(10)
b = tf.constant(5)

# Create a session to evaluate the symbolic expression
sess = tf.Session() 

# Trigger an evaluation of the data flow graph.
print("Constants addition: %i" % sess.run(a + b))
print("Constants  multiplication: %i" % sess.run(a * b))

# Close the session
sess.close()