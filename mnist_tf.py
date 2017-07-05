

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE = 0.01
TRAINING_STEPS = 10000

def train(mnist):
	x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
	y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')


	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
	bias1 = tf.Variable(tf.constant(0.0, shape=[LAYER1_NODE]))
	weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
	bias2 = tf.Variable(tf.constant(0.0, shape=[OUTPUT_NODE]))

	layer1 = tf.nn.relu(tf.matmul(x, weights1) + bias1)
	y = tf.matmul(layer1, weights2) + bias2
	
	global_step = tf.Variable(0, trainable=False)
	
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
	loss = tf.reduce_mean(cross_entropy)
	
	train_op=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
	
	with tf.Session() as sess:
		tf.initialize_all_variables().run()
		validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
	
		test_feed = {x: mnist.test.images, y_: mnist.test.labels}
		
		for i in range(TRAINING_STEPS):
			if i % 1000 == 0:
				validate_acc = sess.run(accuracy, feed_dict=validate_feed)
				print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
			
			xs, ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op, feed_dict={x: xs, y_: ys})

		test_acc = sess.run(accuracy, feed_dict=test_feed)
		print("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None): 
	
	mnist = input_data.read_data_sets("./data", one_hot=True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run()
