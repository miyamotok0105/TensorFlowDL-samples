import tensorflow as tf
import mnist_input_data

#tfの引数を定義
tf.app.flags.DEFINE_integer('training_iteration', 1000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
# x = tf.reshape(images, [-1, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())
scores = tf.nn.softmax(tf.matmul(x, w) + b, name='scores')
cross_entropy = -tf.reduce_sum(y_ * tf.log(scores))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
for _ in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# print("train_step==")
# print(train_step)
# print("scores==")
# print(scores)
# print(scores.eval())

sess.close()

