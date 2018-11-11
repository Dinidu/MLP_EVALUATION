import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
from util import constant as const
from dataset.dataset import Dataset
import numpy


class Model(object):
    def __init__(self):
        self.dataset = Dataset()
        # This is where training samples and labels are fed to the graph.
        # These placeholder nodes will be fed a batch of training data at each
        # training step, which we'll write once we define the graph structure.
        self.train_data_node = tf.placeholder(
            tf.float32,
            shape=(const.BATCH_SIZE, const.IMAGE_SIZE, const.IMAGE_SIZE, const.NUM_CHANNELS)
        )
        self.train_labels_node = tf.placeholder(
            tf.float32,
            shape=(const.BATCH_SIZE, const.NUM_CLASSES)
        )

        # For the validation and test data, we'll just hold the entire dataset in
        # one constant node.

        self.validation_data_node = tf.constant(self.dataset.validation_data)
        self.test_data_node = tf.constant(self.dataset.test_data)

        # The variables below hold all the trainable weights. For each, the
        # parameter defines how the variables will be initialized.
        self.conv1_weights = tf.Variable(
            tf.truncated_normal([5, 5, const.NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
            stddev=0.1,
            seed=const.SEED)
        )
        self.conv1_biases = tf.Variable(tf.zeros([32]))

        self.conv2_weights = tf.Variable(
            tf.truncated_normal([5, 5, 32, 64],
            stddev=0.1,
            seed=const.SEED)
        )
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal([const.IMAGE_SIZE // 4 * const.IMAGE_SIZE // 4 * 64, 512],
            stddev=0.1,
            seed=const.SEED)
        )
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))

        self.fc2_weights = tf.Variable(
            tf.truncated_normal([512, const.NUM_CLASSES],
            stddev=0.1,
            seed=const.SEED))
        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[const.NUM_CLASSES]))

    def model(self,data, train=const.DROP_OUT):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            self.conv1_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')

        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))

        # Max pooling. The kernel size spec ksize also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            self.conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)

        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 0.5, seed=const.SEED)
        return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases


    def train(self,regularize=const.L2_REGULARIZE):
        # Training computation: logits + cross-entropy loss.
        logits = self.model(self.train_data_node, False)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.train_labels_node,logits=logits))

        # L2 regularization for the fully connected parameters.
        if regularize:

            regularizers = (
                    tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                    tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases)
            )
            # Add the regularization term to the loss.
            loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        learning_rate = tf.train.exponential_decay(
            const.LEARNING_RATE,         # Base learning rate.
            batch * const.BATCH_SIZE,    # Current index into the dataset.
            self.dataset.train_size,    # Decay step.
            const.DECAY_RATE,            # Decay rate.
            staircase=True
        )

        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(learning_rate,const.MOMENTUM).minimize(loss,global_step=batch)

        # Predictions for the minibatch, validation set and test set.
        train_prediction = tf.nn.softmax(logits)
        # We'll compute them only once in a while by calling their {eval()} method.
        validation_prediction = tf.nn.softmax(self.model(self.validation_data_node))
        test_prediction = tf.nn.softmax(self.model(self.test_data_node))

        # Create a new interactive session that we'll use in
        # subsequent code cells.
        s = tf.InteractiveSession()

        # Use our newly created session as the default for
        # subsequent operations.
        s.as_default()

        # Initialize all the variables we defined above.
        tf.global_variables_initializer().run()

        steps = self.dataset.train_size // const.BATCH_SIZE
        for step in range(steps):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * const.BATCH_SIZE) % (self.dataset.train_size - const.BATCH_SIZE)
            batch_data = self.dataset.train_data[offset:(offset + const.BATCH_SIZE), :, :, :]
            batch_labels = self.dataset.train_labels[offset:(offset + const.BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.
            feed_dict = {self.train_data_node: batch_data,
                         self.train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = s.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)

            # Print out the loss periodically.
            if step % 100 == 0:
                error, _ = self.error_rate(predictions, batch_labels)
                print('Step %d of %d' % (step, steps))
                print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
                print('Validation error: %.1f%%' % self.error_rate(
                    validation_prediction.eval(), self.dataset.validation_labels)[0])

    def error_rate(self,predictions, labels):
        """Return the error rate and confusions."""
        correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
        total = predictions.shape[0]

        error = 100.0 - (100 * float(correct) / float(total))

        confusions = numpy.zeros([10, 10], numpy.float32)
        bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(labels, 1))
        for predicted, actual in bundled:
            confusions[predicted, actual] += 1

        return error, confusions