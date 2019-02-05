import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from wtte_rnn.models.utils import lazy_property
from wtte_rnn.models.losses import weibull_beta_penalty

tfb = tfp.bijectors
tfd = tfp.distributions

_cells = {
    'lstm': tf.nn.rnn_cell.LSTMCell,
    'lstm_cudnn': tf.contrib.cudnn_rnn.CudnnLSTM
}


def _get_cell(cell_type: str):
    cell_type = cell_type.strip().lower()

    if cell_type not in _cells:
        raise ValueError('Cell type ({}) not defined!'.format(cell_type))

    return _cells[cell_type]


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class WTTERNN:
    def __init__(self, nb_units, nb_features, nb_timesteps, cell_type='lstm', nb_layers=1,
                 summaries_dir='logs', debugging=True, summaries=False):

        super(WTTERNN, self).__init__()

        self.cell_type = cell_type
        self.nb_layers = nb_layers
        self.nb_units = nb_units
        self.nb_features = nb_features
        self.nb_timesteps = nb_timesteps
        self.summaries_dir = summaries_dir
        self.summaries = summaries
        self.debugging = debugging

        self._create_model(self.debugging)

    def _create_model(self, debugging=False):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_timesteps, self.nb_features])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.nb_timesteps])
        self.z = tf.placeholder(dtype=tf.bool, shape=[None, self.nb_timesteps])

        with tf.variable_scope('rnn') as scope:
            carry = self.x
            # Stack n layers
            for i in range(self.nb_layers):
                # cell_type refers to the type of cell e.g., LSTM, GRU, etc
                # RNNCEll is the object class
                RNNCell = _get_cell(self.cell_type)
                rnn_cell = RNNCell(self.nb_units)

                carry, current_state = tf.nn.dynamic_rnn(
                    rnn_cell,
                    carry,
                    dtype=tf.float32
                )

                if self.summaries:
                    variable_summaries(carry)

        self.W_alpha = tf.get_variable('W_alpha', [self.nb_units], initializer=tf.initializers.ones())
        self.W_beta = tf.get_variable('W_beta', [self.nb_units], initializer=tf.initializers.ones())
        self.b_alpha = tf.get_variable('b_alpha', [], initializer=tf.initializers.ones())
        self.b_beta = tf.get_variable('b_beta', [], initializer=tf.initializers.ones())
        self.vars = [self.W_alpha, self.W_beta, self.b_alpha, self.b_beta]

        with tf.name_scope('weibull_parameters'):
            self.alpha = tf.exp(tf.einsum('ijk,k->ij', carry, self.W_alpha) + self.b_alpha)
            self.beta = tf.exp(tf.einsum('ijk,k->ij', carry, self.W_beta) + self.b_beta)

        self.p_y = tfd.TransformedDistribution(
            distribution=tfd.Uniform(low=0., high=1.),
            bijector=tfb.Invert(
                tfb.Weibull(
                    scale=self.alpha,
                    concentration=self.beta
                )
            )
        )

        log_probs = tf.where(
            self.z,
            self.p_y.log_prob(self.y),
            self.p_y.log_survival_function(self.y)
        )

        print(log_probs)

        if self.summaries:
            variable_summaries(log_probs)

        with tf.name_scope('likelihood'):
            ll_avg = tf.reduce_mean(log_probs)

            if self.summaries:
                variable_summaries(ll_avg)

        self.loss = -ll_avg + weibull_beta_penalty(self.beta)

        # Setup debugging
        self.debugging_w_alpha = tf.debugging.check_numerics(self.W_alpha, "", "w_alpha")
        self.debugging_w_beta = tf.debugging.check_numerics(self.W_beta, "", "w_beta")
        self.debugging_b_alpha = tf.debugging.check_numerics(self.b_alpha, "", "b_alpha")
        self.debugging_b_beta = tf.debugging.check_numerics(self.b_beta, "", "b_beta")
        # self.debugging_probabilities = tf.debugging.check_numerics(probabilities, "", probabilities)
        self.debugging_alpha = tf.debugging.check_numerics(self.alpha, "", "alpha")
        self.debugging_beta = tf.debugging.check_numerics(self.beta, "", "beta")

        self.debugging_alpha_zero = tf.debugging.assert_greater(self.alpha, 0.0, name="alpha_zero")
        self.debugging_beta_zero = tf.debugging.assert_greater(self.beta, 0.0, name="beta_zero")

        self.debugging_loss = tf.debugging.check_numerics(self.loss, "", "loss")

        if self.debugging:
            self.debugging_vars = [self.debugging_w_alpha, self.debugging_w_beta,
                                   self.debugging_b_alpha, self.debugging_b_beta,
                                   self.debugging_loss, self.debugging_alpha_zero,
                                   self.debugging_beta_zero, self.debugging_alpha, self.debugging_beta]

    def fit(self, x, y, z, lr=1e-3, nb_epochs=100, batch_size=32):
        if len(y.shape) < 2:
            y = np.expand_dims(y, -1)

        train_op = self.optimize(lr)

        with tf.Session() as session:
            # Normal TensorFlow - initialize values, create a session and run the model
            session.run(tf.global_variables_initializer())

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.summaries_dir + '/train', session.graph)
            test_writer = tf.summary.FileWriter(self.summaries_dir + '/test')

            vars = [train_op, self.loss]

            if self.debugging:
                vars.extend(self.debugging_vars)

            for i in range(nb_epochs):
                for j in range(0, x.shape[0], batch_size):
                    output = session.run(
                        fetches=vars,
                        feed_dict={
                            self.x: x[j: batch_size * (j + 1), :, :],
                            self.y: y[j: batch_size * (j + 1), :],
                            self.z: z[j: batch_size * (j + 1)]
                        }
                    )

                    # train_writer.add_summary(output[0], i * minibatch_i)
                    tf.summary.scalar('loss', output[1])
                    print(output[1])
                    # print(output[len(output) - 2:len(output)])
            # print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))1

    def optimize(self, lr=1e-3):
        self.optimizer = tf.train.AdamOptimizer(lr)

        return self.optimizer.minimize(self.loss)
