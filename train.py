import argparse
import tensorflow as tf
from wtte_rnn.models.model import WTTERNN
from wtte_rnn.data.factory import get_dataset
from wtte_rnn.data.dataset import get_censored

n_timesteps = 200
n_sequences = every_nth = 80
n_features = 1
n_repeats = 1000
noise_level = 0.005
use_censored = True


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", default="fake")
    arg_parser.add_argument('--summaries-dir', default="logs")
    args = arg_parser.parse_args()

    dataset = get_dataset(args.dataset)
    x_train, y_train, x_test, y_test, events = dataset(n_timesteps, every_nth, n_repeats, noise_level, n_features,
                                                       n_sequences, use_censored)
    y_censor = get_censored(y_train)
    y_train = y_train[:, :, 0]

    model = WTTERNN(
        nb_units=2,
        nb_features=n_features,
        sequence_length=n_timesteps
    )
    model.fit(x_train, (y_train, y_censor))


if __name__ == '__main__':
    main()
