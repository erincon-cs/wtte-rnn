import numpy as np
import pandas as pd
import tqdm
import os

from tqdm import tqdm

from sklearn import pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_censored(x, n_timesteps):
    x = np.all(np.all(x != 0, axis=-1), axis=1).astype(int)
    x = x[:, np.newaxis]
    x = np.repeat(x, n_timesteps, axis=-1)

    return x


def get_data(n_timesteps, every_nth, n_repeats, noise_level, n_features, n_sequences=80, use_censored=True):
    def get_equal_spaced(n, every_nth):
        # create some simple data of evenly spaced events recurring every_nth step
        # Each is on (time,batch)-format
        events = np.array([np.array(range(n)) for _ in range(every_nth)])
        events = events + np.array(range(every_nth)).reshape(every_nth, 1) + 1

        tte_actual = every_nth - 1 - events % every_nth

        was_event = (events % every_nth == 0) * 1.0
        was_event[:, 0] = 0.0

        events = tte_actual == 0

        is_censored = (events[:, ::-1].cumsum(1)[:, ::-1] == 0) * 1
        tte_censored = is_censored[:, ::-1].cumsum(1)[:, ::-1] * is_censored
        tte_censored = tte_censored + (1 - is_censored) * tte_actual

        events = np.copy(events.T * 1.0)
        tte_actual = np.copy(tte_actual.T * 1.0)
        tte_censored = np.copy(tte_censored.T * 1.0)
        was_event = np.copy(was_event.T * 1.0)
        not_censored = 1 - np.copy(is_censored.T * 1.0)

        return tte_censored, not_censored, was_event, events, tte_actual

    tte_censored, not_censored, was_event, events, tte_actual = get_equal_spaced(n=n_timesteps, every_nth=every_nth)

    # From https://keras.io/layers/recurrent/
    # input shape rnn recurrent if return_sequences: (nb_samples, timesteps, input_dim)

    u_train = not_censored.T.reshape(n_sequences, n_timesteps, 1)
    x_train = was_event.T.reshape(n_sequences, n_timesteps, 1)
    tte_censored = tte_censored.T.reshape(n_sequences, n_timesteps, 1)
    y_train = np.append(tte_censored, u_train, axis=2)  # (n_sequences,n_timesteps,2)

    u_test = np.ones(shape=(n_sequences, n_timesteps, 1))
    x_test = np.copy(x_train)
    tte_actual = tte_actual.T.reshape(n_sequences, n_timesteps, 1)
    y_test = np.append(tte_actual, u_test, axis=2)  # (n_sequences,n_timesteps,2)

    if not use_censored:
        x_train = np.copy(x_test)
        y_train = np.copy(y_test)
    # Since the above is deterministic perfect fit is feasible. 
    # More noise->more fun so add noise to the training data:

    x_train = np.tile(x_train.T, n_repeats).T
    y_train = np.tile(y_train.T, n_repeats).T

    # Try with more than one feature TODO
    x_train_new = np.zeros([x_train.shape[0], x_train.shape[1], n_features])
    x_test_new = np.zeros([x_test.shape[0], x_test.shape[1], n_features])
    for f in range(n_features):
        x_train_new[:, :, f] = x_train[:, :, 0]
        x_test_new[:, :, f] = x_test[:, :, 0]

    x_train = x_train_new
    x_test = x_test_new

    # xtrain is signal XOR noise with probability noise_level
    noise = np.random.binomial(1, noise_level, size=x_train.shape)
    x_train = x_train + noise - x_train * noise

    return x_train, y_train, x_test, y_test, events


class EngineData:
    """
    Adapted from https://github.com/gm-spacagna/deep-ttf/blob/master/notebooks/Keras-WTT-RNN%20Engine%20failure.ipynb
    """

    def __init__(self):
        self.column_names, self.feature_columns = self._column_names()

        data_path = "enginedata"
        save = False

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        train_path = os.path.join(data_path, 'train.csv')
        test_x_path = os.path.join(data_path, 'test_x.csv')
        test_y_path = os.path.join(data_path, 'test_y.csv')

        if not os.path.exists(train_path) or not os.path.exists(test_x_path) or os.path.exists(test_y_path):
            train_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/train.csv',
                                     header=None, names=self.column_names)
            test_x_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_x.csv',
                                      header=None, names=self.column_names)
            test_y_orig = pd.read_csv('https://raw.githubusercontent.com/daynebatten/keras-wtte-rnn/master/test_y.csv',
                                      header=None, names=['T'])
            save = True
        else:
            train_orig = pd.read_csv(train_path)
            test_x_orig = pd.read_csv(test_x_path)
            test_y_orig = pd.read_csv(test_y_path)

        test_x_orig.set_index(['unit_number', 'time'], verify_integrity=True)

        # Combine the X values to normalize them,
        all_data_orig = pd.concat([train_orig, test_x_orig])
        # all_data = all_data[feature_cols]
        # all_data[feature_cols] = normalize(all_data[feature_cols].values)

        scaler = pipeline.Pipeline(steps=[
            #     ('z-scale', StandardScaler()),
            ('minmax', MinMaxScaler(feature_range=(-1, 1))),
            ('remove_constant', VarianceThreshold())
        ])

        all_data = all_data_orig.copy()
        all_data = np.concatenate(
            [all_data[['unit_number', 'time']], scaler.fit_transform(all_data[self.feature_columns])],
            axis=1)

        # then split them back out
        train = all_data[0:train_orig.shape[0], :]
        test = all_data[train_orig.shape[0]:, :]

        # Make engine numbers and days zero-indexed, for everybody's sanity
        train[:, 0:2] -= 1
        test[:, 0:2] -= 1

        max_time = 100
        mask_value = -99

        self.x_train, y_train = self._build_data(engine=train[:, 0], time=train[:, 1], x=train[:, 2:],
                                                 max_time=max_time,
                                                 is_test=False, mask_value=mask_value)

        self.y_train = y_train[:, 0]
        self.z_train = y_train[:, 1]

        self.test_x, _ = self._build_data(engine=test[:, 0], time=test[:, 1], x=test[:, 2:], max_time=max_time,
                                          is_test=True,
                                          mask_value=mask_value)

        self.init_value = self.get_init_values()

        self.nb_features = self.x_train.shape[-1]
        self.nb_timesteps = self.x_train.shape[1]

        self.y_train = np.tile(np.expand_dims(self.y_train, axis=-1), self.nb_timesteps)
        self.z_train = np.tile(np.expand_dims(self.z_train, axis=-1), self.nb_timesteps)


        #
        # if save:
        #     train_orig.to_csv(train_path, index=False)
        #     test_x_orig.to_csv(test_x_orig, index=False)
        #     test_y_orig.to_csv(test_y_orig, index=False)

    def _column_names(self):
        id_col = 'unit_number'
        time_col = 'time'
        feature_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3'] + ['sensor_measurement_{}'.format(x) for x in
                                                                           range(1, 22)]
        column_names = [id_col, time_col] + feature_cols

        return column_names, feature_cols

    # TODO: replace using wtte data pipeline routine
    def _build_data(self, engine, time, x, max_time, is_test, mask_value):
        # y[0] will be days remaining, y[1] will be event indicator, always 1 for this data
        out_y = []

        # number of features
        d = x.shape[1]

        # A full history of sensor readings to date for each x
        out_x = []

        n_engines = 100
        for i in tqdm(range(n_engines)):
            # When did the engine fail? (Last day + 1 for train data, irrelevant for test.)
            max_engine_time = int(np.max(time[engine == i])) + 1

            if is_test:
                start = max_engine_time - 1
            else:
                start = 0

            this_x = []

            for j in range(start, max_engine_time):
                engine_x = x[engine == i]

                out_y.append(np.array((max_engine_time - j, 1), ndmin=2))

                xtemp = np.zeros((1, max_time, d))
                xtemp += mask_value
                #             xtemp = np.full((1, max_time, d), mask_value)

                xtemp[:, max_time - min(j, 99) - 1:max_time, :] = engine_x[max(0, j - max_time + 1):j + 1, :]
                this_x.append(xtemp)

            this_x = np.concatenate(this_x)
            out_x.append(this_x)
        out_x = np.concatenate(out_x)
        out_y = np.concatenate(out_y)

        return out_x, out_y

    def get_init_values(self):
        tte_mean_train = np.nanmean(self.y_train)
        mean_u = np.nanmean(self.z_train)

        # Initialization value for alpha-bias
        init_alpha = -1.0 / np.log(1.0 - 1.0 / (tte_mean_train + 1.0))
        init_alpha = init_alpha / mean_u

        return init_alpha

    def __call__(self, index, *args, **kwargs):
        return self.x_train[index, :, :], self.train


class DavitaDataDict:
    """
    Python class wrapper for the Davita provided data dictionary excel file
    """

    def __init__(self, data_dict_path, database=None, table=None):
        self.database = database
        self.table = table
        self.data_dict_path = data_dict_path

        self.data_dict = self._setup(data_dict_path)

    def __getitem__(self, index):
        return self.data_dict.iloc[index, :]

    def _setup(self, data_dict_path):
        df = pd.ExcelFile(data_dict_path)
        snappy_elements = df.parse('Snappy Elements')
        columns = snappy_elements.iloc[2, :]
        columns = list(columns.name) + list(columns.values)

        # reset last duplicate table column to 'Table Type'
        columns[-1] = "Table Type"
        columns[len(columns) - 2] = "Column Copy"
        data_dict = snappy_elements.reset_index()
        data_dict = data_dict.iloc[3:, :].reset_index(drop=True)

        data_dict.columns = columns

        data_dict['Database Name'] = data_dict['Database Name'].astype(str)
        data_dict['Table'] = data_dict['Table'].astype(str)
        data_dict['Column'] = data_dict['Column'].astype(str)

        return data_dict

    def table_find_column(self, column):
        """
        Returns the tables with the given column name

        :param column:
        :return:
        """

        tables = self.data_dict[self.data_dict['Column'] == column]["Table"]
        tables = tables.sort_values()

        return tables

    def table_contains_phrase(self, phrase):
        """
        Returns the tables with columns that contain the passed phrase

        :param column:
        :return:
        """

        tables = self.data_dict[self.data_dict['Column'].apply(lambda column: phrase in column.lower())]["Table"]
        tables = tables.sort_values()

        return tables

    def table_describe(self, table_df: pd.DataFrame, table_db: str, table_name: str):
        """
        Takes in DataFrame read from a Davita table csv and creates the summary statistics and concats the
        descriptions for each column

        :param table_df:
        :param table_db:
        :param table_name:
        :return:
        """
        table_df_stats = table_df.describe(include='all').T

        mask = (self.data_dict.Table.str.lower() == table_name.lower()) \
               & (self.data_dict['Database Name'].str.lower() == table_db.lower())

        table_dict = self.data_dict[mask]

        description = table_dict.Description

        description.index = table_dict.Column.str.lower()

        table_df_stats.index = table_df_stats.index.str.lower()

        table_df = table_df_stats.join(description, how="outer")

        table_df_na = pd.DataFrame(table_df.isna().sum() / len(table_df) * 100)
        table_df_na.index = table_df_na.index.str.lower()
        table_df = table_df.join(table_df_na, how="outer")

        return table_df

    @classmethod
    def parse_csv_name(cls, csv_name):
        csv_name_split = csv_name.split("_")
        table_databsae, table_name = csv_name_split[1], csv_name_split[3]

        return table_databsae, table_name


class DavitaTableDict(DavitaDataDict):
    """
    Inherits from DavitaDataDict and requires a table and database
    Mostly to  keep functions neat....
    """

    def __init__(self, data_dict_path, database, table):
        super(DavitaTableDict, self).__init__(data_dict_path, database, table)

    def describe_column(self, column):
        """
        Returns the pandas row for a given column name

        its disgusting right now
        :param column: the column name
        :return: (pd.Series) the row for the given column name
        """
        return self.data_dict[(self.data_dict['Database Name'] == self.database)
                              & (self.data_dict['Table'] == self.table)
                              & (self.data_dict['Column'] == column)]

    def get_column_description(self, column):
        """
        Returns the description for a column in a table

        @TODO rewrite the way the value is gotten from the series its disgusting right now
        :param column: the column name
        :return: (str) the column description
        """
        return str(self.describe_column(column)["Description"].values[0])


class DavitaDataSet:
    def __init__(self, data_path, data_dict_path):
        self.data_path = data_path
        self.data = pd.read_csv(data_path, delimiter='\|')

    def __getitem__(self, item):
        pass
