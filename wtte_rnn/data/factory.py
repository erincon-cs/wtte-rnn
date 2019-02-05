from wtte_rnn.data.dataset import get_data, EngineData

_datasets = {
    'fake': get_data,
    'engine': EngineData
}


def get_dataset(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name not in _datasets:
        raise ValueError('Dataset {} not defined!'.format(dataset_name))

    return _datasets[dataset_name]
