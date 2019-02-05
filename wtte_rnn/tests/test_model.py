import pytest
from wtte_rnn.models.model import WTTERNN
from wtte_rnn.data.factory import get_dataset
from wtte_rnn.data.dataset import get_censored

from train import fit


class TestModel(object):
    def test_build(self):
        fit("fake")

        assert True
