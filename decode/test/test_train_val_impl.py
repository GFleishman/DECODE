import pytest
import torch

from decode.neuralfitter import loss
from decode.neuralfitter import post_processing
from decode.neuralfitter import train_val_impl
from decode.neuralfitter.models import unet_param
from decode.neuralfitter.utils import logger as logger_utils


class TestTrain:

    @pytest.fixture()
    def model(self):
        return unet_param.UNet2d(in_channels=1, out_channels=6, depth=2, initial_features=16, pad_convs=True)

    @pytest.fixture()
    def opt(self, model):
        return torch.optim.Adam(model.parameters())

    @pytest.fixture()
    def loss(self):
        return loss.PPXYZBLoss(torch.device('cpu'))

    @pytest.fixture()
    def dataloader(self):
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 256

            def __getitem__(self, item):
                return torch.rand((1, 64, 64)), torch.rand((6, 64, 64)), torch.rand((6, 64, 64))

        return torch.utils.data.DataLoader(MockDataset(), batch_size=32)

    @pytest.fixture()
    def logger(self):
        return logger_utils.NoLog()

    @pytest.fixture()
    def train_val_environment(self, loss, model):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # ship the things to the apropriate device
        loss.__init__(device=device)
        model = model.to(device)
        param_before = model.encoder[0][0].weight.data.clone()
        return device, model, param_before

    def test_iterate_batch(self, opt, loss, dataloader, logger, train_val_environment):
        device, model, param_before = train_val_environment

        """Run"""
        train_val_impl.train(model, opt, loss, dataloader, False, False, 0, device, logger) is None
        param_after = model.encoder[0][0].weight.data.clone()

        assert (param_before != param_after).any(), "No weights changed although they should have."


class TestVal(TestTrain):

    @pytest.fixture()
    def post_processor(self):
        return post_processing.ConsistencyPostprocessing(raw_th=0.1, em_th=0.5, xy_unit='nm', img_shape=(64, 64),
                                                         lat_th=100)

    def test_iterate_batch(self, model, opt, loss, dataloader, post_processor, logger, train_val_environment):
        device, model, param_before = train_val_environment

        """Run"""
        train_val_impl.test(model, loss, dataloader, 0, device)
        param_after = model.encoder[0][0].weight.data.clone()

        """Assert"""
        assert (param_before == param_after).all(), "Weights must not change."
