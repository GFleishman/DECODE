import torch
import pytest
from unittest import mock

from decode.generic import emitter
from decode.utils import emitter_io


@pytest.fixture()
def em_rand():
    return emitter.RandomEmitterSet(20, xy_unit='px', px_size=(100, 200))


@pytest.fixture()
def em_all_attrs(em_rand):
    em_rand = em_rand.clone()
    em_rand.xyz_sig = torch.rand(20, 3)
    em_rand.xyz_cr = torch.rand_like(em_rand.xyz_sig)
    em_rand.phot_sig = torch.rand(20)
    em_rand.phot_cr = torch.rand_like(em_rand.phot_sig)
    em_rand.bg_sig = torch.rand(20)
    em_rand.bg_cr = torch.rand_like(em_rand.bg_sig)

    return em_rand


def test_save_load_h5py(em_rand, em_all_attrs, tmpdir):
    path = tmpdir / 'emitter.h5'

    for em in (em_rand, em_all_attrs):
        emitter_io.save_h5(path, em.data, em.meta)

        data, meta, decode_meta = emitter_io.load_h5(path)
        em_h5 = emitter.EmitterSet(**data, **meta)

        assert em == em_h5  # if equality check is wrong, this is wrong as well
        assert decode_meta['version'][0] == 'v'


@pytest.mark.parametrize('last_index', ['including', 'excluding'])
def test_streamer(last_index, tmpdir):

    stream = emitter_io.EmitterWriteStream('dummy', '.pt', tmpdir, last_index=last_index)

    with mock.patch.object(emitter.EmitterSet, 'save') as mock_save:
        stream.write(emitter.RandomEmitterSet(20), 0, 100)

    if last_index == 'including':
        mock_save.assert_called_once_with(tmpdir / 'dummy_0_100.pt')
    elif last_index == 'excluding':
        mock_save.assert_called_once_with(tmpdir / 'dummy_0_99.pt')

    with mock.patch.object(emitter.EmitterSet, 'save') as mock_save:
        stream(emitter.RandomEmitterSet(20), 0, 100)

    mock_save.assert_called_once()
