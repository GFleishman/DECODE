import time
import warnings
from functools import partial
from typing import Union, Callable

import torch
from tqdm import tqdm

from .. import dataset
from ...generic import emitter
<<<<<<< HEAD
from functools import partial
import decode.utils
=======
from ...utils import hardware, frames_io
>>>>>>> origin/master


class Infer:

    def __init__(self, model, ch_in: int, frame_proc, post_proc, device: Union[str, torch.device],
                 batch_size: Union[int, str] = 'auto', num_workers: int = 0, pin_memory: bool = False,
                 forward_cat: Union[str, Callable] = 'emitter'):
        """
        Convenience class for inference.

        Args:
            model: pytorch model
            ch_in: number of input channels
            frame_proc: frame pre-processing pipeline
            post_proc: post-processing pipeline
            device: device where to run inference
            batch_size: batch-size or 'auto' if the batch size should be determined automatically (only use in combination with cuda)
            num_workers: number of workers
            pin_memory: pin memory in dataloader
            forward_cat: method which concatenates the output batches. Can be string or Callable.
            Use 'em' when the post-processor outputs an EmitterSet, or 'frames' when you don't use post-processing or if
            the post-processor outputs frames.
        """

        self.model = model
        self.ch_in = ch_in
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.frame_proc = frame_proc
        self.post_proc = post_proc

        self.forward_cat = None
        self._forward_cat_mode = forward_cat

        if str(self.device) == 'cpu' and self.batch_size == 'auto':
            warnings.warn("Automatically determining the batch size does not make sense on cpu device. "
                          "Falling back to reasonable value.")
            self.batch_size = 64

    def forward(self, frames: torch.Tensor, sig_frames: torch.Tensor = None) -> emitter.EmitterSet:
        """
        Forward frames through model, pre- and post-processing and output EmitterSet

        Args:
            frames:

        """

        """Move Model"""
        model = self.model.to(self.device)
        model.eval()

        if model.sig_in:
            assert sig_frames is not None, 'Noise map has to be provided in addition to the frames'

        """Form Dataset and Dataloader"""
        if model.sig_in:
            if frames.shape == sig_frames.shape:
                ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in, sig_frames=sig_frames)
            else:
                assert frames.shape[-2:] == sig_frames.shape[-2:], "Frames and noise map need to have the same image dimension"
        else:
            ds = dataset.InferenceDataset(frames=frames, frame_proc=self.frame_proc, frame_window=self.ch_in, sig_frames=None)

        dl = torch.utils.data.DataLoader(dataset=ds, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers, pin_memory=self.pin_memory,
                                         collate_fn=decode.neuralfitter.utils.collate.smlm_collate)

        out = []

        with torch.no_grad():
            for sample, sig_sample in tqdm(dl):
                if model.sig_in:
                    if sig_sample is not None:
                        sample = torch.cat([sample, sig_sample], 1)
                    elif sig_frames.ndim == 3:
                        sample = torch.cat([sample, sig_frames.unsqueeze(0).repeat(sample.shape[0], sample.shape[1], 1, 1)], 1)
                    elif sig_frames.ndim == 2:
                        sample = torch.cat([sample, sig_frames.unsqueeze(0).unsqueeze(0).repeat(sample.shape[0], sample.shape[1], 1, 1)], 1)

                x_in = sample.to(self.device)

                # compute output
                y_out = model(x_in)

                """In post processing we need to make sure that we get a single Emitterset for each batch, 
                so that we can easily concatenate."""
                if self.post_proc is not None:
                    out.append(self.post_proc.forward(y_out))
                else:
                    out.append(y_out.detach().cpu())

        """Cat to single emitterset / frame tensor depending on the specification of the forward_cat attr."""
        out = self.forward_cat(out)

        return out

    def _setup_forward_cat(self, forward_cat, batch_size: int):

        if forward_cat is None:
            return lambda x: x

        elif isinstance(forward_cat, str):

            if forward_cat == 'emitter':
                return partial(emitter.EmitterSet.cat, step_frame_ix=batch_size)

            elif forward_cat == 'frames':
                return partial(torch.cat, dim=0)

        elif callable(forward_cat):
            return forward_cat

        else:
            raise TypeError(f"Specified forward cat method was wrong.")

        raise ValueError(f"Unsupported forward_cat value.")

    @staticmethod
    def get_max_batch_size(model: torch.nn.Module, frame_size: Union[tuple, torch.Size],
                           limit_low: int, limit_high: int):
        """
        Get maximum batch size for inference.

        Args: 
            model: model on correct device
            frame_size: size of frames (without batch dimension)
            limit_low: lower batch size limit
            limit_high: upper batch size limit
        """

        def model_forward_no_grad(x: torch.Tensor):
            """
            Helper function because we need to account for torch.no_grad()
            """
            with torch.no_grad():
                o = model.forward(x)

            return o

        assert next(model.parameters()).is_cuda, \
            "Auto determining the max batch size makes only sense when running on CUDA device."

        return hardware.get_max_batch_size(model_forward_no_grad, frame_size, next(model.parameters()).device,
                                           limit_low, limit_high)


class LiveInfer(Infer):
    def __init__(self,
                 model, ch_in: int, *,
                 stream, time_wait=5,
                 frame_proc=None, post_proc=None,
                 device: Union[str, torch.device] = 'cuda:0' if torch.cuda.is_available() else 'cpu',
                 batch_size: Union[int, str] = 'auto', num_workers: int = 0, pin_memory: bool = False,
                 forward_cat: Union[str, Callable] = 'emitter'):

        super().__init__(
            model=model, ch_in=ch_in, frame_proc=frame_proc, post_proc=post_proc,
            device=device, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
            forward_cat=forward_cat)

        self._stream = stream
        self._time_wait = time_wait

    def forward(self, frames: Union[torch.Tensor, frames_io.TiffTensor]):

        n_fitted = 0
        n_waited = 0
        while n_waited <= 2:
            n = len(frames)

            if n_fitted == n:
                n_waited += 1
                time.sleep(self._time_wait)  # wait
                continue

            out = super().forward(frames[n_fitted:n])
            self._stream(out, n_fitted, n)

            n_fitted = n
            n_waited = 0


if __name__ == '__main__':
    import argparse
    import yaml

    import decode.neuralfitter.models
    import decode.utils

    parse = argparse.ArgumentParser(
        description="Inference. This uses the default, suggested implementation. "
                    "For anything else, consult the fitting notebook and make your changes there.")
    parse.add_argument('frame_path', help='Path to the tiff file of the frames')
    parse.add_argument('frame_meta_path', help='Path to the meta of the tiff (i.e. camera parameters)')
    parse.add_argument('model_path', help='Path to the model file')
    parse.add_argument('param_path', help='Path to the parameters of the training')
    parse.add_argument('device', help='Device on which to do inference (e.g. "cpu" or "cuda:0"')
    parse.add_argument('-o', '--online', action='store_true')

    args = parse.parse_args()
    online = args.o

    """Load the model"""
    param = decode.utils.param_io.load_params(args.param_path)

    model = decode.neuralfitter.models.SigmaMUNet.parse(param)
    model = decode.utils.model_io.LoadSaveModel(
        model, input_file=args.model_path, output_file=None).load_init(args.device)

    """Load the frame"""
    if not online:
        frames = decode.utils.frames_io.load_tif(args.frame_path)
    else:
        frames = decode.utils.frames_io.TiffTensor(args.frame_path)

    # load meta
    with open(args.frame_meta_path) as meta:
        meta = yaml.safe_load(meta)

    param = decode.utils.param_io.autofill_dict(meta['Camera'], param.to_dict(), mode_missing='include')
    param = decode.utils.param_io.RecursiveNamespace(**param)

    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.device = 'cpu'

    """Prepare Pre and post-processing"""

    """Fit"""

    """Return"""

