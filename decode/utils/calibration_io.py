import scipy.io as sio
import torch

import decode.simulation.psf_kernel as psf_kernel


class SMAPSplineCoefficient:
    """
    Wrapper class as an interface for MATLAB Spline calibration data.

    Attributes
    ----------
        calib_file : str
            path to SMAP calibration .mat file

        calib_mat : dict
            dictionary representation of matlab struct, keys -> var names, vals -> matrices

        coeff : torch.Tensor
            Tensor representation of cubic b-spline coefficients that define psf shape

        ref0 : tuple of float
            origin of coordinate system referenced by cubic spline coefficients

        dz : float
            z-axis anisotropy factor

        spline_roi_shape : tuple of int
            shape of cubic b-spline coefficient grid

    Methods
    -------
        init_spline(
            xextent: tuple of float,
            yextent: tuple of float,
            img_shape: tuple of int,
            device: str,
            **kwargs,
        ) -> decode.simulation.psf_kernel.CubicSplinePSF
            Construct a CubicSplinePSF object based on stored calibration parameters (see attributes) and
            given inputs. Additional keyword arguments passed to CubicSplinePSF constructor.
    """
    def __init__(self, calib_file):
        """
        Loads a calibration file from SMAP and the relevant meta information
        Args:
            file:
        """
        self.calib_file = calib_file
        self.calib_mat = sio.loadmat(self.calib_file, struct_as_record=False, squeeze_me=True)['SXY']

        self.coeff = torch.from_numpy(self.calib_mat.cspline.coeff)
        self.ref0 = (self.calib_mat.cspline.x0 - 1, self.calib_mat.cspline.x0 - 1, self.calib_mat.cspline.z0)
        self.dz = self.calib_mat.cspline.dz
        self.spline_roi_shape = self.coeff.shape[:3]

    def init_spline(self, xextent, yextent, img_shape, device='cuda:0' if torch.cuda.is_available() else 'cpu', **kwargs):
        """
        Initializes the CubicSpline function

        Args:
            xextent:
            yextent:
            img_shape:
            device: on which device to simulate

        Returns:

        """
        psf = psf_kernel.CubicSplinePSF(xextent=xextent, yextent=yextent, img_shape=img_shape, ref0=self.ref0,
                                        coeff=self.coeff, vx_size=(1., 1., self.dz), device=device, **kwargs)

        return psf
