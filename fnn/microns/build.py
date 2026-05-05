import numpy as np
import torch
from fnn.model.feedforwards import InputDense
from fnn.model.recurrents import CvtLstm
from fnn.model.cores import FeedforwardRecurrent
from fnn.model.monitors import Plane
from fnn.model.pixels import StaticPower, SigmoidPower
from fnn.model.retinas import Angular
from fnn.model.perspectives import MlpMonitorRetina
from fnn.model.modulations import MlpLstm
from fnn.model.positions import Gaussian
from fnn.model.bounds import Tanh
from fnn.model.features import Vanilla
from fnn.model.readouts import PositionFeature
from fnn.model.reductions import Mean
from fnn.model.units import Poisson
from fnn.model.networks import Visual, Visual_t, Visual_t_pool


def network(units):
    """
    Parameters
    ----------
    units : int
        number of units

    Returns
    -------
    fnn.model.networks.Visual
        visual neural network
    """
    feedforward = InputDense(
        input_spatial=6,
        input_stride=2,
        block_channels=[32, 64, 128],
        block_groups=[1, 2, 4],
        block_layers=[2, 2, 2],
        block_temporals=[3, 3, 3],
        block_spatials=[3, 3, 3],
        block_pools=[2, 2, 1],
        out_channels=128,
        nonlinear="gelu",
    )
    recurrent = CvtLstm(
        in_channels=256,
        out_channels=128,
        hidden_channels=256,
        common_channels=512,
        groups=8,
        spatial=3,
    )
    core = FeedforwardRecurrent(
        feedforward=feedforward,
        recurrent=recurrent,
    )
    perspective = MlpMonitorRetina(
        mlp_features=16,
        mlp_layers=3,
        mlp_nonlinear="gelu",
        height=128,
        width=192,
        monitor=Plane(),
        monitor_pixel=StaticPower(power=1.7),
        retina=Angular(degrees=75),
        retina_pixel=SigmoidPower(),
    )
    modulation = MlpLstm(
        mlp_features=16,
        mlp_layers=1,
        mlp_nonlinear="gelu",
        lstm_features=16,
    )
    readout = PositionFeature(
        position=Gaussian(),
        bound=Tanh(),
        feature=Vanilla(),
    )
    network = Visual(
        core=core,
        perspective=perspective,
        modulation=modulation,
        readout=readout,
        reduce=Mean(),
        unit=Poisson(),
    )
    network._init(
        stimuli=1,
        perspectives=2,
        modulations=2,
        streams=4,
        units=units,
    )
    return network


def network_s(units):
    """
    Parameters
    ----------
    units : int
        number of units

    Returns
    -------
    fnn.model.networks.Visual
        visual neural network with ~1/4 core parameters (all channel dims halved)
    """
    feedforward = InputDense(
        input_spatial=6,
        input_stride=2,
        block_channels=[16, 32, 64],
        block_groups=[1, 2, 4],
        block_layers=[2, 2, 2],
        block_temporals=[3, 3, 3],
        block_spatials=[3, 3, 3],
        block_pools=[2, 2, 1],
        out_channels=64,
        nonlinear="gelu",
    )
    recurrent = CvtLstm(
        in_channels=128,
        out_channels=64,
        hidden_channels=128,
        common_channels=256,
        groups=8,
        spatial=3,
    )
    core = FeedforwardRecurrent(
        feedforward=feedforward,
        recurrent=recurrent,
    )
    perspective = MlpMonitorRetina(
        mlp_features=16,
        mlp_layers=3,
        mlp_nonlinear="gelu",
        height=128,
        width=192,
        monitor=Plane(),
        monitor_pixel=StaticPower(power=1.7),
        retina=Angular(degrees=75),
        retina_pixel=SigmoidPower(),
    )
    modulation = MlpLstm(
        mlp_features=16,
        mlp_layers=1,
        mlp_nonlinear="gelu",
        lstm_features=16,
    )
    readout = PositionFeature(
        position=Gaussian(),
        bound=Tanh(),
        feature=Vanilla(),
    )
    network = Visual(
        core=core,
        perspective=perspective,
        modulation=modulation,
        readout=readout,
        reduce=Mean(),
        unit=Poisson(),
    )
    network._init(
        stimuli=1,
        perspectives=2,
        modulations=2,
        streams=4,
        units=units,
    )
    return network


def network_t(units, svd_dir='/project/rf/code/fnn/svd', pts_temperature=0.1, pts_n=3):
    """
    Parameters
    ----------
    units : int
        number of units
    svd_dir : str
        directory containing svd_VT.npy and svd_feat_mean.npy
    pts_temperature : float
        PTS scale divisor T (default 0.1)
    pts_n : float
        PTS root exponent n (default 3)

    Returns
    -------
    fnn.model.networks.Visual_t
        visual network identical to network() but with SVD projection + PTS on the
        core output reducing 512 -> 256 channels (128 -> 64 per stream)
    """
    feedforward = InputDense(
        input_spatial=6,
        input_stride=2,
        block_channels=[32, 64, 128],
        block_groups=[1, 2, 4],
        block_layers=[2, 2, 2],
        block_temporals=[3, 3, 3],
        block_spatials=[3, 3, 3],
        block_pools=[2, 2, 1],
        out_channels=128,
        nonlinear="gelu",
    )
    recurrent = CvtLstm(
        in_channels=256,
        out_channels=128,
        hidden_channels=256,
        common_channels=512,
        groups=8,
        spatial=3,
    )
    core = FeedforwardRecurrent(
        feedforward=feedforward,
        recurrent=recurrent,
    )
    perspective = MlpMonitorRetina(
        mlp_features=16,
        mlp_layers=3,
        mlp_nonlinear="gelu",
        height=128,
        width=192,
        monitor=Plane(),
        monitor_pixel=StaticPower(power=1.7),
        retina=Angular(degrees=75),
        retina_pixel=SigmoidPower(),
    )
    modulation = MlpLstm(
        mlp_features=16,
        mlp_layers=1,
        mlp_nonlinear="gelu",
        lstm_features=16,
    )
    readout = PositionFeature(
        position=Gaussian(),
        bound=Tanh(),
        feature=Vanilla(),
    )
    svd_vt   = torch.from_numpy(np.load(f'{svd_dir}/svd_VT.npy')).float()       # [512, 512]
    svd_mean = torch.from_numpy(np.load(f'{svd_dir}/svd_feat_mean.npy')).float() # [512]

    net = Visual_t(
        core=core,
        perspective=perspective,
        modulation=modulation,
        readout=readout,
        reduce=Mean(),
        unit=Poisson(),
        svd_vt=svd_vt,
        svd_mean=svd_mean,
        top_k_total=256,
        top_k_per_stream=64,
        pts_temperature=pts_temperature,
        pts_n=pts_n,
    )
    net._init(
        stimuli=1,
        perspectives=2,
        modulations=2,
        streams=4,
        units=units,
    )
    return net


def network_t_pool(units, svd_dir='/project/rf/code/fnn/svd', pts_temperature=0.1, pts_n=3):
    """
    Same as network_t but SVD projection and PTS are applied to spatially-pooled
    core features. _forward_core returns [N, K] instead of [N, K, H, W].

    Parameters
    ----------
    units : int
    svd_dir : str
    pts_temperature : float
    pts_n : float

    Returns
    -------
    fnn.model.networks.Visual_t_pool
    """
    feedforward = InputDense(
        input_spatial=6,
        input_stride=2,
        block_channels=[32, 64, 128],
        block_groups=[1, 2, 4],
        block_layers=[2, 2, 2],
        block_temporals=[3, 3, 3],
        block_spatials=[3, 3, 3],
        block_pools=[2, 2, 1],
        out_channels=128,
        nonlinear="gelu",
    )
    recurrent = CvtLstm(
        in_channels=256,
        out_channels=128,
        hidden_channels=256,
        common_channels=512,
        groups=8,
        spatial=3,
    )
    core = FeedforwardRecurrent(
        feedforward=feedforward,
        recurrent=recurrent,
    )
    perspective = MlpMonitorRetina(
        mlp_features=16,
        mlp_layers=3,
        mlp_nonlinear="gelu",
        height=128,
        width=192,
        monitor=Plane(),
        monitor_pixel=StaticPower(power=1.7),
        retina=Angular(degrees=75),
        retina_pixel=SigmoidPower(),
    )
    modulation = MlpLstm(
        mlp_features=16,
        mlp_layers=1,
        mlp_nonlinear="gelu",
        lstm_features=16,
    )
    readout = PositionFeature(
        position=Gaussian(),
        bound=Tanh(),
        feature=Vanilla(),
    )
    svd_vt   = torch.from_numpy(np.load(f'{svd_dir}/svd_VT.npy')).float()
    svd_mean = torch.from_numpy(np.load(f'{svd_dir}/svd_feat_mean.npy')).float()

    net = Visual_t_pool(
        core=core,
        perspective=perspective,
        modulation=modulation,
        readout=readout,
        reduce=Mean(),
        unit=Poisson(),
        svd_vt=svd_vt,
        svd_mean=svd_mean,
        top_k_total=256,
        top_k_per_stream=64,
        pts_temperature=pts_temperature,
        pts_n=pts_n,
    )
    net._init(
        stimuli=1,
        perspectives=2,
        modulations=2,
        streams=4,
        units=units,
    )
    return net
