# ignore_header_test
# ruff: noqa: E402
"""
Pix2PixUnet model. This code was modified from, https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

The following license is provided from their source,

Copyright (C) 2019 NVIDIA Corporation. Ting-Chun Wang, Ming-Yu Liu, Jun-Yan Zhu.
BSD License. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL
DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


--------------------------- LICENSE FOR pytorch-CycleGAN-and-pix2pix ----------------
Copyright (c) 2017, Jun-Yan Zhu and Taesung Park
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import functools
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

import physicsnemo  # noqa: F401 for docs

from ..meta import ModelMetaData
from ..module import Module

Tensor = torch.Tensor


def get_scheduler(
    optimizer,
    lr_policy="linear",
    epoch_count=1,
    n_epochs=100,
    n_epochs_decay=100,
    lr_decay_iters=50,
):
    """Return a learning rate scheduler.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer of the network
    lr_policy : str, optional
        The name of learning rate policy: linear | step | plateau | cosine, by default 'linear'
    epoch_count : int, optional
        The starting epoch count, by default 1
    n_epochs : int, optional
        Number of epochs with the initial learning rate, by default 100
    n_epochs_decay : int, optional
        Number of epochs to linearly decay learning rate to zero, by default 100
    lr_decay_iters : int, optional
        Multiply by a gamma every lr_decay_iters iterations (for step policy), by default 50

    Returns
    -------
    torch.optim.lr_scheduler
        The learning rate scheduler

    Notes
    -----
    For 'linear', we keep the same learning rate for the first <n_epochs> epochs
    and linearly decay the rate to zero over the next <n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + epoch_count - n_epochs) / float(
                n_epochs_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError(
            f"learning rate policy [{lr_policy}] is not implemented"
        )
    return scheduler


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters
    ----------
    net : nn.Module
        Network to be initialized
    init_type : str, optional
        The name of an initialization method: normal | xavier | kaiming | orthogonal, by default "normal"
    init_gain : float, optional
        Scaling factor for normal, xavier and orthogonal initialization, by default 0.02
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


@dataclass
class MetaData(ModelMetaData):
    name: str = "Pix2PixUnet"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False  # Reflect padding not supported in bfloat16
    amp_gpu: bool = True
    # Inference
    onnx: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = True
    auto_grad: bool = True


class GANLoss(nn.Module):
    """Define different GAN objectives.

    Parameters
    ----------
    gan_mode : str
        The type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
    target_real_label : float, optional
        Label for a real image, by default 1.0
    target_fake_label : float, optional
        Label for a fake image, by default 0.0

    Notes
    -----
    Do not use sigmoid as the last layer of Discriminator.
    LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters
        ----------
        prediction : tensor
            Typically the prediction from a discriminator
        target_is_real : bool
            If the ground truth label is for real images or fake images

        Returns
        -------
        torch.Tensor
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and ground truth labels.

        Parameters
        ----------
        prediction : tensor
            Typically the prediction output from a discriminator
        target_is_real : bool
            If the ground truth label is for real images or fake images

        Returns
        -------
        torch.Tensor
            The calculated loss.
        """

        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator.

    Parameters
    ----------
    input_nc : int
        The number of channels in input images
    ndf : int, optional
        The number of filters in the last conv layer, by default 64
    n_layers : int, optional
        The number of conv layers in the discriminator, by default 3
    norm_layer : nn.Module, optional
        Normalization layer, by default nn.BatchNorm2d
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN).

    Parameters
    ----------
    input_nc : int
        The number of channels in input images
    ndf : int, optional
        The number of filters in the last conv layer, by default 64
    norm_layer : nn.Module, optional
        Normalization layer, by default nn.BatchNorm2d
    """

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        super(PixelDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias),
        ]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class Pix2PixUnet(Module):
    """Convolutional encoder-decoder based on pix2pix generator models using Unet.

    Note
    ----
    The pix2pix with Unet architecture only supports 2D field.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels: Union[int, Any], optional
        Number of output channels
    n_downsampling : int
        Number of downsampling in UNet
    filter_size : int, optional
        Number of filters in last convolution layer, by default 64
    norm_layer : optional
        Normalization layer, by default nn.BatchNorm2d
    use_dropout : bool, optional
        Use dropout layers, by default False
    gpu_ids : List, optional
        List of GPU IDs to use, by default []
    learning_rate : float, optional
        Learning rate for the optimizer, by default 0.0001
    lr_policy : str, optional
        Learning rate policy, by default 'linear'
    n_epochs : int, optional
        Number of epochs to train, by default 100
    n_epochs_decay : int, optional
        Number of epochs to decay learning rate, by default 100
    save_dir : str, optional
        Directory to save the model, by default './results'
    is_train : bool, optional
        Whether to train the model, by default False

    Note
    ----
    Reference:  Isola, Phillip, et al. “Image-To-Image translation with conditional
    adversarial networks” Conference on Computer Vision and Pattern Recognition, 2017.
    https://arxiv.org/abs/1611.07004

    Reference: Wang, Ting-Chun, et al. “High-Resolution image synthesis and semantic
    manipulation with conditional GANs” Conference on Computer Vision and Pattern
    Recognition, 2018. https://arxiv.org/abs/1711.11585

    Note
    ----
    Based on the implementation: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_downsampling: int,
        filter_size: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
        gpu_ids: List = [],
        learning_rate: float = 0.0001,
        lr_policy: str = "linear",
        n_epochs: int = 100,
        n_epochs_decay: int = 100,
        save_dir: str = "./results",
        is_train: bool = False,
    ):
        if not (filter_size > 0 and n_downsampling >= 0):
            raise ValueError("Invalid arch params")
        super().__init__(meta=MetaData())

        # device
        self.gpu_ids = gpu_ids
        self.model_device = (
            torch.device("cuda:{}".format(self.gpu_ids[0]))
            if self.gpu_ids
            else torch.device("cpu")
        )

        # model names
        self.model_names = ["G"]

        # define generator network
        self.netG = self.define_G(
            in_channels,
            out_channels,
            n_downsampling,
            filter_size,
            norm_layer,
            use_dropout,
        )

        # training flag
        self.isTrain = is_train
        if self.isTrain:
            self.model_names.append("D")

            # set parameters explicitly set in the constructor
            self.learning_rate = learning_rate
            self.lr_policy = lr_policy
            self.n_epochs = n_epochs
            self.n_epochs_decay = n_epochs_decay
            self.save_dir = save_dir

            # set parameters not explicitly set in the constructor
            self.epoch_count = 1
            self.lr_decay_iters = 50
            self.gan_mode = "vanilla"  # ['vanilla', 'lsgan', 'wgangp']
            self.lambda_L1 = 100.0
            self.netD_type = "basic"  # ['basic', 'n_layers', 'pixel']
            self.n_layers_D = 3
            self.beta1 = 0.5

            # define discriminator
            self.netD = self.define_D(
                in_channels + out_channels,
                filter_size,
                self.netD_type,
                self.n_layers_D,
                norm_layer,
            )

            # define loss functions
            self.criterionGAN = GANLoss(self.gan_mode).to(self.model_device)
            self.criterionL1 = torch.nn.L1Loss()

            # specify the training losses to print out
            self.loss_names = ["G_GAN", "G_L1", "D_real", "D_fake"]

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)
            )
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999)
            )
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        else:
            torch.manual_seed(0)  # avoid run-to-run variation

    def define_G(
        self,
        in_channels,
        out_channels,
        n_downsampling,
        filter_size,
        norm_layer,
        use_dropout,
    ):
        """Create generator network.

        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        n_downsampling : int
            Number of downsampling layers in UNet
        filter_size : int
            Number of filters in the last conv layer
        norm_layer : nn.Module
            Normalization layer
        use_dropout : bool
            Whether to use dropout layers

        Returns
        -------
        nn.Module
            The initialized generator network
        """

        net = UnetGenerator(
            in_channels,
            out_channels,
            n_downsampling,
            filter_size,
            norm_layer,
            use_dropout,
        )
        if len(self.gpu_ids) > 0:
            net.to(self.gpu_ids[0])
            net = torch.nn.DataParallel(net, self.gpu_ids)  # multi-GPUs
        init_weights(net)
        return net

    def define_D(self, input_nc, ndf, netD_type, n_layers_D, norm_layer):
        """Create discriminator network.

        Parameters
        ----------
        input_nc : int
            Number of input channels
        ndf : int
            Number of filters in the first conv layer
        netD_type : str
            Type of discriminator: basic | n_layers | pixel
        n_layers_D : int
            Number of conv layers in the discriminator
        norm_layer : nn.Module
            Normalization layer

        Returns
        -------
        nn.Module
            The initialized discriminator network

        Raises
        ------
        NotImplementedError
            If the discriminator type is not recognized
        """

        if netD_type == "basic":
            net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        elif netD_type == "n_layers":
            net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
        elif netD_type == "pixel":
            net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer)
        else:
            raise NotImplementedError(
                f"Discriminator model name {netD_type} is not recognized"
            )

        if len(self.gpu_ids) > 0:
            net.to(self.gpu_ids[0])
            net = torch.nn.DataParallel(net, self.gpu_ids)
        init_weights(net)
        return net

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0) -> None:
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "running_mean" or key == "running_var"
            ):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (
                key == "num_batches_tracked"
            ):
                state_dict.pop(".".join(keys))
        else:
            self.__patch_instance_norm_state_dict(
                state_dict, getattr(module, key), keys, i + 1
            )

    def load_networks(self, model_path: str) -> None:
        """Load all the networks from the disk.

        Parameters
        ----------
        model_path : str
            Path to the saved model
        """

        net = self.netG
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        state_dict = torch.load(model_path, map_location=str(self.model_device))
        if hasattr(state_dict, "_metadata"):
            del state_dict._metadata

        # patch InstanceNorm checkpoints prior to 0.4
        for key in list(
            state_dict.keys()
        ):  # need to copy keys here because we mutate in loop
            self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
        net.load_state_dict(state_dict)

    def setup(self) -> None:
        """Initialize the networks and schedulers.

        Should be called at the beginning of training.
        """

        if self.isTrain:
            self.schedulers = [
                get_scheduler(
                    optimizer,
                    self.lr_policy,
                    self.epoch_count,
                    self.n_epochs,
                    self.n_epochs_decay,
                    self.lr_decay_iters,
                )
                for optimizer in self.optimizers
            ]

    def get_learning_rate(self) -> float:
        """Get current learning rate.

        Returns
        -------
        float
            Current learning rate
        """

        return self.optimizers[0].param_groups[0]["lr"]

    def update_learning_rate(self) -> None:
        """Update learning rates for all the networks.

        This should be called once per epoch.
        """

        for scheduler in self.schedulers:
            if self.lr_policy == "plateau":
                scheduler.step(self.metric)
            else:
                scheduler.step()

    def set_input(self, input_dict: Dict[str, Any]) -> None:
        """Unpack input data from the dataloader.

        Parameters
        ----------
        input_dict : Dict[str, Any] or torch.Tensor
            Either a dictionary containing the data or directly a tensor
        """

        if isinstance(input_dict, dict):
            AtoB = True  # assume AtoB by default
            if "direction" in input_dict and input_dict["direction"] == "BtoA":
                AtoB = False

            self.real_A = input_dict["A" if AtoB else "B"].to(self.model_device)
            self.real_B = input_dict["B" if AtoB else "A"].to(self.model_device)
            self.image_paths = input_dict["A_paths" if AtoB else "B_paths"]
        else:
            # If it's not a dict, assume it's a tensor to be used as real_A
            self.real_A = input_dict.to(self.model_device)

    def test(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            return self.netG(input)

    def forward(self) -> None:
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self) -> None:
        """Calculate GAN loss for the discriminator.

        This will set self.loss_D_fake, self.loss_D_real, and self.loss_D
        """

        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self) -> None:
        """Calculate GAN and L1 loss for the generator.

        This will set self.loss_G_GAN, self.loss_G_L1, and self.loss_G
        """

        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.lambda_L1

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self) -> None:
        """Calculate losses, gradients, and update network weights.

        This is called for each training iteration.
        """

        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requires_grad=False for all the networks to avoid unnecessary computations.

        Parameters
        ----------
        nets : nn.Module or List[nn.Module]
            A list of networks or a single network
        requires_grad : bool, optional
            Whether the networks require gradients or not, by default False
        """

        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_losses(self) -> Dict[str, float]:
        """Return current losses.

        Returns
        -------
        Dict[str, float]
            Dictionary of loss names and values
        """

        return {name: float(getattr(self, "loss_" + name)) for name in self.loss_names}

    def save_networks(self, epoch: int) -> None:
        """Save all the networks to the disk.

        Parameters
        ----------
        epoch : int
            Current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = "%s_net_%s.pth" % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, "net" + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)


class UnetGenerator(nn.Module):
    """Create a Unet-based generator.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    n_downsampling : int
        Number of downsampling in Unet
    filter_size : int, optional
        Number of filters in last convolution layer, by default 64
    norm_layer : optional
        Normalization layer, by default nn.BatchNorm2d
    use_dropout : bool, optional
        Use dropout layers, by default False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_downsampling: int,
        filter_size: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            filter_size * 8,
            filter_size * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(
            n_downsampling - 5
        ):  # add intermediate layers with filter_size * 8 filters
            unet_block = UnetSkipConnectionBlock(
                filter_size * 8,
                filter_size * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )

        # gradually reduce the number of filters from filter_size * 8 to filter_size
        unet_block = UnetSkipConnectionBlock(
            filter_size * 4,
            filter_size * 8,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            filter_size * 2,
            filter_size * 4,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )
        unet_block = UnetSkipConnectionBlock(
            filter_size,
            filter_size * 2,
            input_nc=None,
            submodule=unet_block,
            norm_layer=norm_layer,
        )

        self.model = UnetSkipConnectionBlock(
            out_channels,
            filter_size,
            input_nc=in_channels,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
        )  # add the outermost layer

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """A Unet submodule with skip connections block

    Parameters
    ----------
    outer_nc : int
        Number of filters in the outer conv layer.
    inner_nc : int
        Number of filters in the inner conv layer.
    input_nc: int, optional
        Number of channels in input images/features, by default None, meaning same as outer_nc
    submodule : UnetSkipConnectionBlock, optional
        Previously defined submodules, by default None
    outermost : bool, optional
        if this module is the outermost module, by default False
    innermost : bool, optional
        if this module is the innermost module, by default False
    norm_layer: optional
        normalization layer, by default nn.BatchNorm2d
    use_dropout : bool, optional
        if use dropout layers, by default False
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int = None,
        submodule: nn.Module = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1
            )
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)
