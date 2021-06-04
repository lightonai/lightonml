import torch
import torch.nn as nn
import torch.nn.functional as F


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()


class EncoderDecoder(AE):
    """Autoencoder consisting of two dense layers (encoder, decoder). The decoder weights are the transpose of the
    encoder ones. Backpropagation updates the decoder, in this way the encoder is also updated despite the
    non-differentiable non-linearity. Architecture from Tissier et al. (https://arxiv.org/abs/1803.09065).

    Parameters
    ----------
    input_size: int,
        size of the input
    hidden_size: int,
        size of the hidden layer

    Attributes
    ----------
    proj: nn.Linear,
        encoding-decoding layer
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderDecoder, self).__init__()
        self.proj = nn.Linear(input_size, hidden_size, bias=False)

    def forward(self, input):
        """Returns the reconstructed input or the binary code, depending on self.training.
        Call `.eval()` on the module for the binary code, `.train()` for the reconstruction.

        Parameters
        ----------
        input: torch.Tensor,
            tensor holding the input data

        Returns
        -------
        rec: torch.Tensor float,
            if self.training=True returns the reconstruction of the input
        binary_code: torch.Tensor uint8,
            tensor holding the binary code if self.training=False.
        """
        with torch.no_grad():
            enc = self.proj(input)
            enc = enc > 0
        if self.training:
            rec = torch.mm(enc.float(), self.proj.weight)
            return rec
        else:
            binary_code = enc.type(torch.uint8)
        return binary_code


class LinearAE(AE):
    """Autoencoder consisting of two dense layers (encoder, decoder). The autoencoder learns to produce a binary output
    starting from tanh(beta x)/beta with beta=1 and gradually increasing beta to resemble a step function

    Parameters
    ----------
    input_size: int,
        size of the input
    hidden_size: int,
        size of the hidden layer
    beta: float, default 1.,
        inverse temperature for tanh(beta x)

    Attributes
    ----------
    encoder: nn.Linear,
        encoding layer
    decoder: nn.Linear,
        decoding layer
    beta: float,
        inverse temperature for tanh(beta x)
    """
    def __init__(self, input_size, hidden_size, beta=1.):
        super(LinearAE, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.beta = beta

    def forward(self, input):
        """Returns the reconstructed input or the binary code, depending on self.training.
        Call `.eval()` on the module for the binary code, `.train()` for the reconstruction.

        Parameters
        ----------
        input: torch.Tensor,
            tensor holding the input data

        Returns
        -------
        rec: torch.Tensor float,
            if self.training=True returns the reconstruction of the input
        binary_code: torch.Tensor uint8,
            tensor holding the binary code if self.training=False.
        """
        e = self.encoder(input)
        if self.training:
            binary_code = torch.tanh(self.beta * e) / self.beta
            rec = self.decoder(binary_code)
            return rec
        else:
            binary_code = (self.beta * e) > 0
            binary_code = binary_code.type(torch.uint8)
            return binary_code


class ConvAE(AE):
    """Autoencoder consisting of two convolutional layers (encoder, decoder).

    Parameters
    ----------
    in_ch: int,
        number of input channels
    out_ch: int,
        number of output channels
    kernel_size: int or tuple,
        size of the convolutional filters
    beta: float, default 1.,
        inverse temperature for tanh(beta x)
    stride: int or tuple, optional, default 1,
        stride of the convolution
    padding: int or tuple, optional, default 0,
        zero-padding added to both sides of the input
    padding_mode: str, optional, default 'zeros',
        'zeros' or 'circular'
    dilation: int or tuple, optional, default 1,
        spacing between kernel elements
    groups: int, optional, default 1,
        number of blocked connections from input to output channels
    bias: bool, optional, default ``True``,
        adds a learnable bias to the output.
    flatten: bool, default False,
        whether to return a 2D flattened array (batch_size, x) or a 4D (batch_size, out_ch, out_h, out_w) when encoding

    Attributes
    ----------
    encoder: nn.Conv2d,
        encoding layer
    decoder: nn.TransposeConv2d,
        decoding layer
    beta: float,
        inverse temperature for tanh(beta x)
    flatten: bool, default False,
        whether to return a 2D flattened array (batch_size, x) or a 4D (batch_size, out_ch, out_h, out_w) when encoding
    """
    def __init__(self, in_ch, out_ch, kernel_size, beta, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 padding_mode='zeros', flatten=False):
        super(ConvAE, self).__init__()
        self.encoder = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, dilation=dilation,
                                 groups=groups, bias=bias, padding_mode=padding_mode)
        self.decoder = nn.ConvTranspose2d(out_ch, in_ch, kernel_size, stride=stride, padding=padding,
                                          groups=groups, bias=bias, dilation=dilation, padding_mode=padding_mode)
        self.beta = beta
        self.flatten = flatten

    def forward(self, input):
        """Returns the reconstructed input or the binary code, depending on self.training.
        Call `.eval()` on the module for the binary code, `.train()` for the reconstruction.

        Parameters
        ----------
        input: torch.Tensor,
            tensor holding the input data

        Returns
        -------
        rec: torch.Tensor float,
            if self.training=True returns the reconstruction of the input
        binary_code: torch.Tensor uint8,
            tensor holding the binary code if self.training=False.
        """
        e = self.encoder(input)
        if self.training:
            binary_code = torch.tanh(self.beta * e) / self.beta
            rec = self.decoder(binary_code)
            return rec
        else:
            binary_code = (self.beta * e) > 0
            binary_code = binary_code.type(torch.uint8)
            if self.flatten:
                binary_code = binary_code.reshape(binary_code.shape[0], -1)
            return binary_code


def train(model, dataloader, optimizer, criterion=F.mse_loss, epochs=10, beta_interval=5, device=None, verbose=True):
    """Utility function to train autoencoders quickly.

    Parameters
    ----------
    model: nn.Module,
        autoencoder to trained
    dataloader: torch.utils.data.Dataloader,
        loader for the training dataset of the autoencoder
    optimizer: torch.optim.Optimizer,
        optimizer used to perform the training
    criterion: callable, default torch.nn.functional.mse_loss
        loss function for training
    epochs: int, default 10,
        number of epochs of training
    beta_interval: int, default 5,
        interval in epochs for beta increase by factor 10
    device: str, 'cpu' or 'cuda:{idx}'
        device used to perform the training.
    verbose: bool, default True,
        whether to print info on the training

    Returns
    -------
    model: nn.Module,
        trained model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not model.training:
        print('Setting model to training mode')
        model.train()
        print('Model.training = {}'.format(model.training))
    model = model.to(device)
    for e in range(epochs):
        tot_loss = 0.
        i = 0  # mute warning in later verbose
        for i, data in enumerate(dataloader):
            if isinstance(data, list):
                data, _ = data

            optimizer.zero_grad()
            data = data.to(device)
            out = model(data)
            loss = criterion(out, data)
            tot_loss += loss.detach().item()

            if isinstance(model, EncoderDecoder):
                # if EncoderDecoder, add regularization term
                W = model.proj.weight
                eye = torch.eye(W.shape[1]).to(device)
                reg_loss = 0.5 * (torch.matmul(W.t(), W) - eye).pow(2).sum()
                loss = loss + reg_loss

            loss.backward()
            optimizer.step()

        if e != 0 and e % beta_interval == 0 and not isinstance(model, EncoderDecoder):
            # increase beta every `beta_interval` epochs if the encoder is not EncoderDecoder
            model.beta = model.beta * 10

        if verbose:
            print("Epoch: [{}/{}], Training Loss: {}".format(e+1, epochs, tot_loss / (i + 1)))

    return model
