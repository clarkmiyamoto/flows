import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Two 3×3 convolutions + ReLU, preserving spatial dims (padding=1).
    """
    def __init__(self, in_ch, out_ch, num_dims):
        super().__init__()
        Conv = getattr(nn, f'Conv{num_dims}d')
        self.conv1 = Conv(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return x


class DownBlock(nn.Module):
    """
    ConvBlock followed by 2× downsampling (MaxPool).
    Returns (skip_feature, pooled_feature).
    """
    def __init__(self, in_ch, out_ch, num_dims):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, num_dims)
        Pool = getattr(nn, f'MaxPool{num_dims}d')
        self.pool = Pool(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return skip, down


class UpBlock(nn.Module):
    """
    2× upsampling (ConvTranspose) + ConvBlock.
    If skip_ch>0, concatenate encoder skip before ConvBlock.
    """
    def __init__(self, in_ch, out_ch, skip_ch, num_dims):
        super().__init__()
        ConvTranspose = getattr(nn, f'ConvTranspose{num_dims}d')
        self.upconv = ConvTranspose(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch, num_dims)

    def forward(self, x, skip=None):
        x = self.upconv(x)
        if skip is not None:
            # concatenate along channel axis
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetND(nn.Module):
    """
    Generic N-dimensional U-Net:

      - in_channels → encoder_channels[0] → … → encoder_channels[-1]
      - optional bottleneck
      - decoder_channels[0] → … → decoder_channels[-1] → out_channels

    encoder_channels and decoder_channels can be of different lengths or values.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        encoder_channels: list[int],
        decoder_channels: list[int],
        num_dims: int = 2,
        use_skip: bool = True
    ):
        super().__init__()
        assert len(encoder_channels) == len(decoder_channels), \
            "encoder_channels and decoder_channels must have same length"

        self.use_skip = use_skip
        self.num_dims = num_dims

        # --- Encoder ----------------------------
        self.enc_blocks = nn.ModuleList()
        prev_ch = in_channels
        for ch in encoder_channels:
            self.enc_blocks.append(DownBlock(prev_ch, ch, num_dims))
            prev_ch = ch

        # --- Bottleneck (optional) --------------
        # Here we just keep the same channel size
        self.bottleneck = ConvBlock(prev_ch, prev_ch, num_dims)

        # --- Decoder ----------------------------
        self.dec_blocks = nn.ModuleList()
        for idx, ch in enumerate(decoder_channels):
            skip_ch = encoder_channels[-(idx+1)] if use_skip else 0
            self.dec_blocks.append(UpBlock(prev_ch, ch, skip_ch, num_dims))
            prev_ch = ch

        # --- Final 1×1 conv to map to out_channels
        Conv = getattr(nn, f'Conv{num_dims}d')
        self.final_conv = Conv(prev_ch, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder: collect skips
        skips = []
        for down in self.enc_blocks:
            skip, x = down(x)
            if self.use_skip:
                skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: consume skips in reverse
        for idx, up in enumerate(self.dec_blocks):
            skip = skips[-(idx+1)] if self.use_skip else None
            x = up(x, skip)

        # Final output
        return self.final_conv(x)

if __name__ == '__main__':
    '''
    Test the UNetND implementation
    '''
    dim = 1
    encoder_channels = [2 ** i for i in range(4, 8)]
    decoder_channels = list(reversed(encoder_channels))
    model = UNetND(
        in_channels=dim,
        out_channels=dim,
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
        num_dims=2,
        use_skip=True
    )
    x = torch.rand(10, dim, encoder_channels[0], decoder_channels[-1])
    print(x.shape)
    print(model(x).shape)