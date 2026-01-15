import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1, patch_size),
                              stride=(1, patch_size if patch_stride is None else patch_stride))

    def forward(self, x):
        # x: B,C,T
        x = x.unsqueeze(1)  # B, 1, C, T
        x = self.proj(x).transpose(1, 3)  # B, T, C, D
        return x


class ConvLayer(nn.Module):
    """
    in: [B,N,C,T1]
    out:[B,N,C,D]
    """
    def __init__(self, embed_dim, kernel_sizes=3):
        super(ConvLayer, self).__init__()
        self.embed_dim = embed_dim
        self.numConv = 1
        self.out_channels = 64

        # self.conv1_1 = nn.Conv1d(in_channels=1,out_channels=self.out_channels,kernel_size=kernel_sizes[0],padding=kernel_sizes[0] // 2)
        self.conv1_3 = nn.Conv1d(in_channels=1,out_channels=self.out_channels,kernel_size=kernel_sizes,padding=kernel_sizes // 2)
        self.Linear = nn.Linear(self.numConv * self.out_channels,self.embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        B, N, C, T = x.shape
        x = x.view(-1, 1, T)    # [B*N*C, 1, T]

        x = self.activation(self.conv1_3(x))  # [B*N*C, self.out_channels, T]

        x = x.permute(0, 2, 1).contiguous()
        x = self.activation(self.Linear(x))
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1).squeeze(-1)
        return x.view(B, N, C, self.embed_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000, learned_freq=False, interpolate_factor=1.0):
        """
        Rotary Positional Embedding module to encode sequential information into embeddings.

        Parameters:
            dim (int): Dimension of the frequency embedding.
            theta (float): A hyperparameter that influences the scale of the frequency embedding.
            learned_freq (bool): Whether the frequencies are learnable parameters.
            interpolate_factor (float): Scaling factor for interpolated positional encoding.
        """
        super().__init__()
        assert interpolate_factor >= 1.0, "Interpolate factor must be >= 1.0"

        # Initialize frequency parameters
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)),
            requires_grad=learned_freq)

        self.interpolate_factor = interpolate_factor
        self.cache = {}

    def prepare_freqs(self, num_patches, device='cuda', dtype=torch.float32, offset=0):
        """
        Prepares the frequency embeddings for the given number of patches.

        Parameters:
            num_patches (tuple): Tuple specifying the dimensions (C, N) where
                                 C is the channels and N is the number of positions.
            device (str): Device to store the frequencies on (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for the frequencies.
            offset (float): Offset added to position indexes before scaling.

        Returns:
            torch.Tensor: Prepared frequency embeddings with shape [C * N, dim].
        """
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'

        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Generate sequence positions and apply offset and scale
        seq_pos = torch.arange(N, device=device, dtype=dtype).repeat_interleave(repeats=C)
        seq_pos = (seq_pos + offset) / self.interpolate_factor

        # Compute outer product of positions and frequencies, then expand along the last dimension
        freqs_scaled = torch.outer(seq_pos.type(self.freqs.dtype), self.freqs).repeat_interleave(repeats=2, dim=-1)

        # Cache and return the computed frequencies
        self.cache[cache_key] = freqs_scaled
        return freqs_scaled


