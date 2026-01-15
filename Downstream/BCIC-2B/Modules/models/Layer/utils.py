import torch

def rotate_half(x):
    # x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x = x.reshape((*x.shape[:-1], x.shape[-1] // 2, 2))
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    # return rearrange(x, '... d r -> ... (d r)')
    return x.flatten(-2)

def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    """
    Apply rotary positional embeddings to a tensor.

    The rotary embedding rotates each dimension of the input tensor `t`
    based on the corresponding frequency in `freqs`, using cosine and sine functions.
    This rotation helps the model preserve positional information.

    Parameters:
    - freqs (Tensor): The frequency embeddings (sine and cosine values precomputed).
    - t (Tensor): The input tensor to which the rotary embeddings are applied.
    - start_index (int): Start index where the rotation will begin within the tensor `t`.
    - scale (float): Scaling factor for the rotation applied.

    Returns:
    - Tensor: The tensor `t` after rotary positional embeddings have been applied.
    """

    freqs = freqs.to(t.device)

    rot_dim = freqs.shape[-1]

    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[
        -1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t_middle, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

    # Apply rotary embeddings to the middle segment.
    t_rotated_middle = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    return torch.cat((t_left, t_rotated_middle, t_right), dim=-1)