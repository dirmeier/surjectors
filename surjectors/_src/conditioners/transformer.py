import dataclasses
from typing import Callable, Optional

import haiku as hk
import jax


@dataclasses.dataclass
class _EncoderLayer(hk.Module):
    num_heads: int
    num_layers: int
    key_size: int
    dropout_rate: float
    widening_factor: int = 4
    initializer: Callable = hk.initializers.TruncatedNormal(stddev=0.01)
    name: Optional[str] = None

    def __call__(self, inputs, *, is_training):
        dropout_rate = self.dropout_rate if is_training else 0.0
        # causal_mask = np.tril(np.ones((1, 1, seq_len, seq_len)))
        causal_mask = None
        model_size = self.key_size * self.num_heads

        h = inputs
        for _ in range(self.num_layers):
            attn_block = hk.MultiHeadAttention(
                num_heads=self.num_heads,
                key_size=self.key_size,
                model_size=model_size,
                w_init=self.initializer,
            )
            h_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True
            )(h)
            h_attn = attn_block(h_norm, h_norm, h_norm, mask=causal_mask)
            h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn

            mlp = hk.nets.MLP(
                [self.widening_factor * model_size, model_size],
                w_init=self.initializer,
                activation=jax.nn.gelu,
            )
            h_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True
            )(h)
            h_dense = mlp(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(h)


@dataclasses.dataclass
class _AutoregressiveTransformerEncoder(hk.Module):
    linear: hk.Linear
    transformer: _EncoderLayer
    output_size: int
    name: Optional[str] = None

    def __call__(self, inputs, *, is_training=True):
        h = self.linear(inputs)
        h = self.transformer(h, is_training=is_training)
        return hk.Linear(self.output_size)(h)


# pylint: disable=too-many-arguments
def transformer_conditioner(
    output_size,
    num_heads=2,
    num_layers=2,
    key_size=32,
    dropout_rate=0.1,
    widening_factor=4,
):
    """
    Create a conditioner network based on a transformer

    Returns
    -------
    hk.Module
        a transformable haiku neural network module
    """

    linear = hk.Linear(key_size * num_heads)
    encoder = _EncoderLayer(
        num_heads, num_layers, key_size, dropout_rate, widening_factor
    )
    transformer = _AutoregressiveTransformerEncoder(
        linear, encoder, output_size
    )
    return transformer
