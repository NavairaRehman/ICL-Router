import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import PreTrainedModel, PretrainedConfig


class ProjectorConfig(PretrainedConfig):
    def __init__(self, in_features=4096, out_features=None, expansion_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.expansion_ratio = expansion_ratio

class QFormerProjectorConfig(PretrainedConfig):
    def __init__(
        self,
        in_features=768,
        out_features=None,
        expansion_ratio=4,
        num_attention_heads=12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.expansion_ratio = expansion_ratio
        self.num_attention_heads = num_attention_heads

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
    
class QFormerProjector(PreTrainedModel):

    def __init__(self, config: QFormerProjectorConfig, dtype=torch.float32):
        super().__init__(config)
        self.config = config

        self.in_features = config.in_features
        self.num_attention_heads = config.num_attention_heads
        self.expansion_ratio = config.expansion_ratio
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.in_features,
            num_heads=self.num_attention_heads,
            batch_first=True, 
            bias=True
        )

        self.fc1 = nn.Linear(self.in_features, self.in_features * self.expansion_ratio, bias=True)
        self.act_fn = nn.GELU()
        self.fc2 = nn.Linear(self.in_features * self.expansion_ratio, self.in_features, bias=True)

        self.norm1 = LlamaRMSNorm(self.in_features)
        self.norm2 = LlamaRMSNorm(self.in_features)

    def forward(self, hidden_states, attention_mask=None, return_attn_weights=False):

        residual = hidden_states
        normed_x = self.norm1(hidden_states)  
        attn_output, attn_weights = self.self_attn(
            normed_x, normed_x, normed_x,  
            attn_mask=attention_mask
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        normed_x = self.norm2(hidden_states)
        ffn_output = self.fc2(self.act_fn(self.fc1(normed_x)))
        hidden_states = residual + ffn_output

        if return_attn_weights:
            return hidden_states, attn_weights
        else:
            return hidden_states


class MLPProjector(PreTrainedModel):
    def __init__(self, config: ProjectorConfig, dtype=torch.float32):
        super().__init__(config)
        in_features, out_features, expansion_ratio = config.in_features, config.out_features, config.expansion_ratio
        self.config = config
        if out_features is None:
            out_features = in_features
        mlp_bias = False
        self.input_proj = nn.Linear(in_features, out_features, bias=mlp_bias)
        self.layer_norm = LlamaRMSNorm(in_features)
        self.gate_proj = nn.Linear(out_features, out_features*expansion_ratio, bias=mlp_bias)
        self.up_proj = nn.Linear(out_features, out_features*expansion_ratio, bias=mlp_bias)
        self.down_proj = nn.Linear(out_features*expansion_ratio, out_features, bias=mlp_bias)
        self.act_fn = nn.SiLU()

    def forward(self, input_x):
        x = self.layer_norm(input_x)
        x = self.input_proj(x)
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj + x


class LinearProjector(PreTrainedModel):
    def __init__(self, config: ProjectorConfig, dtype=torch.float32, **kwargs):
        super().__init__(config)
        in_features, out_features = config.in_features, config.out_features
        self.config = config
        if out_features is None:
            out_features = in_features
        self.linear = nn.Linear(in_features, out_features, dtype=dtype, bias=False)

    def forward(self, x):
        return self.linear(x)
