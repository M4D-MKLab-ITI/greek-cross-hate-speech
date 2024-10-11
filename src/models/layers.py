from typing import Tuple, Optional
import torch 
from torch import nn 
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from transformers.activations import ACT2FN

#Supporting layers - they require their own configs from model.configs
class SwitchDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.wi = nn.Linear(self.config.d_model, self.config.d_ff, bias=False)
        self.wo = nn.Linear(self.config.d_ff, self.config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[self.config.dense_act_fn]
    
    def forward(self, hidden_states):
        hidden_states = self.wi(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)
        hidden_states = self.wo(hidden_states)
        return hidden_states

class SwitchTop1Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.d_model, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def _compute_router_probabilities(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes router probabilities from input hidden states.

        Args:
            hidden_states (`torch.Tensor`):
                (batch_size, sequence_length, hidden_dim) from which router probabilities are computed.
        Returns:
            router_probabilities (`torch.Tensor`):
                Tensor of shape (batch_size, sequence_length, num_experts) corresponding to the probabilities for each
                token and expert. Used for routing tokens to experts.
            router_logits (`torch.Tensor`):
                Logits tensor of shape (batch_size, sequence_length, num_experts) corresponding to raw router logits.
                This is used later for computing router z-loss.
        """
        # float32 is used to ensure stability. See the discussion of "selective precision" in
        # https://arxiv.org/abs/2101.03961.
        # We also store the previous dtype to cast back the output to the previous dtype
        self.input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype)

        if self.training and self.jitter_noise > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)

        # Shape: [num_groups, tokens_per_group, num_experts]
        self._cast_classifier()
        router_logits = self.classifier(hidden_states)

        # Apply Softmax and cast back to the original `dtype`
        router_probabilities = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        return router_probabilities, router_logits

    def _cast_classifier(self):
        r"""
        `bitsandbytes` `Linear8bitLt` layers does not support manual casting Therefore we need to check if they are an
        instance of the `Linear8bitLt` class by checking special attributes.
        """
        if not (hasattr(self.classifier, "SCB") or hasattr(self.classifier, "CB")):
            self.classifier = self.classifier.to(self.dtype)
    
    def forward(self, hidden_states): 
        router_probs, router_logits = self._compute_router_probabilities(hidden_states)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

class SwitchSparceMLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config 

        self.router = SwitchTop1Router(self.config)
        self.experts = nn.ModuleDict()
        for idx in range(self.config.num_experts):
               self.experts[f"expert_{idx}"] = SwitchDense(self.config)

    def forward(self, hidden_states):
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)

        hidden_states = router_probs * next_states
        return hidden_states, (router_logits, expert_index)

class SwitchTransformersLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the SwitchTransformers style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # SwitchTransformers uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states
    

class MlpResBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config 
        self.activation_function = nn.GELU()
        self.dropout = nn.Dropout(self.config.block_dropout)
        self.norm = nn.LayerNorm(self.config.mlp_hidden_dim)
        self.fc_in = nn.Linear(self.config.mlp_hidden_dim, self.config.mlp_hidden_dim)
        self.fc_out = nn.Linear(self.config.mlp_hidden_dim, self.config.mlp_hidden_dim)
    
    def forward(self, x: torch.Tensor):
        residual = x 
        x = self.norm(x) 
        x = self.fc_in(x)
        x = self.activation_function(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x + residual

class MlpMixerBlock(nn.Module):
    def __init__(self, config, key='channel') -> None:
        super().__init__()
        assert (key in ['channel','dimension']), 'MLP mixing key should be either \'channel\' or \'dimension\''
        self.config = config
        self.activation_function = nn.GELU()
        if key=='channel': #channel/token mixing - stripped mlp 
             self.fc_in = nn.Linear(self.config.mixer_sequence_dim, self.config.mixer_channel_hidden_dim)
        self.fc_out = nn.Linear(self.config.mixer_channel_hidden_dim, self.config.mixer_sequence_dim)
        if key=='dimension': #normal mlp 
            self.fc_in = nn.Linear(self.config.mlp_hidden_dim, self.config.mixer_dimension_hidden_dim)
            self.fc_out = nn.Linear(self.config.mixer_dimension_hidden_dim, self.config.mlp_hidden_dim)

    def forward(self, x: torch.Tensor):
        x = self.fc_in(x)
        x = self.activation_function(x)
        x = self.fc_out(x)
        return x 
    
class MixerBlock(nn.Module):
    def __init__(self, config) -> None: 
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(self.config.mlp_hidden_dim)
        self.mlp_block_channels = MlpMixerBlock(config, key='channel')
        self.mlp_block_dimensions = MlpMixerBlock(config, key='dimension')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = y.transpose(-1, -2)
        y = self.mlp_block_channels(y) #token mixing (channels) - stripped mlp
        y = y.transpose(-1, -2)
        x = x + y
        y = self.norm(x)
        y = self.mlp_block_dimensions(y) #dimension mixing - normal mlp 
        return y


class SelfAttention(nn.Module):
  def __init__(self, input_dim):
    super(SelfAttention, self).__init__()
    self.input_dim = input_dim
    self.query = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.key = nn.Linear(input_dim, input_dim) # [batch_size, seq_length, input_dim]
    self.value = nn.Linear(input_dim, input_dim)
    self.softmax = nn.Softmax(dim=2)
   
  def forward(self, x): # x.shape (batch_size, seq_length, input_dim)
    queries = self.query(x)
    keys = self.key(x)
    values = self.value(x)

    scores = torch.bmm(queries, keys.transpose(1, 2))/(self.input_dim**0.5)
    attention = self.softmax(scores)
    weighted = torch.bmm(attention, values)
    return weighted