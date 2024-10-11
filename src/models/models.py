from transformers import BertModel, AutoModel
import torch 
from torch import nn 
import sklearn
import math 

from models.layers import MlpResBlock, SwitchDense, SwitchSparceMLP, SwitchTransformersLayerNorm
from models.configs import MlpConfig, SwitchConfig

def loss_fn(logits,labels):
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(logits, labels.view(-1))
    return loss


def router_z_loss_func(router_logits: torch.Tensor) -> float:
    r"""
    Compute the router z-loss implemented in PyTorch.

    The router z-loss was introduced in [Designing Effective Sparse Expert Models](https://arxiv.org/abs/2202.08906).
    It encourages router logits to remain small in an effort to improve stability.

    Args:
        router_logits (`float`):
            Input logits of shape [batch_size, sequence_length, num_experts]

    Returns:
        Scalar router z-loss.
    """
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)

def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        router_probs (`torch.Tensor`):
            Probability assigned to each expert per token. Shape: [batch_size, seqeunce_length, num_experts].
        expert_indices (`torch.Tensor`):
            Indices tensor of shape [batch_size, seqeunce_length] identifying the selected expert for a given token.

    Returns:
        The auxiliary loss.
    """
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


class HateMLPSimpleModel(nn.Module): 
    def __init__(self, lm='bert-base-uncased', config=None) -> None:
        super().__init__()
        
        self.config = config
        self.lm = AutoModel.from_pretrained(lm , cache_dir='greek-cross-hate-speech/hfcache')
        
        mlp_inner_dim = self.config.hidden_size//2
        self.mlp_dropout = nn.Dropout(0.1)
        self.mlp_norm = nn.LayerNorm(self.config.hidden_size)
        self.mlp_in = nn.Linear(self.config.hidden_size, mlp_inner_dim)
        self.mlp_out = nn.Linear(mlp_inner_dim, self.config.hidden_size)
        self.mlp_act = nn.LeakyReLU()

        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids, attention_mask, labels): 
        output = self.lm(input_ids, attention_mask=attention_mask)
        output = output['pooler_output']

        output = self.mlp_in(self.mlp_norm(output))
        output = self.mlp_dropout(self.mlp_act(output))
        output = self.mlp_out(output)

        logits = self.classifier(output)

        loss = None
        if labels!=None: 
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels)
        return loss, logits
    
class HateMLPResModel(nn.Module): 
    def __init__(self, lm='bert-base-uncased', config=None) -> None:
        super().__init__()
        
        self.config = config
        self.lm = AutoModel.from_pretrained(lm , cache_dir='greek-cross-hate-speech/hfcache')
        
        self.mlp_config = MlpConfig()
        self.mlp_config.mlp_hidden_dim = self.config.hidden_size
        self.mlp_config.mlp_enc_layers = 2
        mlp_layers = [] 
        for _ in range(self.mlp_config.mlp_enc_layers):
            mlp_layers.append(MlpResBlock(self.mlp_config))
        self.mlp_layers = nn.ModuleList(mlp_layers)

        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids, attention_mask, labels): 
        output = self.lm(input_ids, attention_mask=attention_mask)
        output = output['pooler_output']

        for mlp_layer in self.mlp_layers: 
            output = mlp_layer(output)
        
        logits = self.classifier(output)

        loss = None
        if labels!=None: 
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels)
        return loss, logits

class HateSwitchMLPModel(nn.Module): 
    def __init__(self, lm='bert-base-uncased', config=None, experts=None) -> None:
        super().__init__()
        
        self.config = config
        self.lm = AutoModel.from_pretrained(lm , cache_dir='greek-cross-hate-speech/hfcache')
        
        self.switch_config = SwitchGemnetConfig()
        self.switch_config.num_experts = experts
        self.switch_config.expert_capacity = self.config.max_length//experts + (self.config.max_length//experts)*0.5
        self.layer_norm = SwitchTransformersLayerNorm(self.config.hidden_size)
        self.switch_config.d_model = self.config.hidden_size
        self.mlp = SwitchSparceMLP(self.switch_config)
        self.dropout = nn.Dropout(self.switch_config.dropout_rate)

        self.pooler = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids, attention_mask, labels): 
        output = self.lm(input_ids, attention_mask=attention_mask)
        output = output['last_hidden_state']

        output_res = self.layer_norm(output)
        output, (router_logits, expert_index) = self.mlp(output_res)
        output = output_res + self.dropout(output)
        
        bs,sq,dim = output.shape
        output = output.view(bs,dim,sq)
        output = self.pooler(output)
        output = output.squeeze(-1)

        logits = self.classifier(output)
        
        router_probs = nn.Softmax(dim=-1)(router_logits)
        router_aux_loss = self.switch_config.router_aux_loss_coef * load_balancing_loss_func(router_probs, expert_index)
        router_z_loss = self.switch_config.router_z_loss_coef * router_z_loss_func(router_logits)
        router_loss = router_aux_loss + router_z_loss
        loss = None
        if labels!=None: 
            loss = loss_fn(logits.view(-1, self.config.num_labels), labels)
            loss = loss + router_loss
        return loss, logits