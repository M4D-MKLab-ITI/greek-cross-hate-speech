from dataclasses import dataclass

@dataclass
class SwitchConfig():
    dense_act_fn = "relu"
    router_z_loss_coef=0.001
    router_aux_loss_coef=0.001
    router_dtype="float32"
    router_bias=False
    router_jitter_noise=0.01
    router_ignore_padding_tokens=False
    add_router_probs=False
    num_experts=8
    d_model=768
    d_kv=64
    d_ff=2048
    expert_capacity=64
    dropout_rate=0.1
    
@dataclass
class MlpConfig(): 
    mlp_hidden_dim = 384
    block_dropout = 0.1
    mlp_enc_layers = 1   