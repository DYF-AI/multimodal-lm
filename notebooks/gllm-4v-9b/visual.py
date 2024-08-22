import torch
from torch import nn
from argparse import Namespace
import torch.nn.functional as F
from transformers.activations import ACT2FN
import math
from torch.nn import LayerNorm

def standard_attention(query_layer, key_layer, value_layer, scaling_attention_score=True):
    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_probs = F.softmax(attention_scores, dim=-1)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def attention_fn_default(query_layer, key_layer, value_layer, scaling_attention_score=True):
    if int(torch.__version__.split('.')[0]) >= 2 and scaling_attention_score:
        # Pytorch 2.0 attention uses very much memory if attention_mask is float, and has NaN bug if attention_mask is None.
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer, key_layer, value_layer, 
            attn_mask=None,
            dropout_p=0.,
            is_causal=False
        )
        return attn_output
    else:
        return standard_attention(
            query_layer, key_layer, value_layer, scaling_attention_score=scaling_attention_score
        )

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size)
        self.cls_embedding = nn.Parameter(torch.zeros(1, config.hidden_size))
        self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.proj(images)   # 经过一个卷积层,卷积核大小是14x14, 步长是14, 得到的特征图为 (80,80), 特征的深度是1792,得到一个(1, 1792, 80, 80)
        x = x.flatten(2).transpose(1, 2) # 进行flatten,尺寸为(1, 1792, 6400), transpose后尺寸为(1, 6400, 1792)
        cls_token = self.cls_embedding.expand(x.shape[0], -1, -1) # 构造一个特殊token的特征
        x = torch.cat((cls_token, x), dim=1)    # 将特殊token的特征和,图片特征x, 进行cat, x的尺寸: (1, 6401, 1792)
        x += self.position_embedding.weight.unsqueeze(0)    # 需要要学习的position_embedding权重
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, _ = x.shape
        qkv = self.query_key_value(x)  # 输入x的shape:(1,6401, 1792), 输出qkv的shap:(1, 6401, 5376)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)  # 3, B, H, L, D # qkv的shape变为(3,1,16,6401,112)
        q, k, v = qkv[0], qkv[1], qkv[2]    # 分别得到q, k, v三个矩阵, 每个矩阵的shape: (1,16,6401,112)
        
        out = attention_fn_default(
            q, k, v
        )  # out的shape: (1,16,6401,112)
        output = self.dense(out.transpose(1, 2).view(B, L, -1)) # 再次经过linear层, 输出output的shape: (1, 6401, 1792)
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states # 到这里hidden_states的shape变成了(1,6401,1792)
        attention_output = self.input_layernorm(self.attention(attention_input)) # attention_output shape: (1, 6401, 1792)
        hidden_states = attention_input + attention_output # hidden_states shape: (1, 6401, 1792)
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input)) # mlp_output shape: (1, 6401, 1792)
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states # hidden_sates shape:(1,6401,1792)


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.ffn_hidden_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.ffn_hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x) # 又是加了个linear层, x的输入shape: (1, 1600, 4096), x的输出shape:  (1, 1600, 4096)
        x = self.act1(self.norm1(x)) # LayerNorm+GLU激活函数
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)  #x的输出shape:  (1, 1600, 13696)
        x = self.dense_4h_to_h(x)
        return x  #x的shape: (1, 1600, 4096)


class EVA2CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=config.hidden_size)
        self.conv = nn.Conv2d(in_channels=vision_config.hidden_size, out_channels=config.hidden_size, kernel_size=2, stride=2)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.scaling_factor = vision_config.scaling_factor

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.patch_embedding(images)    # 经过patch_embedding操作后,x的shape为(1, 1792, 6401)
        x = self.transformer(x)     # 经过63层的transformer-layer(LayerNorm, Attention, MLP)
        x = x[:, 1:]  # 由去掉了第一个cls_token的特征, x的shape: (1, 6400, 1792)

        b, s, h = x.shape
        grid_size = int(s**0.5)
        x = x.view(b, grid_size, grid_size, h).permute(0, 3, 1, 2) #x的shape: (1, 1792, 80, 80)
        x = self.conv(x)    # 又一次做卷积, 卷积核大小2x2, 步长为2, x的shape: (1, 4096, 40, 40)

        x = x.flatten(2).transpose(1, 2) #  x的shape: (1, 1600, 4096)
        x = self.linear_proj(x) #  x的shape: (1, 1600, 4096)
        boi = self.boi.expand(x.shape[0], -1, -1) #  boi: (1, 1, 4096)
        eoi = self.eoi.expand(x.shape[0], -1, -1) #  eoi: (1, 1, 4096)
        x = torch.cat((boi, x, eoi), dim=1) #   x的shape: (1, 1602, 4096)
        x = x / self.scaling_factor
        return x
