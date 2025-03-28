import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from moe_module import *
from itertools import combinations
import math

###############################################################################
# Utility function: Product-of-Experts fusion (assuming Gaussian parameters)
###############################################################################

def product_of_experts(mu_list, log_var_list, calibration = 1.0):
    # Convert log_sigma to sigma and compute precision (1/sigma^2)
    var_list = [torch.exp(lv) for lv in log_var_list]
    precision_list = [1.0 / (sigma + 1e-8) for sigma in var_list] # sigma**2
    precision_sum = sum(precision_list)
    var_fused = 1.0 / precision_sum # or 1.0 / torch.sqrt(precision_sum)
    #mu_fused = var_fused**2 * sum(mu / (sigma**2 + 1e-8) for mu, sigma in zip(mu_list, var_list))
    mu_fused = sum([mu * prec for mu, prec in zip(mu_list, precision_list)]) / precision_sum
    # Calibration: 너무 확신있는 전문가에 치우치지 않도록 보정
    logvar_fused = torch.log(var_fused * calibration + 1e-8)
    return mu_fused, logvar_fused

def reparameterize(mu, logvar):
    logvar = torch.clamp(logvar, min=-10, max=10)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class ExpertModule(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExpertModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc_mu = nn.Linear(output_dim, output_dim)
        self.fc_logvar = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        
    def forward(self, x, variance=None):
        """
        variance (1/precision: 불확실성)가 주어지면, Bayesian gating처럼 inverse variance (confidence)를 반영
        낮은 분산 (높은 confidence)일수록 gating logit에 더 크게 반영함
        """
        logits = self.fc(x)  # (batch_size, num_experts)
        if variance is not None:
            confidence = torch.exp(-variance)  # low var -> high confidence
            logits = logits * confidence
        weights = F.softmax(logits, dim=-1)
        return weights
    

# 사전학습된 encoder의 feature에 추가하여 분포 파라미터(μ, log σ²)를 추정하는 모듈
class DistributionalHead(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super(DistributionalHead, self).__init__()
        self.fc = nn.Linear(in_dim, in_dim)
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)
    
    def forward(self, x):
        # x: (batch_size, feature_dim)
        h = F.relu(self.fc(x))
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar
    

# 0.1. Disentangled Modal Encoder (Per-modality encoder에서 shared와 specific representation 분리)

class DisentangledModalEncoder(nn.Module):
    def __init__(self, feature_dim, latent_dim_shared, latent_dim_specific, embed_dim):
        super(DisentangledModalEncoder, self).__init__()
        """
        feature_dim: 입력 feature 차원 (예: pretrained ViT/BERT/Librosa feature): {'vision': 20, 'audio': 5, 'text': 768} [m]
        latent_dim_shared: 태스크 관련 공통 정보 차원
        latent_dim_specific: 모달리티 고유 정보 차원
        """
        # # feature extractor (pretrained encoder의 output)
        
        #modality_key = feature_dim.keys().lower()
        #input_dim = feature_dim[modality_key]
        # if modality_key not in embed_dim_dict:
        #     raise ValueError(f"Unknown modality: {modality_key}")
        #embed_dim = embed_dim.get(modality_key, feature_dim)
        #embed_dim = [embed_dim_dict[mod] for mod in feature_dim.keys()] # v, a, t
       
        # raw 입력을 embed_dim 차원으로 투영 (raw_input_dim이 embed_dim과 다르면)
        self.feature_dim = feature_dim
        self.input_proj = nn.Linear(feature_dim, embed_dim) if feature_dim != embed_dim else nn.Identity()
        
        #actual_dim = feature_dim if isinstance(feature_dim, int) else int(feature_dim[0])
        self.feature_extractor = nn.Linear(embed_dim, embed_dim)
        # Distributional Head를 붙여 shared 및 specific representation을 위해 확률 분포 파라미터 산출
        self.shared_head = DistributionalHead(embed_dim, latent_dim_shared)
        self.specific_head = DistributionalHead(embed_dim, latent_dim_specific)
        
    def forward(self, x):
        # x: (batch_size, num_patches, input_dim)
        #print(x.dim()) -> 3
        # if x.dim() > 2:
        #     x = x.view(x.size(0), -1) # If 3D (batch_size, num_patches, patch_size) -> flatten
        #x = self.input_proj(x)
        if x.size(-1) == self.feature_dim:
            # here
            x = self.input_proj(x)  # (B, num_patches, embed_dim)
        elif x.size(-1) == self.embed_dim:
            # 이미 투영된 경우
            pass
        else:
            raise ValueError(f"Unexpected input feature dimension: {x.size(-1)}")
        x = x.mean(dim=1)
        features = F.relu(self.feature_extractor(x))
        mu_shared, logvar_shared = self.shared_head(features)
        z_shared = reparameterize(mu_shared, logvar_shared)
        mu_specific, logvar_specific = self.specific_head(features)
        z_specific = reparameterize(mu_specific, logvar_specific)
        return z_shared, mu_shared, logvar_shared, z_specific, mu_specific, logvar_specific

        

''' 
###############################################################################
# Fusion MLP: 최종 fusion 시, 각 expert로부터 나온 feature를 받아서
# 분포 파라미터 (mu, log_sigma)를 예측하는 모듈.
# 수정 필요
###############################################################################

class FusionMLP(nn.Module):
    def __init__(self, total_input_dim, hidden_dim, output_dim, num_layers):
        super(FusionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        # 최종 출력: output_dim 크기의 mu와 log_sigma (총 2*output_dim)
        layers.append(nn.Linear(hidden_dim, output_dim * 2))
        self.network = nn.Sequential(*layers)
    
    def forward(self, *inputs):
        # inputs가 expert feature들의 리스트라고 가정 (예: modality별 pooled features)
        x = torch.cat(inputs, dim=1)
        out = self.network(x) # original output
        mu, log_sigma = torch.chunk(out, 2, dim=1)
        return mu, log_sigma

###############################################################################
# MoE_Disentangled: Learnable token을 포함한 MoE expert pool 구현
# – 각 expert는 learnable token (query)와 입력 feature (key, value)를 통해 self-attention 기반으로 semantic 정보를 교환
# – 또한 gating 네트워크를 통해 각 expert의 confidence를 산출
###############################################################################

class MoE_Disentangled(nn.Module):
    def __init__(self, num_modalities, num_patches, hidden_dim, num_experts, num_routers, top_k, num_heads=2, dropout=0.5):
        super(MoE_Disentangled, self).__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        # 각 expert마다 학습 가능한 토큰 (query)
        self.expert_tokens = nn.Parameter(torch.randn(num_experts, hidden_dim))
        # Transformer layer (여기서는 하나의 layer만 사용하는 예시)
        self.transformer = TransformerEncoderLayer(num_experts, num_routers, hidden_dim,
                                                   num_head=num_heads, dropout=dropout, 
                                                   hidden_times=2, mlp_sparse=True, top_k=top_k)
        # 간단한 gating network: 각 expert token로부터 confidence (0~1)를 예측
        self.gating = nn.Linear(hidden_dim, 1)
        
    def forward(self, inputs, expert_indices=None, num_intra=1, num_inter=1, inter_weight=0.5):
        # inputs: shape (B, N, hidden_dim) — 모달리티별 혹은 patch-level feature
        batch_size = inputs.shape[0]
        # expert tokens를 배치 차원에 맞게 확장: (B, num_experts, hidden_dim)
        expert_tokens_expanded = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        # expert token과 입력 feature를 concatenation하여 attention의 query/key/value로 사용
        combined = torch.cat([expert_tokens_expanded, inputs], dim=1)
        # transformer에 통과 (여기서는 단일 layer로 가정)
        x = self.transformer([combined], expert_indices)  # x: list of tensors
        x = x[0]  # 예시로 첫 번째 출력 사용; shape (B, num_experts + N, hidden_dim)
        # expert 영역의 출력 추출 (첫 num_experts 토큰)
        expert_features = x[:, :self.num_experts, :]  # (B, num_experts, hidden_dim)
        # 각 expert에 대해 gating을 통해 confidence 계산
        confidence = torch.sigmoid(self.gating(expert_features))  # (B, num_experts, 1)
        
        # 추가: 만약 일부 모달리티가 missing이면, query로 학습 토큰을 그대로 사용하고
        # shared expert의 token로 key/value를 대체하는 방식도 구현할 수 있음.
        # (여기서는 placeholder로 처리)
        
        # Intra/Inter weight 조정 (예시)
        intra_weight = (1 - inter_weight) / num_intra
        inter_weight_adj = inter_weight / num_inter
        # inputs를 intra와 inter로 나눈다고 가정 (placeholder)
        intra_part = inputs[:, :num_intra, :]
        inter_part = inputs[:, num_intra:num_intra+num_inter, :]
        weighted_intra = intra_weight * torch.sum(intra_part, dim=1)
        weighted_inter = inter_weight_adj * torch.sum(inter_part, dim=1)
        fused_features = weighted_intra + weighted_inter  # (B, hidden_dim)
        
        # 최종적으로 expert_features, confidence, 그리고 fused_features 반환
        return expert_features, confidence, fused_features
    
    def gate_loss(self):
        # gating 관련 loss를 계산 (예시: confidence의 분산을 줄여서 균등 분포 유도)
        # 실제로는 각 expert의 routing 결과와 target 분포를 비교하는 loss 등을 사용할 수 있음.
        g_loss = []
        for mn, mm in self.named_modules():
            # print(mn)
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        print(f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice.")
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)
    
    

###############################################################################
# FusionLayer: 여러 modality에서 나온 feature들을 transformer와 fusion MLP로 통합
# – 각 모달리티의 feature를 입력받아 positional embedding을 더하고, 여러 transformer layer를 통과시킨 후, modality별로 pooled feature를 추출.
# – FusionMLP를 통해 각 modality에 대한 (mu, log_sigma)를 산출하고, Product-of-Experts로 최종 분포를 fusion
###############################################################################

class FusionLayer(nn.Module):
    def __init__(self, num_modalities, num_patches, hidden_dim, output_dim, num_layers, num_layers_pred, num_experts, num_routers, top_k, num_heads=2, dropout=0.5, mlp_sparse=False):
        super(FusionLayer, self).__init__()
        layers = []
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=mlp_sparse, top_k=top_k))
        for j in range(num_layers - 1):
            tmp = mlp_sparse and (j % 2 == 1)
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=tmp, top_k=top_k))
        self.transformer_layers = nn.Sequential(*layers)
        # FusionMLP: modality별 pooled feature를 받아 (mu, log_sigma)를 예측
        self.fusion_mlp = FusionMLP(hidden_dim * num_modalities, hidden_dim, output_dim, num_layers_pred)
        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([num_patches]*num_modalities), hidden_dim))
    
    def forward(self, *inputs):
        # inputs: list of modality features, each with shape (B, num_patches, hidden_dim)
        chunk_size = [inp.shape[1] for inp in inputs]
        x = torch.cat(inputs, dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        # split back into modality-specific chunks
        x = torch.split(x, chunk_size, dim=1)
        # Pass each modality's features through transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        # Pool modality features (mean pooling)
        modality_features = [feat.mean(dim=1) for feat in x]  # list of (B, hidden_dim)
        # Fusion MLP produces (mu, log_sigma) per modality (concatenated)
        mu, log_sigma = self.fusion_mlp(*modality_features)
        # 가정: FusionMLP의 output을 modality 개수로 나누어 expert별 distribution parameter로 해석
        expert_dim = mu.shape[1] // len(modality_features)
        mu_list = torch.chunk(mu, len(modality_features), dim=1)
        log_sigma_list = torch.chunk(log_sigma, len(modality_features), dim=1)
        # Product-of-Experts fusion
        mu_fused, log_sigma_fused = product_of_experts(mu_list, log_sigma_list)
        fused = torch.cat([mu_fused, log_sigma_fused], dim=1)
        return fused
    
    def gate_loss(self):
        # 각 모듈 내 gate loss를 누적 (placeholder)
        g_loss = 0.0
        for mn, mm in self.named_modules():
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is not None:
                        g_loss += i_loss
        return g_loss
'''


###############################################################################
# FMoETransformerMLP: MoE MLP with sparse routing 및 learnable expert token 포함
# 1. MoE-based Transformer FFN modules
#    - MoE는 full shared expert, combination experts (for missing modality combinations),
#      그리고 specific experts (with Bayesian gating)를 포함함.
###############################################################################

class FMoETransformerMLP(nn.Module):
    def __init__(self, num_expert, n_router, d_model, d_hidden, activation=nn.GELU, top_k=2, dropout=0.1, **kwargs):
        super(FMoETransformerMLP, self).__init__()
        self.num_expert = num_expert
        self.top_k = top_k
        # expert MLPs
        self.experts = nn.ModuleList([
            MLP(d_model, d_hidden, d_model, num_layers=2, activation=activation, dropout=dropout) for _ in range(num_expert)])
        # 각 expert에 대해 learnable token (추가 semantic summary -> query 역할)
        self.expert_tokens = nn.Parameter(torch.randn(num_expert, d_model))
    
    def forward(self, x, expert_index=None):
        # x: (B, N, d_model)
        B, N, d_model = x.shape
        # 간단한 routing: 각 token x와 expert token 간의 dot-product similarity 계산
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_expert, d_model)
        x_flat = x.view(B * N, d_model)
        
        scores = torch.matmul(x_flat, expert_tokens.transpose(1,2))  # (B*N, num_expert)
        scores = scores.view(B, N, self.num_expert)
        # 각 토큰마다 top_k expert 선택 (여기서는 단순 평균)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        # top = torch.argmax(scores, dim= -1).unsqueeze(-1).float() # (B, N, 1)
        outputs = torch.zeros_like(x)
        for i in range(self.num_expert):
            mask = (topk_indices == i).float()  # (B, N, top_k)
            if mask.sum() > 0:
                expert_out = self.experts[i](x)  # (B, N, d_model)
                # expert token과의 attention weighted average (간단 구현)
                outputs += expert_out * mask.mean(dim=-1, keepdim=True)
                
        return outputs
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.GELU, dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(activation if isinstance(activation, nn.Module) else activation())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation if isinstance(activation, nn.Module) else activation())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


###############################################################################
# 2. Transformer Encoder Layer with MoE FFN (대체 FFN 대신 MoE 적용)
# TransformerEncoderLayer 및 Attention: mlp_sparse 모듈 내에 learnable token (expert_tokens)을 활용하는 구조를 포함
###############################################################################

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_experts, num_routers, d_model, num_head, dropout=0.1, activation=nn.GELU, hidden_times=2, mlp_sparse=True, self_attn=True, top_k=2, **kwargs):
        super(TransformerEncoderLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, num_heads=num_head, attn_drop=dropout, proj_drop=dropout)
        self.self_attn = self_attn
        self.mlp_sparse = mlp_sparse
        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, 
                                           d_hidden=d_model * hidden_times, activation=activation, top_k=top_k, dropout=dropout) # **kwargs
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=activation, dropout=dropout)
    
    def forward(self, x, expert_index=None, **kwargs):
        # x: list of tensors (one per modality or patch group)
        if not isinstance(x, (list, tuple)):
            x = [x]
        if self.self_attn:
            chunk_size = [item.shape[1] for item in x]
            ### dimension error
            x_cat = torch.cat(x, dim=1)  # (B, total_tokens, d_model)
            x_norm = self.norm1(x_cat)
            attn_out = self.attn(x_norm, x_norm)
            x_cat = x_cat + self.dropout(attn_out)
            x = torch.split(x_cat, chunk_size, dim=1) # 다시 modality 별로 split
            #x = [item for item in x] # type: list
            
            # Apply MLP (or MoE-MLP)
            # for i in range(len(x)):
            #     x[i] = x[i] + self.dropout(self.mlp(self.norm2(x[i]))) # , expert_index -> error
            # return x
            #print(len(x)) 1 
            #print(len(chunk_size)) 1
            temp = []
            
            if self.mlp_sparse:
                    for item in x: # chunk_size
                        temp.append(item + self.dropout(self.mlp(self.norm2(item), expert_index)))
            else:
                for item in x:
                    temp.append(item + self.dropout(self.mlp(self.norm2(item))))
            # if isinstance(x, list):
            #     x = torch.cat(x, dim=1)
            x = torch.cat(temp, dim=1)
        else:
            # 다른 attention 전략? (placeholder)
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m)
            x = [x[i]+self.dropout(x[i]) for i in range(len(chunk_size))]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout(self.mlp(self.norm2(x[i]), expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout(self.mlp(self.norm2(x[i])))
        return x
 


###############################################################################
# 3. Attention Modules: 기본 Multi-Head Attention 및 Learnable Token Attention
###############################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, kv, attn_mask=None):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0,2,1,3)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1,2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    
class LearnableTokenAttention(nn.Module):
    def __init__(self, embed_dim, num_tokens):
        super(LearnableTokenAttention, self).__init__()
        # 학습 가능한 토큰 (low-dimension embedding): query 역할 (essential modality information으로 저차원으로 transfer)
        self.learnable_tokens = nn.Parameter(torch.randn(num_tokens, embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4)
        
    def forward(self, expert_features):
        # expert_features: (batch_size, num_experts, embed_dim)
        batch_size = expert_features.size(0)
        # learnable token을 batch에 맞게 확장: (num_tokens, batch_size, embed_dim)
        query = self.learnable_tokens.unsqueeze(1).repeat(1, batch_size, 1)
        # expert_features를 transpose: (num_experts, batch_size, embed_dim)
        expert_features_t = expert_features.transpose(0, 1)
        key = self.key_proj(expert_features_t)
        value = self.value_proj(expert_features_t)
        attn_output, attn_weights = self.attn(query, key, value)
        # 여러 토큰에 대해 평균 pooling
        output = attn_output.mean(dim=0)  # (batch_size, embed_dim)
        return output, attn_weights
    
    
def orthogonality_loss(shared_repr, specific_repr):
    dot_product = torch.sum(shared_repr * specific_repr, dim=1)
    return torch.mean(dot_product ** 2)


## 4. MoE Expert Pool with Calibration and Bayesian Gating

class MoEExpertPool(nn.Module):
    def __init__(self, embed_dim, num_full_shared, num_combination, num_specific):
        """
        - full_shared_experts: 모든 모달리티의 shared representation에 대해 무조건 적용
        - combination_experts: missing modality 조합별로 적용 (예, m2가 missing이면 m1과 m3 조합)
        - specific_experts: 모달리티별 고유 정보에 대해 적용 (Bayesian gating 적용)
        """
        
        super(MoEExpertPool, self).__init__()
        # Expert 그룹들
        self.full_shared_experts = nn.ModuleList([ExpertModule(embed_dim, embed_dim) for _ in range(num_full_shared)])
        self.combination_experts = nn.ModuleList([ExpertModule(embed_dim, embed_dim) for _ in range(num_combination)])
        self.specific_experts = nn.ModuleList([ExpertModule(embed_dim, embed_dim) for _ in range(num_specific)])
        
        # 각 그룹별 gating network
        self.full_shared_gate = GatingNetwork(embed_dim, num_full_shared)
        self.combination_gate = GatingNetwork(embed_dim, num_combination)
        self.specific_gate = GatingNetwork(embed_dim, num_specific)
    
    def forward(self, x, modality_mask):
        """
        x: aggregated shared representation (batch_size, embed_dim)
        modality_mask: binary mask (batch_size, num_modalities) -> 1이면 present, 0이면 missing
        """
        batch_size = x.size(0)
        
        # Full Shared Experts (모든 modality에 대해 무조건 적용)
        full_shared_outputs = []
        gate_full = self.full_shared_gate(x)  # (batch_size, num_full_shared)
        for i, expert in enumerate(self.full_shared_experts):
            mu, logvar = expert(x)
            weight = gate_full[:, i].unsqueeze(-1)
            full_shared_outputs.append((mu * weight, logvar))
        
        # Combination Experts: routing은 missing modality 조합에 따라 적용
        # (여기서는 단순화를 위해 x를 그대로 입력받아 gating)
        combination_outputs = []
        gate_comb = self.combination_gate(x)  # (batch_size, num_combination)
        # 예시: modality_mask의 평균이 1이면 모두 present → gating weight 0.0, 그렇지 않으면 사용
        mask_mean = modality_mask.float().mean(dim=1, keepdim=True)
        for i, expert in enumerate(self.combination_experts):
            mu, logvar = expert(x)
            weight = gate_comb[:, i].unsqueeze(-1) * (1.0 - mask_mean)  # missing modality가 있으면 weight 부여
            combination_outputs.append((mu * weight, logvar))
        
        # Specific Experts: 모달리티별 고유 정보, Bayesian gating 적용 (variance 반영)
        specific_outputs = []
        gate_specific = self.specific_gate(x)  # (batch_size, num_specific)
        for i, expert in enumerate(self.specific_experts):
            mu, logvar = expert(x)
            weight = gate_specific[:, i].unsqueeze(-1)
            specific_outputs.append((mu * weight, logvar))
        
        # PoE Fusion: 모든 experts의 출력(mu, logvar)를 결합
        expert_mu_list = []
        expert_logvar_list = []
        for mu, logvar in full_shared_outputs + combination_outputs + specific_outputs:
            expert_mu_list.append(mu)
            expert_logvar_list.append(logvar)
        
        # 각 배치 샘플마다 PoE 결합 수행 ?
        combined_mu = []
        combined_logvar = []
        
        for bs in range(batch_size):
            mu_bs = [mu[bs] for mu in expert_mu_list]
            logvar_bs = [lv[bs] for lv in expert_logvar_list]
            mu_comb, logvar_comb = product_of_experts(mu_bs, logvar_bs, calibration=1.0) ## or not
            combined_mu.append(mu_comb.unsqueeze(0))
            combined_logvar.append(logvar_comb.unsqueeze(0))
        combined_mu = torch.cat(combined_mu, dim=0)  # (batch_size, embed_dim)
        combined_logvar = torch.cat(combined_logvar, dim=0)  # (batch_size, embed_dim)
        return combined_mu, combined_logvar


# 6. Transformer Encoder with MoE: 최종적으로 MoE 출력 + learnable token attention + Transformer + Prediction head
class TransformerEncoderMoE(nn.Module):
    def __init__(self, embed_dim, num_tokens, num_classes, 
                 num_experts=7, num_routers = 3, hidden_times=2, activation=nn.GELU, top_k=2, dropout=0.2, mlp_sparse=True):
                 #num_full_shared=2, num_combination=2, num_specific=3, mlp_sparse=True):
        super(TransformerEncoderMoE, self).__init__()
        self.embed_dim = embed_dim
        # Learnable token attention: expert pool의 출력을 종합
        self.learnable_attention = LearnableTokenAttention(embed_dim, num_tokens)
        # MoE expert pool
        self.moe_pool = MoEExpertPool(embed_dim, num_full_shared=1, num_combination=3, num_specific=5)
        # Transformer encoder (기본적인 layer로 구성)
        encoder_layer = TransformerEncoderLayer(num_experts, num_routers, embed_dim, num_head=4, dropout=dropout, hidden_times=hidden_times, mlp_sparse=mlp_sparse, top_k=top_k)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # 최종 분류 head
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.mlp_sparse = mlp_sparse
        
        # if self.mlp_sparse:
        #     self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=embed_dim, 
        #                                     d_hidden=embed_dim * hidden_times, activation=activation, top_k=top_k, **kwargs)
        # else:
        #     self.mlp = MLP(input_dim=embed_dim, hidden_dim=embed_dim * hidden_times, output_dim=embed_dim, num_layers=2, activation=activation, dropout=dropout)

        
    def forward(self, shared_representation, modality_mask):
        """
        shared_representation: 모달리티들의 aggregated shared feature, shape (batch_size, embed_dim)
        modality_mask: (batch_size, num_modalities), 1이면 present, 0이면 missing
        """
        # (1) MoE Expert Fusion with PoE: expert들의 outputs (mu, logvar)를 PoE 방식으로 결합
        moe_mu, moe_logvar = self.moe_pool(shared_representation, modality_mask)
        # reparameterization: latent sample 생성
        std = torch.exp(0.5 * moe_logvar)
        z = moe_mu + torch.randn_like(std) * std  # (batch_size, embed_dim)
        
        # (2) Learnable Token Attention: MoE 출력(z)을 토큰으로 사용하여 semantic 정보 추출
        # 여기서는 단순화를 위해 z를 하나의 토큰으로 간주하여 attention 적용
        expert_features = z.unsqueeze(1)  # (batch_size, 1, embed_dim)
        token_attn_out, attn_weights = self.learnable_attention(expert_features)
        
        # (3) Transformer Encoder: attention output을 sequence로 처리 (add a sequence dimension)
        seq = token_attn_out.unsqueeze(0)  # (seq_len=1, batch_size, embed_dim)
        ### 여기서 error
        transformer_out = self.transformer_encoder(seq)  # (1, batch_size, embed_dim)
        #print(f"Type of transformer_out: {type(transformer_out)}") <class 'list'>
        #print(f"Length of transformer_out (if list): {len(transformer_out) if isinstance(transformer_out, list) else 'N/A'}") 1

        transformer_out = transformer_out.squeeze(0)  # (batch_size, embed_dim)
        
        # (4) 최종 분류
        logits = self.classifier(transformer_out)
        return logits, moe_mu, moe_logvar, attn_weights
    


# 7. Final MultiModal Model: 전체 알고리즘 통합
#   - (0) 모달리티별 encoder: 각 모달리티에 대해 DisentangledModalEncoder 사용
#   - (1) shared representations를 fusion (예: concatenation 후 선형 변환)
#   - (2) TransformerEncoderMoE를 통해 최종 예측

class MMDisentangled(nn.Module):
    def __init__(self, modality_feature_dims, latent_dim_shared, latent_dim_specific, num_classes):
        """
        modality_feature_dims: list, 각 모달리티 입력 feature 차원 (예: [512, 768, 128])
        latent_dim_shared: 각 모달리티의 shared representation 차원
        latent_dim_specific: 각 모달리티의 specific representation 차원
        """
        super(MMDisentangled, self).__init__()
        self.modalities = list(modality_feature_dims.keys())
        embed_dim_dict = {'vision': 512, 'text': 768, 'audio': 128}
        # if raw_input_dims is None:
        #     raw_input_dims = [None] * self.num_modalities
        # 모달리티별 disentangled encoder (각 encoder는 shared와 specific representation 모두 출력)
        self.encoders = nn.ModuleList([
            DisentangledModalEncoder(modality_feature_dims[m], latent_dim_shared, latent_dim_specific, embed_dim = embed_dim_dict[m]) for m in self.modalities
        ])
        #self.classifier = TransformerClassifier(latent_dim, num_tokens=self.num_modalities, num_classes=num_classes)
        # 간단한 fusion: 각 모달리티의 shared representation을 concatenate한 후 선형변환
        fusion_input_dim = len(self.modalities) * latent_dim_shared
        self.shared_fusion_linear = nn.Linear(fusion_input_dim, latent_dim_shared)
        # Transformer Encoder with MoE를 통한 최종 예측 (shared 정보 사용)
        self.transformer_moe = TransformerEncoderMoE(embed_dim=latent_dim_shared, num_tokens=len(self.modalities), num_classes=num_classes)
    
    
    def forward(self, inputs):
        """
        inputs: list of modality feature vectors (tensor), 각 tensor shape은 (batch_size, feature_dim) - per modality 입력
        """
        shared_list = []
        specific_list = []
        # 각 모달리티 encoder 실행
        # latent_tokens = []
        # mus, logvars = [], []
        for i, x in enumerate(inputs):
            #print(x[0]) # torch.Size([50, 20]),torch.Size([50, 5]),torch.Size([50, 768])
            z_shared, mu_shared, logvar_shared, z_specific, mu_specific, logvar_specific = self.encoders[i](x) ### dimension error 
            shared_list.append(z_shared)      # (B, latent_dim_shared)
            specific_list.append(z_specific)  # (B, latent_dim_specific)
            # contrastive loss 등 disentanglement loss는 학습 시 별도 계산
            
        # # (batch_size, num_modalities, latent_dim)
        # latent_seq = torch.cat(latent_tokens, dim=1)
        # logits = self.transformer(latent_seq)
        
        # Fusion: shared representation들을 concatenate 후 선형 변환
        fused_shared = torch.cat(shared_list, dim=1)  # (B, num_modalities*latent_dim_shared)
        fused_shared = self.shared_fusion_linear(fused_shared)  # (B, latent_dim_shared)
        
        # modality_mask: 각 모달리티가 존재하는지 (여기서는 모두 present로 가정)
        modality_mask = torch.ones((inputs[0].shape[0], len(self.modalities)), device=fused_shared.device)
        ### 여기까지는 옴
        logits, moe_mu, moe_logvar, attn_weights = self.transformer_moe(fused_shared, modality_mask)
        return logits, (shared_list, specific_list), moe_mu, moe_logvar, attn_weights

    def gate_loss(self):
        g_loss = []
        for mn, mm in self.named_modules():
            # print(mn)
            if hasattr(mm, 'all_gates'):
                for i in range(len(mm.all_gates)):
                    i_loss = mm.all_gates[f'{i}'].get_loss()
                    if i_loss is None:
                        print(f"[WARN] The gate loss if {mn}, modality: {i} is emtpy, check weather call <get_loss> twice.")
                    else:
                        g_loss.append(i_loss)
        return sum(g_loss)

'''
class TransformerClassifier(nn.Module):
    def __init__(self, latent_dim, num_tokens, num_classes, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(latent_dim, latent_dim)
        self.pos_encoder = PositionalEncoding(latent_dim)
        
        encoder_layer = TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(latent_dim, num_classes)
        
    def forward(self, x):
        # x: (batch_size, num_tokens, latent_dim)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (seq_len, batch_size, latent_dim)
        encoded = self.transformer_encoder(x)
        encoded = encoded.mean(dim=0)  # 전체 토큰에 대해 평균 pooling
        logits = self.classifier(encoded)
        return logits

'''
    


class Custom3DCNN(nn.Module):
    #Architecture provided by: End-To-End Alzheimer's Disease Diagnosis and Biomarker Identification
    def __init__(self, hidden_dim=128):
        super(Custom3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.dropout1 = nn.Dropout3d(0.2)


        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=3)
        self.dropout2 = nn.Dropout3d(0.2)

        self.conv5 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.conv6 = nn.Conv3d(128, hidden_dim, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=(4, 4, 4))
        self.dropout3 = nn.Dropout3d(0.2)

        # Flatten the output and add a fully connected layer to reduce to hidden_dim
        self.fc = nn.Linear(hidden_dim * 3 * 3 * 4, hidden_dim)
        
        # self.fc1 = nn.Linear(128*6*6, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, num_classes)
        # self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(self.pool1(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(self.pool2(x))

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.dropout3(self.pool3(x))

        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)  # Apply the fully connected layer
        
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x)

        return x


class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """
    def __init__(self, feature_size, num_patches, embed_dim, dropout=0.25):
        super().__init__()
        patch_size = math.ceil(feature_size / num_patches)
        pad_size = num_patches*patch_size - feature_size
        self.pad_size = pad_size
        self.num_patches = num_patches
        self.feature_size = feature_size
        self.patch_size = patch_size
        self.projection = nn.Linear(patch_size, embed_dim)

    def forward(self, x):
        x = F.pad(x, (0, self.pad_size)).view(x.shape[0], self.num_patches, self.patch_size)
        # x = F.normalize(x, dim=-1)
        x = self.projection(x)
        return x
    
        
##################################################
# 단순 linear projection이었던 위의 patch embedding 대신
# modality specific encoder (ViT, BERT, Librosa) 사용
# +++ 평균, log var까지 return -> transformer, attention layer도 수정해야 함
##################################################
class ModalitySpecificEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()
        
    def forward(self, x):
        
        return x


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=False, has_padding=False, batch_first=True, last_only=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.last_only = last_only
        self.batch_first = batch_first
        self.has_padding = has_padding
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            batch_first=batch_first, 
            bidirectional=False
        )
        self.dropout_layer = nn.Dropout(0.5) if dropout else nn.Identity()

    def forward(self, x):
        # GRU forward pass
        output, _ = self.gru(x)
        if self.last_only:
            # Use only the last output for each sequence
            if self.batch_first:
                output = output[:, -1, :]
            else:
                output = output[-1, :, :]
        output = self.dropout_layer(output)
        return output

class VGG11Slim(nn.Module):
    def __init__(self, out_channels, dropout=False, dropoutp=0.2, freeze_features=True):
        super(VGG11Slim, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropoutp),
            nn.Linear(4096, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x