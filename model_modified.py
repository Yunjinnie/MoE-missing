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
    var_list = [torch.exp(i) for i in log_var_list]
    precision_list = [1.0 / (sigma**2 + 1e-8) for sigma in var_list]
    precision_sum = sum(precision_list)
    sigma_fused = 1.0 / torch.sqrt(precision_sum)
    mu_fused = sigma_fused**2 * sum(mu / (sigma**2 + 1e-8) for mu, sigma in zip(mu_list, var_list))
    #combined_mu = sum([mu * prec for mu, prec in zip(mu_list, precision_list)]) / precision_sum
    # Calibration: 너무 확신있는 전문가에 치우치지 않도록 보정
    log_sigma_fused = torch.log(sigma_fused * calibration + 1e-8)
    return mu_fused, log_sigma_fused

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
    
'''
class FusionMLP(nn.Module):
    def __init__(self, total_input_dim, hidden_dim, output_dim, num_layers):
        super(FusionMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(total_input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.network(x)
'''

###############################################################################
# MoE_Retriever: Learnable token을 포함한 MoE expert pool 구현
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
    
    
'''
class MoE_Retriever(nn.Module):
    def __init__(self, num_modalities, num_patches, hidden_dim, num_experts, num_routers, top_k, num_heads=2, dropout=0.5):
        super(MoE_Retriever, self).__init__()
        layers = []
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=True, top_k=top_k))
        # Try adding layers !
        self.network = nn.Sequential(*layers)

    def forward(self, inputs, expert_indices=None, num_intra=1, num_inter=1, inter_weight=0.5):
        # expert_indices become constraint for MoE-Retriever
        x = self.network[-1](inputs, expert_indices)

        # Adjusting the weight for intra and inter modal inputs
        intra_weight = (1 - inter_weight) / num_intra
        inter_weight = inter_weight / num_inter

        # Intra: first num_intra elements, Inter: remaining num_inter elements
        intra_part = inputs[:num_intra]
        inter_part = inputs[num_intra:]

        # Weighted mean calculation
        weighted_intra = intra_weight * torch.sum(intra_part, dim=0)
        weighted_inter = inter_weight * torch.sum(inter_part, dim=0)

        # Final output combines intra and inter with adjusted weights
        x = weighted_intra + weighted_inter

        return x


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
class FusionLayer(nn.Module):
    def __init__(self, num_modalities, num_patches, hidden_dim, output_dim, num_layers, num_layers_pred, num_experts, num_routers, top_k, num_heads=2, dropout=0.5, mlp_sparse=False):
        super(FusionLayer, self).__init__()
        layers = []
        layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=mlp_sparse, top_k=top_k))
        for j in range(num_layers - 1):
            tmp = (mlp_sparse) & (j % 2 == 1)
            layers.append(TransformerEncoderLayer(num_experts, num_routers, hidden_dim, num_head=num_heads, dropout=dropout, hidden_times=2, mlp_sparse=tmp, top_k=top_k))
        layers.append(MLP(hidden_dim*num_modalities, hidden_dim, output_dim, num_layers_pred, activation=nn.ReLU(), dropout=0.5))
        
        self.network = nn.Sequential(*layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, np.sum([num_patches]*num_modalities), hidden_dim))

    def forward(self, *inputs):
        chunk_size = [input.shape[1] for input in inputs]
        x = torch.cat(inputs, dim=1)
        if self.pos_embed != None:
            x += self.pos_embed

        x = torch.split(x, chunk_size, dim=1)

        for i in range(len(self.network) - 1):
            x = self.network[i](x)
        x = [item.mean(dim=1) for item in x]
        x = torch.cat(x, dim=1)
        x = self.network[-1](x)
        return x

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

###############################################################################
# TransformerEncoderLayer 및 Attention은 기존 코드와 유사하되,
# mlp_sparse 모듈 내에 learnable token (expert_tokens)을 활용하는 구조를 포함
###############################################################################

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
                                           d_hidden=d_model * hidden_times, activation=activation, top_k=top_k, **kwargs)
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=activation, dropout=dropout)
    
    def forward(self, x, expert_index=None):
        # x: list of tensors (one per modality or patch group)
        if self.self_attn:
            chunk_size = [item.shape[1] for item in x]
            x_cat = torch.cat(x, dim=1)  # (B, total_tokens, d_model)
            x_norm = self.norm1(x_cat)
            attn_out = self.attn(x_norm, x_norm)
            x_cat = x_cat + self.dropout(attn_out)
            x = torch.split(x_cat, chunk_size, dim=1)
            x = [item for item in x] # type: list
            # Apply MLP (or MoE-MLP)
            # for i in range(len(x)):
            #     x[i] = x[i] + self.dropout(self.mlp(self.norm2(x[i]))) # , expert_index -> error
            # return x
            if self.mlp_sparse:
                    for i in range(len(chunk_size)):
                        x[i] = x[i] + self.dropout(self.mlp(self.norm2(x[i]), expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout(self.mlp(self.norm2(x[i])))
        else:
            # 다른 attention 전략이 필요하면 구현 (placeholder)
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
        
'''
class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                num_experts,
                num_routers,
                d_model, 
                num_head, 
                dropout=0.1, 
                activation=nn.GELU, 
                hidden_times=2, 
                mlp_sparse = False, 
                self_attn = True,
                top_k=2,
                **kwargs) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation()
        self.attn = Attention(
            d_model, num_heads=num_head, qkv_bias=False, attn_drop=dropout, proj_drop=dropout)
        
        self.mlp_sparse = mlp_sparse
        self.self_attn = self_attn

        if self.mlp_sparse:
            self.mlp = FMoETransformerMLP(num_expert=num_experts, n_router=num_routers, d_model=d_model, d_hidden=d_model * hidden_times, activation=nn.GELU(), top_k=top_k, **kwargs)
        else:
            self.mlp = MLP(input_dim=d_model, hidden_dim=d_model * hidden_times, output_dim=d_model, num_layers=2, activation=nn.GELU(), dropout=dropout)

    def forward(self, x, expert_index=None):
        if self.self_attn:
            if expert_index:
                x = self.attn(x, x)
                x = x + self.dropout1(x)
                x = self.mlp(self.norm2(x), expert_index)
                return x
            
            else:
                chunk_size = [item.shape[1] for item in x]
                x = self.norm1(torch.cat(x, dim=1))
                kv = x
                x = self.attn(x, kv)
                x = x + self.dropout1(x)
                x = torch.split(x, chunk_size, dim=1)
                x = [item for item in x]

                if self.mlp_sparse:
                    for i in range(len(chunk_size)):
                        x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), expert_index))
                else:
                    for i in range(len(chunk_size)):
                        x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        else:
            chunk_size = [item.shape[1] for item in x]
            x = [item for item in x]
            for i in range(len(chunk_size)):
                other_m = [x[j] for j in range(len(chunk_size)) if j != i]
                other_m = torch.cat([x[i], *other_m], dim=1)
                x[i] = self.attn(x[i], other_m)
            x = [x[i]+self.dropout1(x[i]) for i in range(len(chunk_size))]
            if self.mlp_sparse:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i]), expert_index))
            else:
                for i in range(len(chunk_size)):
                    x[i] = x[i] + self.dropout2(self.mlp(self.norm2(x[i])))
        return x

'''

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
    
'''
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), dropout=0.5):
        super(MLP, self).__init__()
        layers = []
        self.drop = nn.Dropout(dropout)
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(activation)
            layers.append(self.drop)
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(self.drop)
            layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
'''

###############################################################################
# FMoETransformerMLP: MoE MLP with sparse routing 및 learnable expert token 포함
###############################################################################

class FMoETransformerMLP(nn.Module):
    def __init__(self, num_expert, n_router, d_model, d_hidden, activation=nn.GELU, top_k=2, dropout=0.1, **kwargs):
        super(FMoETransformerMLP, self).__init__()
        self.num_expert = num_expert
        self.top_k = top_k
        # expert MLPs
        self.experts = nn.ModuleList([MLP(d_model, d_hidden, d_model, num_layers=2, activation=activation, dropout=dropout) 
                                      for _ in range(num_expert)])
        # 각 expert에 대해 learnable token (추가 semantic summary)
        self.expert_tokens = nn.Parameter(torch.randn(num_expert, d_model))
    
    def forward(self, x, expert_index=None):
        # x: (B, N, d_model)
        B, N, d_model = x.shape
        # 간단한 routing: 각 token x와 expert token 간의 유사도 (dot-product)를 계산
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_expert, d_model)
        x_flat = x.view(B * N, d_model)
        scores = torch.matmul(x_flat, expert_tokens.transpose(1,2))  # (B*N, num_expert)
        scores = scores.view(B, N, self.num_expert)
        # 각 토큰마다 top_k expert 선택 (여기서는 단순 평균)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        outputs = torch.zeros_like(x)
        for i in range(self.num_expert):
            mask = (topk_indices == i).float()  # (B, N, top_k)
            if mask.sum() > 0:
                expert_out = self.experts[i](x)  # (B, N, d_model)
                # expert token과의 attention weighted average (간단 구현)
                outputs += expert_out * mask.mean(dim=-1, keepdim=True)
        return outputs

###############################################################################
# Attention: 기본적인 multi-head self-attention (변경 없이 사용)
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
    
'''
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.head_dim = head_dim
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(head_dim * self.num_heads, dim)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, kv, attn_mask=None):
        # attn_mask: (B, N+1, N+1) input-dependent
        eps = 1e-6

        Bx, Nx, Cx = x.shape
        B, N, C = kv.shape
        q = self.q(x).reshape(Bx, Nx, self.num_heads, Cx//self.num_heads)
        q = q.permute(0, 2, 1, 3)
        kv = self.kv(kv)
        kv = kv.reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N+1, C/H) @ (B, H, C/H, N+1) -> (B, H, N+1, N+1)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(Bx, Nx, -1)  # (B, H, N+1, N+1) * (B, H, N+1, C/H) -> (B, H, N+1, C/H) -> (B, N+1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

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