import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from itertools import combinations

# ---------- Utility Functions ----------
def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10, max=10)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def product_of_experts(mu_list, logvar_list, calibration=1.0):
    # Convert logvar to var and compute precision
    var_list = [torch.exp(lv) for lv in logvar_list]
    precision_list = [1.0 / (v + 1e-8) for v in var_list]
    precision_sum = sum(precision_list)
    mu_fused = sum(mu * prec for mu, prec in zip(mu_list, precision_list)) / precision_sum
    var_fused = 1.0 / precision_sum
    logvar_fused = torch.log(var_fused * calibration + 1e-8)
    return mu_fused, logvar_fused


class Critic(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
    def forward(self, a, b):
        return torch.sum(self.net(a) * b, dim=-1)  # (B,)
    
class CondCritic(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.feat_dim = dim
        self.embed_y = nn.Linear(num_classes, dim)
        self.critic = Critic(dim * 2)
    def forward(self, a, b, y_onehot):
        # a,b: (B, d), y_onehot: (B, C)
        y_emb = self.embed_y(y_onehot) # (B, d)
        ab = torch.cat([a, y_emb], dim=-1) # (B,2d)
        bb = torch.cat([b, y_emb], dim=-1)
        return self.critic(ab, bb)
    
# InfoNCE lower bound
def info_nce(z1, z2, critic, temperature=0.1):
    B = z1.size(0)
    # compute (B,B) logits
    z1_rep = z1.unsqueeze(1).expand(-1, B, -1).reshape(B*B, -1)# -1, z1.size(-1) 
    z2_rep = z2.repeat(B, 1)
    logits = critic(z1_rep, z2_rep).view(B, B) / temperature
    labels = torch.arange(B, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

# NCE-CLUB upper bound
def club_upper_bound(z1, z2, critic):
    pos = critic(z1, z2).mean()
    neg = critic(z1, z2[torch.randperm(z2.size(0))]).mean()
    return pos - neg

# Jensen-Shannon divergence loss
def jsd_loss(a, b, temperature=0.1):
    p = F.softmax(a / temperature, dim=-1)
    q = F.softmax(b / temperature, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * (F.kl_div(m.log(), p, reduction='batchmean') +
                  F.kl_div(m.log(), q, reduction='batchmean'))

# Orthogonality loss between shared & specific encoder
def orth_loss(shared, specific, eps=1e-6):
    # or, batch-wise Gram
    shared = F.normalize(shared, dim=-1, eps=eps)
    specific = F.normalize(specific, dim=-1, eps=eps)
    cos = torch.sum(shared * specific, dim= -1) #/ (shared.norm(dim=-1) * specific.norm(dim=-1) + eps)
    return torch.mean(1- cos ** 2) # cos.abs().mean()

# ---------- Distributional Head ----------
class DistributionalHead(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, in_dim)
        self.mu = nn.Linear(in_dim, latent_dim)
        self.logvar = nn.Linear(in_dim, latent_dim)
    def forward(self, x):
        h = F.relu(self.fc(x))
        return self.mu(h), self.logvar(h)

# ---------- Disentangled Modal Encoder ----------
class DisentangledModalEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256,
                 latent_shared=128, latent_specific=128,
                 n_layers=1):
        super().__init__()
        self.emb = nn.Linear(input_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4), num_layers=n_layers)
        self.shared_head = DistributionalHead(d_model, latent_shared)
        self.specific_head = DistributionalHead(d_model, latent_specific)
    def forward(self, x):
        # x: (B, T, D_in)
        h = self.emb(x) * (self.emb.out_features**0.5)
        h = h.transpose(0,1)  # seq first
        h = self.encoder(h)
        h = h.transpose(0,1)  # batch first
        pooled = h.mean(dim=1)
        mu_s, logv_s = self.shared_head(pooled)
        mu_u, logv_u = self.specific_head(pooled)
        
        sigma_shared = torch.exp(0.5 * logv_s)
        sigma_specific = torch.exp(0.5 * logv_u)
        z_s = reparameterize(mu_s, logv_s)
        z_u = reparameterize(mu_u, logv_u)
        return {'rep_shared': pooled, 'mu_shared': mu_s, 'logvar_shared': logv_s, 'sigma_shared': sigma_shared, 'z_shared': z_s,
                'rep_specific': pooled, 'mu_specific': mu_u, 'logvar_specific': logv_u, 'sigma_specific': sigma_specific, 'z_specific': z_u}

# ---------- Expert Module ----------
class ExpertModule(nn.Module):
    # def __init__(self, dim):
    #     self.fc1 = nn.Linear(dim, dim)
    #     self.head = DistributionalHead(dim, dim)
    # def forward(self, x):
    #     h = F.relu(self.fc1(x))
    #     mu, logv = self.head(h)
    #     return mu, logv
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, input_dim)
        self.fc_mu    = nn.Linear(input_dim, output_dim)
        self.fc_logvar= nn.Linear(input_dim, output_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

# ---------- Learnable Token Attention ----------
class LearnableTokenAttention(nn.Module):
    def __init__(self, embed_dim, num_tokens=1):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=4)
    def forward(self, features, apply_mask=None):
        # features: (B, N, d), tokens: (num_tokens, d)
        B, N, d = features.size()
        # prepare: query: (num_tokens, B, d)
        q = self.tokens.unsqueeze(1).expand(-1, B, -1)
        kv = features.transpose(0,1)  # (N, B, d)
        attn_out, _ = self.attn(q, kv, kv, key_padding_mask=apply_mask)
        # return mean over tokens
        return attn_out.mean(dim=0)
    
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, query, key_value, attn_mask=None):
        B, Q, C = query.shape
        _, K, _ = key_value.shape
        H = self.num_heads
        D = C // H

        # 1) Linear + split heads
        q = self.q(query)                          # (B, Q, C)
        q = q.view(B, Q, H, D).permute(0,2,1,3)    # (B, H, Q, D)

        kv = self.kv(key_value)                   # (B, K, 2*C)
        kv = kv.view(B, K, 2, H, D).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]                       # each: (B, H, K, D)

        # 2) Scaled dot‑product
        scores = (q @ k.transpose(-2,-1)) * self.scale  # (B, H, Q, K)

        # 3) Masking - optional
        if attn_mask is not None:
            # attn_mask: (B, Q, K)
            scores = scores.masked_fill(~attn_mask.unsqueeze(1), float('-inf'))

        # 4) Softmax + dropout
        attn = scores.softmax(dim=-1)             # (B, H, Q, K)
        attn = self.attn_drop(attn)

        # 5) Aggregate
        out = (attn @ v)                          # (B, H, Q, D)
        out = out.permute(0,2,1,3).reshape(B, Q, C) # (B, Q, C)

        # 6) Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

# ---------- Gating Network ----------
class GatingNetwork(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(dim, num_experts)
    def forward(self, x, variance=None, alpha=1.0, beta=1.0):
        # x: (B, d), variance: (B, E)
        logits = self.fc(x)
        if variance is not None:
            precision = 1.0 / (variance + 1e-8)
            logits = alpha * logits + beta * torch.log(precision + 1e-8)
        return F.softmax(logits, dim=-1)

# ---------- MoE Expert Pool ----------
class MoEExpertPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        modalities = ['audio','vision','text']
        pairs = list(combinations(modalities,2))
        # experts
        self.expert_sh_all = ExpertModule(dim, dim)
        ## self.experts_sh_pair = nn.ModuleDict({f"{i}{j}": ExpertModule(dim*2) for i,j in pairs})
        self.experts_sh_pair = nn.ModuleDict({
            f"{i}{j}": ExpertModule(input_dim=dim*2, output_dim=dim) for i, j in pairs
        })
        self.experts_sp = nn.ModuleDict({m: ExpertModule(dim, dim) for m in modalities})
        # tokens for missing fallback
        self.tokens_pair = nn.ParameterDict({f"{i}{j}": nn.Parameter(torch.randn(dim)) for i,j in pairs})
        # attention for fallback
        self.token_attn = LearnableTokenAttention(dim, num_tokens=1)
        self.attn = Attention(dim, num_heads=4)
        # gating
        total_experts = 1 + len(pairs) + len(modalities)
        self.gate = GatingNetwork(dim, total_experts)
        
        # last forward -> gate weights
        self._last_gate_w = None
        
    def forward(self, reps_sh, reps_sp, present):
        B = next(iter(reps_sh.values())).size(0)
        dim = next(iter(reps_sh.values())).size(1)
        device = next(iter(reps_sh.values())).device
        
        mus, logvs = [], []
        expert_names = []   # 1) full, 2) pairwise, 3) specific
        inputs_for_fallback = []
        
        # 1) full shared
        # avg over present modalities
        sh_stack = torch.stack([reps_sh[m] for m in present], dim=1) # (B, P, d)
        avg_sh   = sh_stack.mean(dim=1) # (B, d)
        mu_all, lv_all = self.expert_sh_all(avg_sh)
        mus.append(mu_all)
        logvs.append(lv_all)
        expert_names.append("all")
        
        # 2) pairwise
        #pairs = list(self.experts_sh_pair.keys())
        for key, expert in self.experts_sh_pair.items():
            expert_names.append(key)
            i,j = key[0],key[1]
            if i in present and j in present:
                inp = torch.cat([reps_sh[i], reps_sh[j]], dim=-1) # (B, 2d)
                #mu, lv = expert(inp)
            else:
                # fallback via attention: token -> feature to full-shared
                token = self.tokens_pair[key] #(d,) 
                query = token.unsqueeze(0).unsqueeze(1).expand(B,1,dim) # -1
                #feat = avg_sh.unsqueeze(1) # (B, 1, d)
                feat = reps_sh[present[0]].unsqueeze(1)  # use any shared as kv
                attn_out = self.attn(query, feat)  # (B,d)
                #attn_out = attn_out.squeeze(1) 
                # project to pair input
                inp_pair = torch.cat([attn_out, attn_out], dim=-1).squeeze(1)  # (B, 2d = 512)
            mu, lv = expert(inp_pair)
            mus.append(mu)
            logvs.append(lv)
            
        # 3) specific
        for m, expert in self.experts_sp.items():
            expert_names.append(f"sp_{m}")
            if m in present:
                mu, lv = expert(reps_sp[m])
            else:
                # missing modality → dummy 분포 (very large var 로 confidence 낮춤)
                mu = torch.zeros(B, logvs[0].size(-1), device=device)
                lv = torch.full_like(mu, fill_value=+10.0) 
            mus.append(mu)
            logvs.append(lv)
            
        # 4) gating: compute per-expert variance mean over dim
        variance = [torch.exp(lv) for lv in logvs] # list of (B, d)
        var_stack = torch.stack(variance, dim=1)  # (B, E, d)
        var_mean = var_stack.mean(dim=-1) # (B, E)
        # use avg_sh (full shared mu_all) as gating input
        gate_w = self.gate(mu_all, var_mean)
        self._last_gate_w = gate_w
        
        # 5) PoE fusion with precision weighting
        # weight each expert mu by gate_w * precision
        precisions = 1.0 / (var_mean + 1e-8)           # (B, E)
        weights = gate_w * precisions                  # (B, E)
        # normalize weights to sum=1
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # (B, E)
        # produce fused mu, lv via weighted PoE
        weighted_mus  = [w.unsqueeze(-1) * mu for w,mu in zip(weights.T, mus)]
        fused_mu, fused_lv = product_of_experts(weighted_mus, logvs)

        return fused_mu, fused_lv
    
    def gate_loss(self):
        """
        CV^2 load-balance regularization on the last gate weights.
        """
        if self._last_gate_w is None:
            return torch.tensor(0., device=next(self.parameters()).device)
        g = self._last_gate_w                           # (B, E)
        mean = g.mean(dim=0)                            # (E,)
        var  = g.var(dim=0, unbiased=False)             # (E,)
        cv2  = (var / (mean**2 + 1e-8)).mean()
        return cv2

# ---------- Final Model ----------
class MMDisentangled(nn.Module):
    def __init__(self, input_dim: dict, d_model=256,
                 latent_shared=128, latent_specific=128,
                 n_layers=1, num_classes=2):
        super().__init__()
        #self.modalities = ['audio','vision','text'] # not ['a','v','t']
        self.modalities = list(input_dim.keys()) 
        
        self.encoders = nn.ModuleDict({m: DisentangledModalEncoder(input_dim[m], d_model,
                                                                 latent_shared, latent_specific, n_layers)
                                       for m in self.modalities})
        self.pool = MoEExpertPool(latent_shared)
        self.classifier = nn.Linear(latent_shared, num_classes)
    def forward(self, inputs):
        # inputs: list m -> Tensor(B, T, D)
        # 5, 768, 20 -> unsqueexze(1) (B, D) -> (B, 1, D)
        
        if isinstance(inputs, (list, tuple)):
            inputs = {m: inputs[i] for i, m in enumerate(self.modalities) if i < len(inputs)}
        
        reps_sh, reps_sp = {}, {}
        outs = {}
        present = []
        for m in self.modalities:
            if m in inputs:
                o = self.encoders[m](inputs[m])
                outs[m]       = o
                reps_sh[m]    = o['rep_shared']
                reps_sp[m]    = o['rep_specific']
                present.append(m)
        mu_fused, lv_fused = self.pool(reps_sh, reps_sp, present)
        z = reparameterize(mu_fused, lv_fused)
        logits = self.classifier(z)
        # now return outs as well
        return logits, mu_fused, lv_fused, outs