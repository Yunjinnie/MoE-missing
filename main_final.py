import os
import torch
import torch.nn.functional as F
from torch import nn, optim
import argparse
import random
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from tqdm import trange
from itertools import combinations
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from model_final import MMDisentangled, jsd_loss, orth_loss, info_nce, club_upper_bound, CondCritic, Critic
from utils import seed_everything, setup_logger
from data import load_and_preprocess_data, load_and_preprocess_data_mimic, create_loaders, process_2d_to_3d
from dataset import load_and_preprocess_data_mosi, load_and_preprocess_data_enrico
import logging
import wandb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

wandb.login(key="94280b6b804c2e90f0d865715d73580707493ccc")

# Utility function to convert string to bool
def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='MoE-Disentangled-Re')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='mosi') # adni
    parser.add_argument('--modality', type=str, default='AVT') # I G C B for ADNI, L N C for MIMIC
    parser.add_argument('--missing_rate_a', type=float, default=0.0)
    parser.add_argument('--missing_rate_v', type=float, default=0.0)
    parser.add_argument('--missing_rate_t', type=float, default=0.0)
    parser.add_argument('--w_disent', type=float, default=1e-3, help='weight for disentanglement losses (JSD, orth, MI)') # 1.0, 1e-3
    parser.add_argument('--w_gate', type=float, default=1e-1, help='weight for gate regularization loss') # 1e-2, 1e-1
    parser.add_argument('--w_jsd', type=float, default=1e-2)
    parser.add_argument('--w_orth', type=float, default=1e-2)
    parser.add_argument('--w_shared', type=float, default=1e-2)
    parser.add_argument('--w_unique', type=float, default=1e-2)
    parser.add_argument('--latent_shared', type=int, default=256)
    parser.add_argument('--latent_specific', type=int, default=256)
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4) # 1e-5
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction model
    parser.add_argument('--num_heads', type=int, default=4) # Number of attention heads
    parser.add_argument('--num_workers', type=int, default=1) # Number of workers for DataLoader
    parser.add_argument('--patch', type=str2bool, default=True) # Use common ids across modalities
    parser.add_argument('--num_patches', type=int, default=16) # Use common ids across modalities
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities
    parser.add_argument('--num_experts', type=int, default=7) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--noise_level', type=float, default=0.1) # 0.1 0.3 0.5
    parser.add_argument('--grad_clip', type=float, default=1.0, help='gradient clipping max norm') # 1.0 0.5 2.0
    parser.add_argument('--top_k', type=int, default=2) # 4 for adni 3 for mimic # Number of k
    parser.add_argument('--dropout', type=float, default=0.5) # Dropout rates
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--save', type=str2bool, default=False) # Use common ids across modalities
    # Flex-MoE와 동일 setting
    
    return parser.parse_known_args()

def compute_disentangle_mi_loss(shared_reps, specific_reps, mus_shared, mus_specific, y_onehot, critics, device):
    """
    shared_reps: list of Tensor [(B,d),...], M modalities
    specific_reps: list of Tensor [(B,d),...]
    mus_shared: list of Tensor [(B,d),...]  # mu for shared
    mus_specific: list of Tensor [(B,d),...]
    y_onehot: Tensor (B,C)
    critics: dict with
        'shared':      Critic(d)
        'shared_cond': CondCritic(d, C)
        'unique':      [Critic(d+C) for _ in modalities]
    returns:
        L_jsd,
        L_orth,
        L_info_s,
        L_club_s_cond,
        L_shared,   # = L_info_s - L_club_s_cond
        L_unique    # = (L_club_s - L_club_s_cond) - Σ_i L_info_u(i)
    """
    M = len(shared_reps)
    pairs = list(combinations(range(M), 2))
    
    if len(pairs) == 0: # modality 1개
        # No pairs, return zero losses
        L_orth = sum(orth_loss(shared_reps[i], specific_reps[i]) for i in range(M)) / M
        y_emb = critics['shared_cond'].embed_y(y_onehot) # (B, d)
        L_info_u_list = []
        for i in range(M):
            # embed Y
            L_info_u_list.append(info_nce(mus_specific[i], y_emb, critics['unique'][i]))
        L_unique = sum(L_info_u_list) / M
        L_unique = F.softplus(L_unique)
        zero = torch.tensor(0., device=device)
        return zero, L_orth, zero, L_unique
    
    # 1. JSD on shared reps
    L_jsd = sum(jsd_loss(shared_reps[i], shared_reps[j]) for i,j in pairs) / len(pairs)
    # 2. Orthogonality on shared & specific reps
    L_orth = sum(orth_loss(shared_reps[i], specific_reps[i]) for i in range(M)) / M
    # 3. Shared MI
    # InfoNCE lower bound
    L_info_s = sum(info_nce(mus_shared[i], mus_shared[j], critics['shared']) for i,j in pairs) / len(pairs)
    # CLUB upper bound conditional on Y
    L_club_s_cond = sum(
        club_upper_bound(mus_shared[i], mus_shared[j],
                         lambda a,b: critics['shared_cond'](a,b,y_onehot))
        for i,j in pairs) / len(pairs)
    L_shared = L_club_s_cond - L_info_s
    # 4. Unique MI
    L_club_s = sum(club_upper_bound(mus_shared[i], mus_shared[j], critics['shared'])
                   for i,j in pairs) / len(pairs)
    
    y_emb = critics['shared_cond'].embed_y(y_onehot) # (B, d)
    delta_club = L_club_s - L_club_s_cond # 불필요 정보 제거
    L_info_u_list = []
    for i in range(M):
        # embed Y
        L_info_u_list.append(info_nce(mus_specific[i], y_emb, critics['unique'][i]))
    L_unique = sum(L_info_u_list) / M - delta_club 
    
    margin = 0.1
    beta = 2.0
    #L_shared = (1.0 / beta) * torch.log1p(torch.exp(beta * L_shared))
    #L_unique = (1.0 / beta) * torch.log1p(torch.exp(beta * L_unique))
    #L_shared = torch.relu(L_shared + margin)
    L_shared = F.softplus(L_shared) #torch.clamp(L_shared, min=0.0)
    #L_unique = torch.relu(L_unique + margin) 
    L_unique = F.softplus(L_unique) #torch.clamp(L_unique, min=0.0)

    return L_jsd, L_orth, L_shared, L_unique

def train_one_epoch(model, dataloader, optimizer, criterion, critics, device, args):        
    model.train()
    total_losses = [] # 0.0
    for batch_samples, batch_labels, batch_mcs, batch_observed in dataloader:
        # batch_samples: dict modality->Tensor
        #print(batch_samples.keys()) # dict_keys(['vision', 'audio', 'text'])
        # batch_labels: Tensor
        
        for k in batch_samples: 
            batch_samples[k] = batch_samples[k].to(device)
        labels = batch_labels.to(device)
        
        #input_batch = {m: batch_samples[char_to_modality[m]] for m in model.modalities}
        input_batch = {m: batch_samples[m] for m in model.modalities}
        
        # forward
        logits, mu_fused, lv_fused, outs = model(input_batch)
        
        # -------------------
        # 1) CE loss
        L_ce = criterion(logits, labels)
        # 2) Disentangle & MI losses
        # gather per-modality outputs from model.encoders[*] if you stored them
        # here we assume model returns lists? adapt if not
        # 2.1) shared & specific reps, mu        
        present = list(outs.keys())
        
        shared_reps =    [ outs[m]['rep_shared']   for m in present ]
        specific_reps =  [ outs[m]['rep_specific'] for m in present ]
        mus_s =          [ outs[m]['mu_shared']     for m in present ]
        mus_u =          [ outs[m]['mu_specific']   for m in present ]
        # (2) per‐modality precision τ_i 계산
        # logvar_shared 은 outs[m]['logvar_shared']
        logvars_s = [outs[m]['logvar_shared'] for m in present]     # list of (B, d)
        logvars_u = [outs[m]['logvar_specific'] for m in present]
        
        # sigma² = exp(logvar), so precision = 1/sigma²
        y_onehot = F.one_hot(labels, num_classes=args.num_classes).float()
        
        M = len(shared_reps)
        pairs = list(combinations(range(M), 2))
        
        # for m in shared_reps:
        #     print(m.mean().item(), m.std().item())
        # for m in specific_reps:
        #     print(m.mean().item(), m.std().item())
        # mean ~ 0, std ~ 1
        
        # raw losses
        L_jsd, L_orth, L_shared, L_unique = compute_disentangle_mi_loss(
            shared_reps, specific_reps, mus_s, mus_u, y_onehot, critics, device)
        
        # precision calculation - pairwise expert precisions: average over dimensions
        
        tau_shared = []  # list of (B,) tensors
        # for (i,j), mu_ij, lv_ij in zip(pairs, mus_pair, logvars_pair):
        #     sigma_ij = torch.exp(0.5 * lv_ij)           # (B, d)
        #     tau_ij   = 1.0 / (sigma_ij**2 + 1e-8)       # (B, d)
        #     tau_shared_pair.append(tau_ij.mean(dim=1))  # (B,)
        for lv in logvars_s:
            sigma2 = torch.exp(lv)           # (B, d)
            tau_i  = 1.0 / (sigma2 + 1e-8)   # (B, d)
            tau_shared.append(tau_i.mean(dim=1))  # (B,) 평균 내서 scalar precision
        
        tau_shared = torch.stack(tau_shared, dim=1)  # (B, M)
        
        # (3) pairwise precision τ_{i,j} = (τ_i + τ_j)/2
        tau_shared_pair = []
        for i, j in pairs:
            tau_pair = 0.5 * (tau_shared[:, i] + tau_shared[:, j])  # (B,)
            tau_shared_pair.append(tau_pair)

        if len(tau_shared_pair) == 0:
            # No pairs, return zero losses
            tau_shared_pair = torch.zeros_like(tau_shared[:, 0])
        else:
            tau_shared_pair = torch.stack(tau_shared_pair, dim=1)  # (B, #pairs)
        
        # (4) shared precision τ_{shared} = 1/M Σ_i τ_i
        tau_specific = []
        # for m in args.modality:
        #     sigma_u = outs[m]['sigma_specific']           # (B, d)
        #     tau_m   = 1.0 / (sigma_u**2 + 1e-8)           # (B, d)
        #     tau_specific.append(tau_m.mean(dim=1))        # (B,)
            
        for lv in logvars_u:
            sigma2 = torch.exp(lv)
            tau_i  = 1.0 / (sigma2 + 1e-8)
            tau_specific.append(tau_i.mean(dim=1))  # (B,)
        tau_specific = torch.stack(tau_specific, dim=1)  # (B, M)
        
        # (3) precision 으로 가중 AVG
        # Reduce to scalar by mean over batch, sum over experts
        # tau_shared_pair 를 .mean() 해서 전체 scalar weight w_shared 로 쓰거나
        # per‐pair weight vector로 그대로 가중합해도...
        if len(pairs) == 0:
        # No pairs, all‐zero precision weight
            w_shared = 0.0
        else:
            w_shared = tau_shared_pair.mean()   # scalar
        w_unique = tau_specific.mean()      # scalar
        
        ## 고쳐볼 부분: 
        # weight precision 기반 말고 (expert fusion 말고 loss 합칠 때) experimental하게 해보기
        # calibration
        # expert mixture weight 조절 - expert간 confidence 찍어보기
        
        
        # (4) 최종 loss
        #L_shared = w_shared * L_shared
        #L_unique = w_unique * L_unique
        
        #L_disent = args.w_jsd*L_jsd + args.w_orth*L_orth + args.w_shared*L_shared + args.w_unique*L_unique
        L_disent = L_jsd + L_orth + L_shared + L_unique
        
        # 3) Gating loss (inside MoE pool)
        L_gate = model.pool.gate_loss()
        # 4) Total
        loss = L_ce + args.w_disent*L_disent + args.w_gate*L_gate
        
        # print(f"Scaled losses:"
        #     f" jsd={args.w_jsd*L_jsd.item():.4e},"
        #     f" orth={args.w_orth*L_orth.item():.4e},"
        #     f" shared={args.w_shared*L_shared.item():.4e},"
        #     f" unique={args.w_unique*L_unique.item():.4e},"
        #     f" gate={args.w_gate*L_gate.item():.4e}")
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        total_losses.append(loss.item()) #+= loss.item() * labels.size(0)
        
    print(f"CE={L_ce.item():.4f}, JSD={L_jsd.item():.4f}, ORTH={L_orth.item():.4f}, "
        f"SHARED={L_shared.item():.4f}, UNIQUE={L_unique.item():.4f}, GATE={L_gate.item():.4f}")
    avg_total_loss = sum(total_losses)/len(total_losses)
    wandb.log({
            "epoch": args.train_epochs,
            "Train task loss": avg_total_loss
        })
    
    return avg_total_loss

@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()
    preds, targets, probs = [], [], []
    for batch_samples, batch_labels, *_ in dataloader:
        for k in batch_samples: batch_samples[k] = batch_samples[k].to(device)
        labels = batch_labels.to(device)
        logits, _, _, _ = model(batch_samples)
        prob = F.softmax(logits, dim=-1)
        _, pred = logits.max(dim=-1)
        preds.append(pred.cpu().numpy())
        targets.append(labels.cpu().numpy())
        # binary or multi
        if args.num_classes == 2:
            probs.append(prob[:,1].cpu().numpy())
        else:
            probs.append(prob.cpu().numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    if args.num_classes==2:
        probs = np.concatenate(probs)
        auc = roc_auc_score(targets, probs)
    else:
        probs = np.vstack(probs)
        auc = roc_auc_score(targets, probs, multi_class='ovr')
    acc = accuracy_score(targets, preds)
    f1  = f1_score(targets, preds, average='macro')
    return acc, f1, auc

def train_and_evaluate(args, seed, save_path=None):
    wandb.init(
        name = "AVT", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "MoE-Disentangled-Final", ### Project should be created in your wandb account
        config = args
        # {
        #     'epochs': args.train_epochs,
        #     'learning_rate': args.lr
        #     #model_kwargs ### Wandb Config for your run
        # }
    )
    seed_everything(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    # load data
    if args.data == 'adni':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data(args)
    elif args.data == 'mimic':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data_mimic(args)
    elif args.data == 'mosi' or args.data == 'mosei':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data_mosi(args)
        
    # print(input_dims) # {'vision': 20, 'audio': 5, 'text': 768}
    # print(encoder_dict) # Architecture
    # print(data_dict) # {'vision': [[...]], ...}
    train_loader, val_loader, test_loader = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.use_common_ids, args.data)
    num_modalities = len(args.modality)
    
    if args.data == 'adni':
        modality_dict = {'image':0, 'genomic': 1, 'clinical': 2, 'biospecimen': 3}
        char_to_modality = {'I': 'image', 'G': 'genomic', 'C': 'clinical', 'B': 'biospecimen'}

    elif args.data == 'mimic':
        modality_dict = {'lab':0, 'note': 1, 'code': 2}
        char_to_modality = {'L': 'lab', 'N': 'note', 'C': 'code'}
    
    elif args.data == 'mosi' or args.data == 'mosei':
        modality_dict = {'vision':0, 'audio': 1, 'text': 2}
        char_to_modality = {'V': 'vision', 'A': 'audio', 'T': 'text'}
        
    #print(input_dims) # {'vision': 20, 'audio': 5, 'text': 768}
    
    # feat_dims = {
    #     'a': input_dims['audio'],
    #     'v': input_dims['vision'],
    #     't': input_dims['text'],
    # }
        
    # critics for MI
    critics = {
        'shared': Critic(args.latent_shared).to(device),
        'shared_cond': CondCritic(args.latent_shared, args.num_classes).to(device),
        'unique': nn.ModuleList([Critic(args.latent_specific).to(device) for _ in args.modality])
    }
    # model & optim
    model = MMDisentangled(input_dim=input_dims, d_model=256, latent_shared=args.latent_shared, latent_specific=args.latent_specific,
                 n_layers=args.num_layers_enc, num_classes=n_labels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max', factor=0.5, patience=2, verbose=True) # 최대화 지표: validation accuracy
    criterion = nn.CrossEntropyLoss()
    wandb.watch(model, criterion, log="all")
    best_val_acc = 0.0
    save_path = f'./saves/best_data_{args.data}_seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc*100:.2f}.pth'
    

    #for epoch in range(1, args.train_epochs+1):
    for epoch in trange(1, args.train_epochs+1, desc='Training Epochs'):
        # 1) train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, critics, device, args)
        
        if epoch < args.warm_up_epochs:
            print(f"[Seed {seed}/{args.n_runs-1}] [Warm-Up Epoch {epoch}/{args.warm_up_epochs}] Task Loss: {np.mean(train_loss):.2f}")
        else:
            # 2) validate
            val_acc, val_f1, val_auc = evaluate(model, val_loader, device, args)
            scheduler.step(val_acc)
            print(f"Epoch {epoch} | Train Loss {train_loss:.4f} | Val Acc {val_acc:.4f}, F1 {val_f1:.4f}, AUC {val_auc:.4f}")
            
            # 3) checkpoint
            if val_acc > best_val_acc:
                print(f" [(**Best**) Epoch {epoch}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc
                best_model = deepcopy(model.state_dict())
                best_model_enc = {modality: deepcopy(encoder.state_dict()) for modality, encoder in encoder_dict.items()}
                #best_model_retriever = deepcopy(retriever_model.state_dict())

                # Move the models to CPU for saving (only state_dict)
                if args.save:
                    best_model_cpu = {k: v.cpu() for k, v in best_model.items()}
                    #best_model_enc_cpu = {modality: {k: v.cpu() for k, v in enc_state.items()} for modality, enc_state in best_model_enc.items()}
                    #best_model_retriever_cpu = {k: v.cpu() for k, v in best_model_retriever.items()}
    if args.save:
        torch.save(model.state_dict(), args.save_path)
        wandb.save(save_path)
        print(f"Best model saved to {save_path}")

    # test
    model.load_state_dict(best_model) # torch.load(args.save_path)
    test_acc, test_f1, test_auc = evaluate(model, test_loader, device, args)
    print(f"Test | Acc {test_acc:.4f}, F1 {test_f1:.4f}, AUC {test_auc:.4f}")
    
    wandb.log({
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_auc": test_auc
            })
    
    wandb.finish()
    
    return best_val_acc, best_val_f1, best_val_auc, test_acc, test_f1, test_auc


def main():
    args, _ = parse_args()
    logger = setup_logger('./logs', f'{args.data}', f'{args.modality}.txt')
    seeds = np.arange(args.n_runs) # [0, 1, 2]
    val_accs = []
    val_f1s = []
    val_aucs = []
    test_accs = []
    test_f1s = []
    test_aucs = []
    
    log_summary = "======================================================================================\n"
    
    model_kwargs = {
        "model": 'MoE-Disentangled-Final',
        "modality": args.modality,
        "missing_rate_a": args.missing_rate_a,
        "missing_rate_v": args.missing_rate_v,
        "missing_rate_t": args.missing_rate_t,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "latent_shared": args.latent_shared,
        "latent_specific": args.latent_specific,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "num_patches": args.num_patches,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "noise_level": args.noise_level,
        "grad_clip": args.grad_clip
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print('Modality:', args.modality)

    for seed in seeds:
        if (not args.save) & (args.load_model):
            save_path = f'./saves/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth'
        else:
            save_path = None
        val_acc, val_f1, val_auc, test_acc, test_f1, test_auc = train_and_evaluate(args, seed, save_path=save_path)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc)
        test_accs.append(test_acc)
        test_f1s.append(test_f1)
        test_aucs.append(test_auc)
    
    val_avg_acc = np.mean(val_accs)*100
    val_std_acc = np.std(val_accs)*100
    val_avg_f1 = np.mean(val_f1s)*100
    val_std_f1 = np.std(val_f1s)*100
    val_avg_auc = np.mean(val_aucs)*100
    val_std_auc = np.std(val_aucs)*100

    test_avg_acc = np.mean(test_accs)*100
    test_std_acc = np.std(test_accs)*100
    test_avg_f1 = np.mean(test_f1s)*100
    test_std_f1 = np.std(test_f1s)*100
    test_avg_auc = np.mean(test_aucs)*100
    test_std_auc = np.std(test_aucs)*100

    log_summary += f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} '
    log_summary += f'[Val] Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} '
    log_summary += f'[Val] Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f} / '  
    log_summary += f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} '
    log_summary += f'[Test] Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} '
    log_summary += f'[Test] Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f} '  

    print(model_kwargs)
    print(f'[Val] Average Accuracy: {val_avg_acc:.2f} ± {val_std_acc:.2f} / Average F1 Score: {val_avg_f1:.2f} ± {val_std_f1:.2f} / Average AUC: {val_avg_auc:.2f} ± {val_std_auc:.2f}')
    print(f'[Test] Average Accuracy: {test_avg_acc:.2f} ± {test_std_acc:.2f} / Average F1 Score: {test_avg_f1:.2f} ± {test_std_f1:.2f} / Average AUC: {test_avg_auc:.2f} ± {test_std_auc:.2f}')

    logger.info(log_summary)

if __name__ == '__main__':
    main()
