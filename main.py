import os
import torch
import numpy as np
import argparse
import random
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from copy import deepcopy
from tqdm import trange
from model_modified import MMDisentangled
from utils import seed_everything, setup_logger
from data import load_and_preprocess_data, load_and_preprocess_data_mimic, create_loaders, process_2d_to_3d
from dataset import load_and_preprocess_data_mosi, load_and_preprocess_data_enrico
import logging
import wandb
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

wandb.login(key="94280b6b804c2e90f0d865715d73580707493ccc") #API Key is in your wandb account, under settings (wandb.ai/settings)

# os.chdir(os.getcwd() + '/moe_retriever_final')
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
    parser.add_argument('--num_candidates', type=int, default=10) # Number of candidates in each modality bank
    parser.add_argument('--num_candidates_shared', type=int, default=2) # Number of candidates in each modality bank
    parser.add_argument('--num_supporting_samples', type=int, default=4) # Number of supporting samples
    parser.add_argument('--latent_dim_shared', type=int, default=64)
    parser.add_argument('--latent_dim_specific', type=int, default=32)
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction model
    parser.add_argument('--num_heads', type=int, default=4) # Number of attention heads
    parser.add_argument('--num_workers', type=int, default=1) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities
    parser.add_argument('--patch', type=str2bool, default=True) # Use common ids across modalities
    parser.add_argument('--num_patches', type=int, default=16) # Use common ids across modalities
    parser.add_argument('--num_experts', type=int, default=16) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    #parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--noise_level', type=float, default=0.1) # 0.1 0.3 0.5
    parser.add_argument('--fusion_sparse', type=str2bool, default=False) # Whether to include SMoE in Fusion Layer
    parser.add_argument('--top_k', type=int, default=2) # 4 for adni 3 for mimic # Number of k
    parser.add_argument('--dropout', type=float, default=0.5) # Dropout rates
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--save', type=str2bool, default=False) # Use common ids across modalities
    # Flex-MoE와 동일 setting
    
    return parser.parse_known_args()


def run_epoch(args, loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False, optimizer=None, gate_loss_weight=0.0):
    all_preds = []
    all_labels = []
    all_probs = []
    task_losses = []
    gate_losses = []
    
    if is_training:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

    for batch_samples, batch_labels, batch_mcs, batch_observed in loader:
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_mcs = batch_mcs.to(device, non_blocking=True)
        batch_observed = batch_observed.to(device, non_blocking=True)
        
        fusion_input = []
        for i, (modality, samples) in enumerate(batch_samples.items()):
            mask = batch_observed[:, modality_dict[modality]]
            encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])
            if (~mask).sum() > 0:
                encoded_samples[~mask] = missing_embeds[batch_mcs[~mask], modality_dict[modality]]
            fusion_input.append(encoded_samples)

        outputs = fusion_model(*fusion_input, expert_indices=batch_mcs)

        if is_training:
            optimizer.zero_grad()
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())
            gate_loss = fusion_model.gate_loss()
            gate_losses.append(float(gate_loss))
            loss = task_loss + gate_loss_weight * gate_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            optimizer.step()
            
        else:
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            #all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())
            if args.data == 'adni':
                all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())
            else:
                all_probs.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy())                        
            

    if is_training:
        return task_losses, gate_losses
    else:
        return all_preds, all_labels, all_probs


def train_and_evaluate(args, seed, save_path=None):
    wandb.init(
        name = "modified-AVT", ## Wandb creates random run names if you skip this field
        reinit = True, ### Allows reinitalizing runs when you re-run this cell
        # run_id = ### Insert specific run id here if you want to resume a previous run
        # resume = "must" ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "MoE-Disentangled", ### Project should be created in your wandb account
        config = args
        # {
        #     'epochs': args.train_epochs,
        #     'learning_rate': args.lr
        #     #model_kwargs ### Wandb Config for your run
        # }
    )
    seed_everything(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    if args.data == 'adni':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data(args)
    elif args.data == 'mimic':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data_mimic(args)
    elif args.data == 'mosi' or args.data == 'mosei':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data_mosi(args)
    elif args.data == 'enrico':
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, num_mc_dict = load_and_preprocess_data_enrico(args)

    # print(input_dims) # {'vision': 20, 'audio': 5, 'text': 768}
    # print(encoder_dict) # Architecture
    # print(data_dict) # {'vision': [[...]], ...}
    train_loader, val_loader, test_loader = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.use_common_ids, args.data)
    #num_experts_retriever = (args.num_candidates * num_modalities) + args.num_candidates_shared
    
    best_val_acc = 0.0
    missing_tokens = {}
    
    if args.data == 'adni':
        modality_dict = {'image':0, 'genomic': 1, 'clinical': 2, 'biospecimen': 3}
        char_to_modality = {'I': 'image', 'G': 'genomic', 'C': 'clinical', 'B': 'biospecimen'}

    elif args.data == 'mimic':
        modality_dict = {'lab':0, 'note': 1, 'code': 2}
        char_to_modality = {'L': 'lab', 'N': 'note', 'C': 'code'}
    
    elif args.data == 'mosi' or args.data == 'mosei':
        modality_dict = {'vision':0, 'audio': 1, 'text': 2}
        char_to_modality = {'V': 'vision', 'A': 'audio', 'T': 'text'}
    
    elif args.data == 'enrico':
        modality_dict = {'screenshot': 0, 'wireframe': 1}
        char_to_modality = {'S': 'screenshot', 'W': 'wireframe'}
        
    for modality in modality_dict.keys():
        # 만약 input_dims에 modality가 없다면 기본값 사용 (여기서는 임의의 기본값)
        #dim = input_dims.get(modality, {'vision':20, 'audio':5, 'text':768}[modality])
        #missing_tokens[modality] = torch.nn.Parameter(torch.randn(1, dim), requires_grad=True).to(device)
        missing_tokens = {modality: torch.nn.Parameter(torch.randn(1, input_dims[modality]), requires_grad=True).to(device)}
        
    missing_tokens = {modality: torch.nn.Parameter(torch.randn(1, input_dims[modality], device=device), requires_grad=True)
                  for modality in input_dims.keys()}
        
    # Define latent dimensions for shared and specific representations
    latent_dim_shared = args.latent_dim_shared   # e.g., 64
    latent_dim_specific = args.latent_dim_specific  # e.g., 32
    #modality_keys = sorted(modality_dict.keys()) -> a, t, v
    num_modalities = len(args.modality)
    
    #modality_feature_dims = list(map(int, args.target_dims.split(',')))
    
    #for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
        #print(f"Batch size: {batch_samples[list(batch_samples.keys())[0]].shape[0]}")


    #retriever_model = MMDisentangled(num_modalities, args.num_patches, args.hidden_dim, num_experts_retriever, args.num_routers, args.top_k, args.num_heads, args.dropout).to(device)
    #fusion_model = FusionLayer(num_modalities, args.num_patches, args.hidden_dim, n_labels, args.num_layers_fus, args.num_layers_pred, args.num_experts, args.num_routers, args.top_k, args.num_heads, args.dropout, args.fusion_sparse).to(device)
    model = MMDisentangled(modality_feature_dims=input_dims, latent_dim_shared=latent_dim_shared, latent_dim_specific=latent_dim_specific, num_classes=n_labels).to(device)
    params = list(model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()] + list(missing_tokens.values())
    
    # if num_modalities >= 1:
    #     missing_embeds = torch.nn.Parameter(torch.randn((2**num_modalities)-1, args.n_full_modalities, args.num_patches, args.hidden_dim, dtype=torch.float, device=device), requires_grad=True)
    #     params += [missing_embeds]

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    #criterion = torch.nn.CrossEntropyLoss() if args.data == 'adni' else torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))
    criterion = torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device)) if args.data == 'mimic' else torch.nn.CrossEntropyLoss()
    
    wandb.watch(model, criterion, log="all")

    # mc_idx_dict_train = {k: list(set(v) & set(train_ids)) for k, v in mc_idx_dict.items()}
    # mc_idx_dict_valid = {k: list(set(v) & set(train_ids+valid_ids)) for k, v in mc_idx_dict.items()}
    # mc_idx_dict_test = {k: list(set(v) & set(train_ids+test_ids)) for k, v in mc_idx_dict.items()}


    for epoch in trange(args.train_epochs, desc='Training Epochs'):
        # if epoch < args.warm_up_epochs:
        #     moe_disentangled=False
        #     fusion_model.train()

        # else:
            # moe_disentangled=True
            # if epoch == args.warm_up_epochs:
            #     optimizer.add_param_group({'params': retriever_model.parameters()})  # Add new parameters to existing optimizer
            #fusion_model.train()
            #retriever_model.train()
        model.train()
        task_loss = 0.0
        gate_loss = 0.0

        for encoder in encoder_dict.values():
            encoder.train()
        
        task_losses = []
        gate_losses = []

        for batch_samples, batch_labels, batch_mcs, batch_observed in train_loader:
            batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            missing_rates = {'vision': args.missing_rate_v, 'audio': args.missing_rate_a, 'text': args.missing_rate_t}

            fusion_input = []
            # for modality, samples in batch_samples.items():
            #     mask = batch_observed[:, modality_dict[modality]]
            #     encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
                
            #     if mask.sum() > 0:
            #         encoded_samples[mask] = encoder_dict[modality](samples[mask])
                    
            #     if moe_disentangled:
            #         missing_sample_indices = (~mask).nonzero(as_tuple=False).flatten().cpu().numpy()
            #         if len(missing_sample_indices) > 0:
            #             ### MoE-Retriever ###
            #             for missing_idx in missing_sample_indices:
            #                 mc_num = batch_mcs[missing_idx].item()
            #                 observed_modalities_char = num_mc_dict[mc_num]
            #                 observed_modalities = [char_to_modality[char] for char in observed_modalities_char]

            #                 inter_modal_length = len(observed_modalities) # args.num_supporting_samples 
            #                 supporting_group_indices = get_supporting_group(missing_modality=modality[0].upper(), observed_modalities=observed_modalities_char, mc_idx_dict=mc_idx_dict_train, num_samples=inter_modal_length)
            #                 input_arr = data_dict[modality][supporting_group_indices]
            #                 input_tensor = torch.tensor(input_arr, dtype=torch.float32).to(device)
                            
            #                 intra_embeds = encoder_dict[modality](input_tensor)
            #                 B_intra, N_intra, D = intra_embeds.shape
            #                 intra_embeds = intra_embeds.view(1, B_intra * N_intra, D)
                            
            #                 inter_embeds = []
            #                 for obs_mod in observed_modalities:
            #                     emb = encoder_dict[obs_mod](batch_samples[obs_mod][missing_idx].unsqueeze(0))
            #                     inter_embeds.append(emb)
            #                 moe_disentangled_input = torch.cat([intra_embeds] + inter_embeds, dim=1)
                            
            #                 num_intra = B_intra * N_intra
            #                 num_inter = len(inter_embeds)
                            
            #                 expert_features, confidence, fused_features = retriever_model(moe_disentangled_input, expert_indices=8, num_intra = num_intra, num_inter=num_inter, inter_weight=0.5)
            #                 fused_rep = fused_features.unsqueeze(1).repeat(1, args.num_patches, 1)
            #                 encoded_samples[missing_idx] = fused_rep
            
                
                #print(encoder_dict[modality])  # 모델 구조 확인
                #print(samples[mask].shape)  # 입력 데이터 크기 확인
                
            for modality in modality_dict.keys():  # modality_keys is e.g., sorted(encoder_dict.keys()), model.modalities
                # Get observation mask for this modality.
                # Here we assume batch_observed's columns follow the order of modality_keys.
                
                #modality_idx = modality_dict.keys().index(modality)
                batch_size = batch_labels.shape[0]
                if modality in batch_samples:
                    modality_input = batch_samples[modality].clone()
                    rand_mask = (torch.rand(batch_size, device=device) < missing_rates[modality])
                    # 만약 어떤 샘플이 missing이면, 해당 인덱스에 대해 learnable missing token으로 대체
                    if rand_mask.any():
                        missing_tensor = missing_tokens[modality].expand(rand_mask.sum().item(), -1)
                        modality_input[rand_mask] = missing_tensor
                    #mask = batch_observed[:, modality_dict[modality]].bool()
                    # if mask.shape[0] != modality_input.shape[0]:
                    #     # 만약 모달리티 데이터가 patch-level tensor라면, 적절히 reshape 필요
                    #     mask = mask.unsqueeze(-1)
                    # if (~mask).any():
                    #     # missing sample에 대해 missing token 사용
                    #     num_missing = (~mask).sum().item()
                    #     missing_tensor = missing_tokens[modality].expand(num_missing, -1)
                    #     modality_input[~mask] = missing_tensor
                else:
                    # 해당 모달리티가 완전히 missing한 경우, batch_size x input_dims[modality] 크기의 0 텐서 생성
                    #modality_input = torch.zeros((batch_size, input_dims[modality]), device=device)
                    modality_input = missing_tokens[modality].expand(batch_size, -1)
                    
                fusion_input.append(modality_input)
                
                
                # x = batch_samples[modality]  # (B, feature_dim)
                # mask = batch_observed[:, modality_dict[modality]].unsqueeze(1)  # (B, 1), 모달리티 존재 여부
                # x = x * mask  # missing modality인 경우 0으로 처리
                # fusion_input.append(x)
                    
                '''
                for modality in encoder_dict:
                    print(f"Modality: {modality}")
                    print(f"Expected input size: {encoder_dict[modality][0].gru.input_size}")
                    print(f"Actual input size: {samples[mask].shape[-1]}")
                '''

                #fusion_input.append(encoded_samples)
                #print(fusion_input)

            outputs =  model(fusion_input)
            
            #print("Batch labels:", batch_labels)
            #print("Unique labels:", batch_labels.unique())
            ### print(outputs)
            '''
            (tensor([[ 0.0176],
            [ 0.0199],
            [-0.0089],
            [ 0.0027],
            [ 0.0267],
            [ 0.0176],
            [ 0.0260],
            [ 0.0284]], device='cuda:0', grad_fn=<MulBackward0>), tensor([[-0.4449],
            [-0.4257],
            [-0.4228],
            [-0.4244],
            [-0.4159],
            [-0.4358],
            [-0.4387],
            [-0.4279]], device='cuda:0', grad_fn=<LogBackward0>))
            '''
            task_loss = criterion(outputs[0], batch_labels) # outputs
            task_losses.append(task_loss.item())
            #gate_loss = fusion_model.gate_loss()
            gate_loss = model.transformer_moe.gate_loss() if hasattr(model.transformer_moe, 'gate_loss') else 0.0
            gate_losses.append(float(gate_loss))
            loss = task_loss + args.gate_loss_weight * gate_loss
            
            # print(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        wandb.log({
            "epoch": epoch+1,
            "Train task loss": sum(task_losses)/len(task_losses),
            "Train router_loss": sum(gate_losses)/len(gate_losses),
        })
        
        if epoch < args.warm_up_epochs:
            print(f"[Seed {seed}/{args.n_runs-1}] [Warm-Up Epoch {epoch+1}/{args.warm_up_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f}")
        else:
            #fusion_model.eval()
            #retriever_model.eval()
            model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            
            val_preds = []
            val_labels = []
            val_probs = []
            
            # Need to update!!
            with torch.no_grad(): # Validation
                for batch_samples, batch_labels, batch_mcs, batch_observed in val_loader:
                    batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
                    batch_labels = batch_labels.to(device, non_blocking=True)
                    batch_mcs = batch_mcs.to(device, non_blocking=True)
                    batch_observed = batch_observed.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    
                    fusion_input = []
                    #
                    for modality in modality_dict.keys():  # modality_keys is e.g., sorted(encoder_dict.keys())
                        # Get observation mask for this modality.
                        # Here we assume batch_observed's columns follow the order of modality_keys.
                        #modality_idx = modality_dict.keys().index(modality)
                        if modality in batch_samples:
                            modality_input = batch_samples[modality].clone()
                            mask = batch_observed[:, modality_dict[modality]].bool()
                            
                            if (~mask).any():
                                modality_input[~mask] = torch.zeros_like(modality_input[~mask])
                        else:
                            # if missing: batch_size x input_dims[modality] 크기의 learning token 생성
                            batch_size = batch_labels.shape[0]
                            if modality in input_dims:
                                mod_dim = input_dims[modality]
                            else:
                                mod_dim = input_dims.get(modality)
                                if mod_dim is None:
                                    raise ValueError(f"Modality {modality} is missing and no default dimension is provided.")
                            #modality_input = torch.zeros((batch_size, input_dims[modality]), device=device)
                            
                        fusion_input.append(modality_input)
                        
                        '''
                        for modality in encoder_dict:
                            print(f"Modality: {modality}")
                            print(f"Expected input size: {encoder_dict[modality][0].gru.input_size}")
                            print(f"Actual input size: {samples[mask].shape[-1]}")
                        '''

                        #fusion_input.append(encoded_samples)
                        #print(fusion_input)

                    outputs =  model(fusion_input)


                    _, preds = torch.max(outputs[0], 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
                    if args.data == 'adni':
                        val_probs.extend(torch.nn.functional.softmax(outputs[0], dim=1).cpu().numpy())
                    else:
                        val_probs.extend(torch.nn.functional.softmax(outputs[0], dim=1)[:, 1].cpu().numpy())                        
                        
            #print(np.array(val_probs).shape) # (229, 2)
            #print(np.array(val_labels).shape)  # (229,) v 또는 (229, 1)인지 확인
            #print(outputs.shape)  # torch.Size([5, 2]), (batch_size, num_classes)인지 확인


            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='macro') # unweighted
            val_auc = roc_auc_score(val_labels, val_probs) if args.data == 'mimic' or 'mosi' or 'mosei' else roc_auc_score(val_labels, val_probs, multi_class='ovr')
            # val_auc = roc_auc_score(val_labels, np.array(val_probs)[:, 1]) if args.data == 'mimic' else roc_auc_score(val_labels, np.array(val_probs), multi_class='ovr')

            print(f"[Seed {seed}/{args.n_runs-1}] [Epoch {epoch+1}/{args.train_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f} / Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
            
            wandb.log({
                "epoch": epoch +1,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_auc": val_auc
            })


            if val_acc > best_val_acc:
                print(f" [(**Best**) Epoch {epoch+1}/{args.train_epochs}] Val Acc: {val_acc*100:.2f}, Val F1: {val_f1*100:.2f}, Val AUC: {val_auc*100:.2f}")
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_val_auc = val_auc
                best_model = deepcopy(model.state_dict())
                best_model_enc = {modality: deepcopy(encoder.state_dict()) for modality, encoder in encoder_dict.items()}
                #best_model_retriever = deepcopy(retriever_model.state_dict())

                # Move the models to CPU for saving (only state_dict)
                if args.save:
                    best_model_cpu = {k: v.cpu() for k, v in best_model.items()}
                    best_model_enc_cpu = {modality: {k: v.cpu() for k, v in enc_state.items()} for modality, enc_state in best_model_enc.items()}
                    #best_model_retriever_cpu = {k: v.cpu() for k, v in best_model_retriever.items()}

    # Save the best model
    if args.save:
        save_path = f'./saves/best_data_{args.data}_seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}_val_acc_{best_val_acc*100:.2f}.pth'
        torch.save({
            'fusion_model': best_model_cpu,
            'encoder_dict': best_model_enc_cpu
        }, save_path)
        wandb.save(save_path)

        print(f"Best model saved to {save_path}")
    
    # Load best model for test evaluation
    for modality, encoder in encoder_dict.items():
        encoder.load_state_dict(best_model_enc[modality])
        encoder.eval()
    #retriever_model.load_state_dict(best_model_retriever)
    #retriever_model.eval()
    model.load_state_dict(best_model)
    model.eval()
        
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch_samples, batch_labels, batch_mcs, batch_observed in test_loader:
            batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_mcs = batch_mcs.to(device, non_blocking=True)
            batch_observed = batch_observed.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            fusion_input = []
            for modality in modality_dict.keys():  # modality_keys is e.g., sorted(encoder_dict.keys())
                # Get observation mask for this modality.
                # Here we assume batch_observed's columns follow the order of modality_keys.
                #modality_idx = modality_dict.keys().index(modality)
                if modality in batch_samples:
                    modality_input = batch_samples[modality].clone()
                    mask = batch_observed[:, modality_dict[modality]].bool()
                    
                    if (~mask).any():
                        modality_input[~mask] = torch.zeros_like(modality_input[~mask])
                else:
                    # 해당 모달리티가 완전히 missing한 경우, batch_size x input_dims[modality] 크기의 0 텐서 생성
                    batch_size = batch_labels.shape[0]
                    modality_input = torch.zeros((batch_size, input_dims[modality]), device=device)
                    
                fusion_input.append(modality_input)
                '''
                for modality in encoder_dict:
                    print(f"Modality: {modality}")
                    print(f"Expected input size: {encoder_dict[modality][0].gru.input_size}")
                    print(f"Actual input size: {samples[mask].shape[-1]}")
                '''

                #fusion_input.append(encoded_samples)
                #print(fusion_input)

            outputs =  model(fusion_input)

            _, preds = torch.max(outputs[0], 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(batch_labels.cpu().numpy())
            if args.data == 'adni':
                test_probs.extend(torch.nn.functional.softmax(outputs[0], dim=1).cpu().numpy())
            else:
                test_probs.extend(torch.nn.functional.softmax(outputs[0], dim=1)[:, 1].cpu().numpy())  

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    #test_auc = roc_auc_score(test_labels, test_probs) if args.data == 'mimic' else roc_auc_score(test_labels, test_probs, multi_class='ovr')
    test_auc = roc_auc_score(test_labels, test_probs) if args.data == 'mimic' or 'mosi' or 'mosei' else roc_auc_score(test_labels, test_probs, multi_class='ovr')
    wandb.log({
                "epoch": epoch +1,
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
        "model": 'MoE-Disentangled-Re',
        "modality": args.modality,
        "missing_rate_a": args.missing_rate_a,
        "missing_rate_v": args.missing_rate_v,
        "missing_rate_t": args.missing_rate_t,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "num_candidates": args.num_candidates,
        "num_candidates_shared": args.num_candidates_shared,
        "latent_dim_shared": args.latent_dim_shared,
        "latent_dim_specific": args.latent_dim_specific,
        "num_supporting_samples": args.num_supporting_samples,
        "fusion_sparse": args.fusion_sparse,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
        "noise_level": args.noise_level
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