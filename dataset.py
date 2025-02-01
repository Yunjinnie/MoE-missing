import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader
import json
#from models.common_modules import PatchEmbeddings, GRU, VGG11Slim
from models import PatchEmbeddings, GRU, VGG11Slim
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
import pickle
import csv
import random
from PIL import Image


class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, transforms, masks, use_common_ids=True):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = np.array(ids)
        self.labels = np.array(labels)
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.use_common_ids = use_common_ids
        self.data = {modality: [data[i] for i in ids] for modality, data in self.data_dict.items() if 'modality' not in modality} # {modality: np.array(data)[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label = self.labels[ids]
        self.mc = self.mc[ids]
        self.observed = self.observed[ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data.items():
            sample_data[modality] = np.nan_to_num(data[idx])
            if modality == 'image':
                sample_data[modality] = process_2d_to_3d(data, idx, self.masks, self.transforms)

        label = self.label[idx]
        mc = self.mc[idx]
        observed = self.observed[idx]

        return sample_data, label, mc, observed

def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)

    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset

def process_2d_to_3d(data, idx, masks, transforms):
    subj1 = data[idx]
    subj_gm_3d = np.zeros(masks.shape, dtype=np.float32)
    subj_gm_3d.ravel()[masks] = subj1
    subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
    if transforms:
        subj_gm_3d = transforms(subj_gm_3d)
    sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
    output = np.array(sample)

    return output

def load_and_preprocess_data_mosi(args):
    filepath = '/shared/s2/lab01/yunjinna/mosi/aligned_50.pkl' #'data/cmu-mosi/mosi_data.pkl'
    with open(filepath, "rb") as f:
        alldata = pickle.load(f)

    processed_dataset = {'train': {}, 'test': {}, 'valid': {}}

    # alldata['train'].keys(): dict_keys(['vision', 'audio', 'text', 'labels', 'id'])
    # 데이터셋 구성: {'raw_text': array, 'audio': array, 'vision': array, 'id': array, 'text': array, 'text_bert': array, 'annotations': array, 'classification_labels': array, 'regression_labels': array}
    # Audio: 5, 
    alldata['train'] = drop_entry(alldata['train'])
    alldata['valid'] = drop_entry(alldata['valid'])
    alldata['test'] = drop_entry(alldata['test'])
    
    '''
    for key, value in alldata['train'].items():   
        if isinstance(value, (np.ndarray, torch.Tensor)): 
            print(key, value.shape)
            
    supported seed types are: None, int, float, str, bytes, and bytearray.
    random.seed(seed)
    raw_text (1284,)
    audio (1284, 50, 5)
    vision (1284, 50, 20)
    id (1284,)
    text (1284, 50, 768)
    text_bert (1284, 3, 50)
    annotations (1284,)
    classification_labels (1284,)
    regression_labels (1284,)
    '''
    
    # Binary classification!
    train_labels = alldata['train']['regression_labels'].flatten() # classification_labels
    train_labels = np.array([0 if label<=0 else 1 for label in train_labels])
    valid_labels = alldata['valid']['regression_labels'].flatten()
    valid_labels = np.array([0 if label<=0 else 1 for label in valid_labels])
    test_labels = alldata['test']['regression_labels'].flatten()
    test_labels = np.array([0 if label<=0 else 1 for label in test_labels])
    labels = np.concatenate((train_labels, valid_labels, test_labels))
    n_labels = len(set(labels))

    train_ids = alldata['train']["id"]
    val_ids = alldata['valid']["id"]
    test_ids = alldata['test']["id"]

    train_ids = [''.join(list(arr.astype(str))) for arr in train_ids]
    val_ids = [''.join(list(arr.astype(str))) for arr in val_ids]
    test_ids = [''.join(list(arr.astype(str))) for arr in test_ids]

    all_ids = train_ids + val_ids + test_ids

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(all_ids)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0],3), dtype=bool)

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality
    
    # Load modalities
    if 'V' in args.modality or 'v' in args.modality:
        train_vision = alldata['train']['vision']
        train_vision = [np.nan_to_num(train_vision[i]) for i in range(train_vision.shape[0])]
        valid_vision = alldata['valid']['vision']
        valid_vision = [np.nan_to_num(valid_vision[i]) for i in range(valid_vision.shape[0])]
        test_vision = alldata['test']['vision']
        test_vision = [np.nan_to_num(test_vision[i]) for i in range(test_vision.shape[0])]
        
        arr = train_vision + valid_vision + test_vision
        observed_idx_arr[:,0] = [True] * len(arr)
        data_dict['vision'] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            if np.random.random_sample() < args.noise_level:
                data_dict['vision'][idx] = torch.rand(data_dict['vision'][idx].shape[0], data_dict['vision'][idx].shape[1])
                observed_idx_arr[idx, 0] = False
            else:
                update_modality_combinations(idx, 'V')
        encoder_dict['vision'] = torch.nn.Sequential(GRU(20, 256, dropout=True, has_padding=False, batch_first=True, last_only=True).cuda(),
                                                   PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda())
        input_dims['vision'] = arr[0].shape[1]
        # transforms['vision'] = None
        # masks['image'] = mask
    if 'A' in args.modality or 'a' in args.modality:
        train_audio = alldata['train']['audio']
        train_audio = [np.nan_to_num(train_audio[i]) for i in range(train_audio.shape[0])]
        valid_audio = alldata['valid']['audio']
        valid_audio = [np.nan_to_num(valid_audio[i]) for i in range(valid_audio.shape[0])]
        test_audio = alldata['test']['audio']
        test_audio = [np.nan_to_num(test_audio[i]) for i in range(test_audio.shape[0])]
        
        arr = train_audio + valid_audio + test_audio
        observed_idx_arr[:,1] = [True] * len(arr)
        data_dict['audio'] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            if np.random.random_sample() < args.noise_level:
                data_dict['audio'][idx] = torch.rand(data_dict['audio'][idx].shape[0], data_dict['audio'][idx].shape[1])
                observed_idx_arr[idx, 1] = False
            else:
                update_modality_combinations(idx, 'A')
        encoder_dict['audio'] = torch.nn.Sequential(GRU(5, 256, dropout=True,has_padding=False, batch_first=True, last_only=True).cuda(),
                                                   PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda())
        input_dims['audio'] = arr[0].shape[1]
    if 'T' in args.modality or 't' in args.modality:
        train_text = alldata['train']['text']
        train_text = [np.nan_to_num(train_text[i]) for i in range(train_text.shape[0])]
        valid_text = alldata['valid']['text']
        valid_text = [np.nan_to_num(valid_text[i]) for i in range(valid_text.shape[0])]
        test_text = alldata['test']['text']
        test_text = [np.nan_to_num(test_text[i]) for i in range(test_text.shape[0])]
        
        arr = train_text + valid_text + test_text
        observed_idx_arr[:,2] = [True] * len(arr)
        data_dict['text'] = np.array(arr)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)
        for idx in filtered_idx:
            if np.random.random_sample() < args.noise_level:
                data_dict['text'][idx] = torch.rand(data_dict['text'][idx].shape[0], data_dict['text'][idx].shape[1])
                observed_idx_arr[idx, 2] = False
            else:
                update_modality_combinations(idx, 'T')
        encoder_dict['text'] = torch.nn.Sequential(GRU(768, 256, dropout=True, has_padding=False, batch_first=True, last_only=True).cuda(),
                                                   PatchEmbeddings(256, args.num_patches, args.hidden_dim).cuda())
        ## 300에서 계속 에러남
        input_dims['text'] = arr[0].shape[1]
   
    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]

    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in val_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return data_dict['modality_comb'][idx] == -1

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    valid_idxs = [idx for idx in valid_idxs if not all_modalities_missing(idx)]
    test_idxs = [idx for idx in test_idxs if not all_modalities_missing(idx)]

    mc_num_to_mc = {v:k for k,v in combination_to_index.items()}
    mc_idx_dict = {mc_num_to_mc[mc_num]: list(np.where(np.array(data_dict['modality_comb']) == mc_num)[0]) for mc_num in set(data_dict['modality_comb']) if mc_num != -1}

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, mc_num_to_mc

def load_and_preprocess_data_enrico(args):
    data_dir = 'data/enrico/'
    csv_file = os.path.join(data_dir, "design_topics.csv")
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        example_list = list(reader)
    # self.img_missing_rate = img_missing_rate
    # self.wireframe_missing_rate = wireframe_missing_rate
    img_dim_x=128
    img_dim_y=256
    random_seed=42
    train_split=0.65
    val_split=0.15
    test_split=0.2
    normalize_image=False
    seq_len=64
    csv_file = os.path.join(data_dir, "design_topics.csv")
    img_dir = os.path.join(data_dir, "screenshots")
    wireframe_dir = os.path.join(data_dir, "wireframes")
    hierarchy_dir = os.path.join(data_dir, "hierarchies")

    # the wireframe files are corrupted for these files
    IGNORES = set(["50105", "50109"])
    example_list = [
        e for e in example_list if e['screen_id'] not in IGNORES]
    
    keys = list(range(len(example_list)))
    # shuffle and create splits
    random.Random(random_seed).shuffle(keys)


    # train split is at the front
    train_start_index = 0
    train_stop_index = int(len(example_list) * train_split)
    train_keys = keys[train_start_index:train_stop_index]

    # val split is in the middle
    val_start_index = int(len(example_list) * train_split)
    val_stop_index = int(len(example_list) * (train_split + val_split))
    val_keys = keys[val_start_index:val_stop_index]

    # test split is at the end
    test_start_index = int(len(example_list) * (train_split + val_split))
    test_stop_index = len(example_list)
    test_keys = keys[test_start_index:test_stop_index]

    img_transforms = [
            Resize((img_dim_y, img_dim_x)),
            ToTensor()
        ]
    if normalize_image:
        img_transforms.append(Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    img_transforms = Compose(img_transforms)
        
    # make maps
    topics = set()
    for e in example_list:
        topics.add(e['topic'])
    topics = sorted(list(topics))

    idx2Topic = {}
    topic2Idx = {}

    for i in range(len(topics)):
        idx2Topic[i] = topics[i]
        topic2Idx[topics[i]] = i

    idx2Topic = idx2Topic
    topic2Idx = topic2Idx

    UI_TYPES = ["Text", "Text Button", "Icon", "Card", "Drawer", "Web View", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab",
                "Background Image", "Image", "Video", "Input", "Number Stepper", "Checkbox", "Radio Button", "Pager Indicator", "On/Off Switch", "Modal", "Slider", "Advertisement", "Date Picker", "Map View"]

    idx2Label = {}
    label2Idx = {}

    for i in range(len(UI_TYPES)):
        idx2Label[i] = UI_TYPES[i]
        label2Idx[UI_TYPES[i]] = i

    ui_types = UI_TYPES

    def featurizeElement(element):
        """Convert element into tuple of (bounds, one-hot-label)."""
        bounds, label = element
        labelOneHot = [0 for _ in range(len(UI_TYPES))]
        labelOneHot[label2Idx[label]] = 1
        return bounds, labelOneHot
    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(keys)}
    common_idx_list = []
    observed_idx_arr = np.zeros((len(keys),2), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality



    # Load modalities
    if ('S' in args.modality) and ('W' in args.modality):
        s_list = []
        w_list = []
        label_list = []
        observed_idx_arr_0 = []
        observed_idx_arr_1 = []
        for idx in range(len(keys)):
            example = example_list[keys[idx]]
            screenId = example['screen_id']
            # image modality
            screenImg = Image.open(os.path.join(img_dir, screenId + ".jpg")).convert("RGB")
            screenImg = img_transforms(screenImg)
            screenWireframeImg = Image.open(os.path.join(wireframe_dir, screenId + ".png")).convert("RGB")
            screenWireframeImg = img_transforms(screenWireframeImg)
            screenLabel = topic2Idx[example['topic']]

            if np.random.random_sample() < args.noise_level:
                screenImg = torch.rand(screenImg.size(0), screenImg.size(1), screenImg.size(2))
                observed_idx_arr_0.append(False) 
            else:
                update_modality_combinations(idx, 'S')
                observed_idx_arr_0.append(True) 
            if np.random.random_sample() < args.noise_level:
                screenWireframeImg = torch.rand(screenWireframeImg.size(0), screenWireframeImg.size(1), screenWireframeImg.size(2))
                observed_idx_arr_1.append(False) 
            else:
                update_modality_combinations(idx, 'W')
                observed_idx_arr_1.append(True) 

            s_list.append(screenImg)
            w_list.append(screenWireframeImg)
            label_list.append(screenLabel)

        observed_idx_arr = np.zeros((len(label_list),2), dtype=bool) # IGCB order
        observed_idx_arr[:, 0] = observed_idx_arr_0
        observed_idx_arr[:, 1] = observed_idx_arr_1

        data_dict['screenshot'] = np.array(s_list)
        data_dict['wireframe'] = np.array(w_list)
        filtered_idx = set(list(id_to_idx.values()))
        common_idx_list.append(filtered_idx)

            
        if args.patch:
            encoder_dict['screenshot'] = torch.nn.Sequential(
                VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(),
                PatchEmbeddings(16, num_patches=args.num_patches, embed_dim=args.hidden_dim).to(args.device)
                )
            encoder_dict['wireframe'] = torch.nn.Sequential(
                VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(),
                PatchEmbeddings(16, num_patches=args.num_patches, embed_dim=args.hidden_dim).to(args.device)
                )
        else:
            encoder_dict['screenshot'] = VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()
            encoder_dict['wireframe'] = VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()
        input_dims['screenshot'] = 16
        input_dims['wireframe'] = 16

    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]

    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_keys if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in val_keys if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_keys if id in id_to_idx]

    def all_modalities_missing(idx):
        return data_dict['modality_comb'][idx] == -1

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]
    valid_idxs = [idx for idx in valid_idxs if not all_modalities_missing(idx)]
    test_idxs = [idx for idx in test_idxs if not all_modalities_missing(idx)]

    mc_num_to_mc = {v:k for k,v in combination_to_index.items()}
    mc_idx_dict = {mc_num_to_mc[mc_num]: list(np.where(np.array(data_dict['modality_comb']) == mc_num)[0]) for mc_num in set(data_dict['modality_comb']) if mc_num != -1}

    labels = np.array(label_list)
    n_labels = len(set(labels))
    # breakpoint()
    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, mc_idx_dict, mc_num_to_mc



def collate_fn(batch):
    data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds

def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, transforms, masks, use_common_ids=True, dataset='mosi'):
    if 'image' in list(data_dict.keys()):
        train_transfrom = val_transform = test_transform = transforms['image']
        # val_transform = test_transform = False
        mask = masks['image']
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, train_transfrom, mask, use_common_ids)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, val_transform, mask, use_common_ids)
    test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, test_transform, mask, use_common_ids)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader

# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index