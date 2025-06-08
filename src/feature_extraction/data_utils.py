import os
import pandas as pd
import numpy as np
from PIL import Image


def preprocess_image_np(img_path, size=(224, 224), mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # Open image and convert to RGB
    img = Image.open(img_path).convert("RGB")

    # Resize image
    img = img.resize(size, Image.BILINEAR)

    img_np = np.array(img).astype(np.float32) / 255.0  # scale to [0,1]

    # Normalize each channel: (x - mean) / std
    for c in range(3):
        img_np[..., c] = (img_np[..., c] - mean[c]) / std[c]
        
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


def get_all_image_paths(root_dir, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                full_path = os.path.join(subdir, file)
                image_paths.append(full_path)
    return image_paths


def get_ham_paths(base_ham_path):
    HAM_part1 = os.path.join(base_ham_path,'HAM10000_images_part_1')
    HAM_part2 = os.path.join(base_ham_path,'HAM10000_images_part_2')

    part1_files = os.listdir(HAM_part1)
    part2_files = os.listdir(HAM_part2)

    ham_pths = [
        os.path.join(HAM_part1, fname) for fname in part1_files
    ] + [
        os.path.join(HAM_part2, fname) for fname in part2_files
    ]

    return ham_pths

def get_d7p_paths(base_d7p_path): 
    d7p_pths_lst = get_all_image_paths(os.path.join(base_d7p_path,'release_v0/images'))
    meta_d7p= pd.read_csv(os.path.join(base_d7p_path,'release_v0/meta/meta.csv'))
    meta_d7p.rename(columns={'diagnosis': 'class'}, inplace=True)

    filtered_paths = [
    path for path in d7p_pths_lst
    if os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)) in meta_d7p.derm.values
    ]

    derm_to_case = dict(zip(meta_d7p['derm'], meta_d7p['case_num']))
    
    result_dict = {}

    for full_path in filtered_paths:
        matched = None
        for derm_path in derm_to_case.keys():
            if full_path.endswith(derm_path):
                matched = derm_path
                break
        
        if matched is not None:
            case_id = derm_to_case[matched]
            if case_id not in result_dict:
                result_dict[case_id] = []
            result_dict[case_id].append(full_path)


    sorted_dict = dict(sorted(result_dict.items()))
    sorted_list = list(sorted_dict.values())
    d7p_pths = [img_path[0] for img_path in sorted_list]

    return d7p_pths

def get_dmf_paths(base_dmf_path):
    meta_dmf=pd.read_csv(os.path.join(base_dmf_path,'meta-dmf.csv'))
    dmf_pths = [
        base_dmf_path+"/images/"+image_id+".png" for image_id in meta_dmf.image_id 
        if image_id+".png" in os.listdir(base_dmf_path+"/images")]
    
    return dmf_pths

def simplify_diagnosis(label):
    label = label.lower()
    if 'melanoma' in label:
        return 'melanoma'
    elif 'nevus' in label:
        return 'nevus'
    else:
        return label

def merge_df(pre_df_path, meta, meta_attrs):
    pre_df = pd.read_csv(pre_df_path)
    df = pre_df.astype({col: np.float32 for col in pre_df.select_dtypes(include=[np.float64]).columns})
    df = pd.merge(df, meta[meta_attrs], on='image_id', how='left')
    return df


def save_df(df, output_path, pre_df_path):
    os.remove(pre_df_path)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

def create_ham_df(base_path, pre_df_path, output_path):
    meta_ham= pd.read_csv(os.path.join(base_path,'HAM10000_metadata.csv'))
    meta_ham.rename(columns={'dx': 'class'}, inplace=True)
    
    df = merge_df(pre_df_path, meta_ham, ['image_id', 'class', 'lesion_id'])
    save_df(df, output_path, pre_df_path)

def create_d7p_df(base_path, pre_df_path, output_path):
    meta_d7p= pd.read_csv(os.path.join(base_path,'release_v0/meta/meta.csv'))
    meta_d7p = meta_d7p[~meta_d7p['diagnosis'].isin(['lentigo', 'melanosis', 'miscellaneous'])]
    meta_d7p.rename(columns={'case_num': 'image_id'}, inplace=True)
    meta_d7p['class'] = meta_d7p['diagnosis'].apply(simplify_diagnosis)

    df = merge_df(pre_df_path, meta_d7p, ['image_id', 'class'])
    save_df(df, output_path, pre_df_path)


def create_dmf_df(base_path, pre_df_path, output_path):
    meta_dmf= pd.read_csv(os.path.join(base_path,'meta-dmf.csv'))
    meta_dmf = meta_dmf[meta_dmf['image_id'] != 'B511b'] # B511b is a missed image in the dataset
    meta_dmf.reset_index(inplace=True)
    meta_dmf.rename(columns={'dx': 'class', 'dx_idx':'cell_type_idx'}, inplace=True)

    pre_df = pd.read_csv(pre_df_path)
    pre_df.rename(columns={'image_id': 'image_idx'}, inplace=True)
    df = pre_df.astype({col: np.float32 for col in pre_df.select_dtypes(include=[np.float64]).columns})
    df = pd.merge(df, meta_dmf[['image_id', 'class', 'cell_type_idx']], right_index=True, left_index=True)
    
    save_df(df, output_path, pre_df_path)


def create_df(base_path, data_name, pre_df_path, output_path):
    if data_name == 'ham':
        create_ham_df(base_path)
    elif data_name == 'd7p':
        create_d7p_df(base_path, pre_df_path, output_path)
    elif data_name == 'dmf':
        create_dmf_df(base_path, pre_df_path, output_path)
    else:
        raise ValueError(f"Unsupported dataset name: {data_name}")