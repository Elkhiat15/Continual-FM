import os
import pandas as pd


def get_all_image_paths(root_dir, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}):
    image_paths = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                full_path = os.path.join(subdir, file)
                image_paths.append(full_path)
    return image_paths


def get_ham_paths(base_ham_path, part=1):
    HAM_part1 = os.path.join(base_ham_path,'HAM10000_images_part_1')
    part1_len = len(os.listdir(HAM_part1))
    print(f'Lengh of part1: {part1_len}')

    HAM_part2 = os.path.join(base_ham_path,'HAM10000_images_part_2')
    part2_len = len(os.listdir(HAM_part2))
    print(f'Lengh of part2: {part2_len}')

    base_dir = HAM_part1
    if part == 2:
        base_dir = HAM_part2

    ham_pths = [
            os.path.join(base_dir, fname)
            for fname in os.listdir(base_dir)
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