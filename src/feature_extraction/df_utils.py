import os
from src.feature_extraction.data_utils import *
from src.feature_extraction.utils import extract_features

def get_base_path(data_name):
    if data_name == 'd7p':
        return os.path.join('data', 'd7p')
    elif data_name == 'dmf':
        return os.path.join('data', 'dmf', 'DMF')
    elif data_name == 'ham':
        return os.path.join('data', 'ham')
    else:
        raise ValueError("Unknown data name. Use 'd7p', 'dmf', or 'ham'.")
        
def get_image_paths(data_name, base_path, part=1):
    if data_name == 'd7p':
        return get_d7p_paths(base_path)
    elif data_name == 'dmf':
        return get_dmf_paths(base_path)
    elif data_name == 'ham':
        return get_ham_paths(base_path, part=part)  # Adjust part as needed
    else:
        raise ValueError("Unknown data name. Use 'd7p', 'dmf', or 'ham'.")
    
def create_df_paths(model_name, data_name):
    prefix = ''
    suffix = data_name
    if model_name == 'PanDerm':
         prefix = 'panderm'
    elif model_name == 'derm-foundation':
         prefix = 'derm'
    elif model_name == 'vit':
        prefix = 'clip'
    else:
        raise ValueError("Unknown model name. Use 'PanDerm', 'derm-foundation', or 'vit'.")
    
    aux_path = os.path.join('outputs', f'{prefix}_{suffix}_pre.csv')
    output_path = os.path.join('outputs', f'{prefix}_{suffix}_.csv')

    return aux_path, output_path


def extract_and_save(model_name, data_name, start=0, end = None, batch_size = 100):
    
    base_path = get_base_path(data_name)
    pths = get_image_paths(data_name, base_path, part=1) # part attribute is only for HAM dataset as it has two parts
    aux_path, output_path = create_df_paths(model_name, data_name)
   
    extract_features(
        rslt_dict = pths,
        start=start, end = end,
        batch_size = batch_size, 
        model_name = model_name, ## Options: 'PanDerm', 'derm-foundation', 'vit'
        data_name = data_name, ## Options: 'ham', 'd7p', 'dmf'
        output_csv = aux_path)
    
    
    create_df(base_path= base_path,
            data_name= data_name, 
            pre_df_path= aux_path,
            output_path= output_path)

    
