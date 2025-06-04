import pandas as pd
from sklearn.model_selection import train_test_split


def create_dmf_df(df_original):

    y = df_original['cell_type_idx']
    df_train, df_test = train_test_split(df_original, test_size=0.2, random_state=42, stratify=y)
    
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()


    return df_train, df_test

def create_d7p_df(df_original):
    df_original['cell_type_idx'] = pd.Categorical(df_original['class']).codes

    y = df_original['cell_type_idx']
    df_train, df_test = train_test_split(df_original, test_size=0.2, random_state=42, stratify=y)
    
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()


    return df_train, df_test

def create_ham_df(df_original):
    
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df_original['cell_type'] = df_original['class'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    # print(df_original['duplicates'].value_counts())


    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # now we create a test set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_test = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    # print(df_test.shape)

    # This set will be df_original excluding all rows that are in the test set
    # This function identifies if zuan image is part of the train or test set.
    def get_test_rows(x):
        # create a list of all the lesion_id's in the test set
        val_list = list(df_test['image_id'])
        if str(x) in val_list:
            return 'test'
        else:
            return 'train'

    # identify train and test rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_test'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_test'] = df_original['train_or_test'].apply(get_test_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_test'] == 'train']

    # print(df_train['cell_type_idx'].value_counts())
    # print(df_train['cell_type'].value_counts())

    train_class_counts = df_train['cell_type_idx'].value_counts()
    # print("Train Set:")
    # print(train_class_counts)
    test_class_counts = df_test['cell_type_idx'].value_counts()
    # print("Test Set:")
    # print(test_class_counts)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()


    return df_train, df_test


def setup(file_path, data_name = 'ham'):
    embeddings = pd.read_csv(file_path)
    embeddings.head()

    if data_name == 'ham':
        df_train, df_test = create_ham_df(embeddings)
    elif data_name == 'd7p':
        df_train, df_test = create_d7p_df(embeddings)
    elif data_name == 'dmf':
        df_train, df_test = create_dmf_df(embeddings)
        
    numerical_columns = [col for col in df_train.columns if col.isdigit()]
    
    X_train = df_train[numerical_columns].values 
    y_train = df_train["cell_type_idx"].values

    X_test = df_test[numerical_columns].values
    y_test = df_test["cell_type_idx"].values

    return X_train, y_train, X_test, y_test