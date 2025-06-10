from src.evaluation.constants import *
from src.evaluation.utils import *
from src.evaluation.accuracy import get_NMC_accuracies
from src.evaluation.MLP import train_MLP

def run_experiment(X_train, y_train, X_test, y_test, model_name = 'panderm', data_name='ham'):
    print(f"Running experiment for {model_name.upper()}  over {data_name.upper()} Dataset ....")

    in_dim, out_dim = IN_DIM_PANDERM, OUT_DIM_PANDERM
    if model_name == 'derm':
        in_dim, out_dim = IN_DIM_DERM, OUT_DIM_DERM
    elif model_name == 'clip':
        in_dim, out_dim = IN_DIM_CLIP, OUT_DIM_CLIP

    tasks = ham_tasks
    if data_name == 'd7p':
        tasks = d7p_tasks
    elif data_name == 'dmf':
        tasks = dmf_tasks

    
    print("\n\n$$$$$$$$$$$$$$$ Results with: MLP $$$$$$$$$$$$$$$")
    if data_name == 'ham':
        X_train_norm, X_test_norm = normalize_data(X_train, X_test)
        train_MLP(tasks, X_train_norm, y_train, X_test_norm, y_test)
    else:
        train_MLP(tasks, X_train, y_train, X_test, y_test)

    print("\n\n$$$$$$$$$$$$$$$ Results with: NMC Variations $$$$$$$$$$$$$$$")
    print("\n\n======= Results with: Base NMC ======")
    class_prototypes = get_prototypes(X_train, y_train)
    get_NMC_accuracies(X_test, y_test, class_prototypes)

    print("\n\n======= Results with: Base NMC & Normalization ======")
    X_train_norm, X_test_norm = normalize_data(X_train, X_test)
    class_prototypes = get_prototypes(X_train_norm, y_train)
    get_NMC_accuracies(X_test_norm, y_test, class_prototypes)
    
    print("\n\n======= Results with: Random Projection ======")
    X_train_proj, X_test_proj = get_projected_datasets(X_train, X_test, in_dim, out_dim)
    class_prototypes_proj = get_prototypes(X_train_proj, y_train)
    get_NMC_accuracies(X_test_proj, y_test, class_prototypes_proj)

    print("\n\n======= Results with: Random Projection & Normalization ======")
    X_train_proj_norm, X_test_proj_norm = normalize_data(X_train_proj, X_test_proj)
    class_prototypes_proj_norm = get_prototypes(X_train_proj_norm, y_train)
    get_NMC_accuracies(X_test_proj_norm, y_test, class_prototypes_proj_norm)
    print("\n\n======= Results with: LDA ======")    
    X_train_lda, X_test_lda = apply_LDA(X_train, y_train, X_test)
    class_prototypes_lda = get_prototypes(X_train_lda, y_train)
    get_NMC_accuracies(X_test_lda, y_test, class_prototypes_lda)

    print("\n\n======= Results with: PCA ======")    
    X_train_pca, X_test_pca = apply_PCA(X_train, X_test)
    class_prototypes_pca = get_prototypes(X_train_pca, y_train)
    get_NMC_accuracies(X_test_pca, y_test, class_prototypes_pca)

    print("\n\n======= Results with: PCA & Normalization ======")  
    X_train_norm_pca, X_test_norm_pca = apply_PCA(X_train_norm, X_test_norm)
    class_prototypes_norm_pca = get_prototypes(X_train_norm_pca, y_train)
    get_NMC_accuracies(X_test_norm_pca, y_test, class_prototypes_norm_pca)

    print("\n\n======= Results with: Hyperbolic Projections ======")
    run_hyper_experiment(X_train, y_train, X_test, y_test)

    print("\n\n======= Results with: Hyperbolic Projections & Norm ======")
    run_hyper_experiment(X_train_norm, y_train, X_test_norm, y_test)
