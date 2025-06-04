import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from src.evaluation.utils import predict_nearest_mean


def get_NMC_accuracies(X_test, y_test, class_prototypes):
    predictions = [predict_nearest_mean(sample,class_prototypes) for sample in X_test]

    accuracy = np.mean(np.array(predictions) == np.array(y_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    baccuracy = balanced_accuracy_score(np.array(y_test), np.array(predictions))
    print(f"Balanced Accuracy: {baccuracy * 100:.2f}%")


def get_MLP_accuracies(y_test_task, y_pred):
    accuracy = accuracy_score(y_test_task, y_pred)        
    baccuracy = balanced_accuracy_score(y_test_task, y_pred)
    return accuracy, baccuracy