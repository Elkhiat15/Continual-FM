
from sklearn.neural_network import MLPClassifier
from src.evaluation.utils import get_task_data
from src.evaluation.accuracy import get_MLP_accuracies

def train_MLP(tasks, X_train, y_train, X_test, y_test):
    task_models = {}
    acc_scores = {} 
    baac_scores = {}
    
    for task_name, task_classes in tasks.items():
        print(f"\nTraining {task_name}...")
    
        X_train_task, y_train_task = get_task_data(X_train, y_train, task_classes)
        X_test_task, y_test_task = get_task_data(X_test, y_test, task_classes)
    
    
        mlp = MLPClassifier(hidden_layer_sizes=(512, 256), random_state = 15,  activation='relu', solver='adam', max_iter=200)
        mlp.fit(X_train_task, y_train_task)
    
        task_models[task_name] = mlp
    
        y_pred = mlp.predict(X_test_task)

        accuracy, baccuracy = get_MLP_accuracies(y_test_task, y_pred)
        acc_scores[task_name] = accuracy
        baac_scores[task_name] = baccuracy
        
        print(f"Current classes in MLP: {mlp.classes_}")
        print(f"{task_name} Accuracy: {accuracy * 100:.2f}%")
        print(f"{task_name} Balanced Accuracy: {baccuracy * 100:.2f}%")

    baac_scores
    avg_acc = (sum(acc_scores.values())/len(acc_scores))
    avg_baac = (sum(baac_scores.values())/len(acc_scores))
    print(f"\nTotal Average Accuracy: {avg_acc * 100:.2f}%")
    print(f"Average Balanced Accuracy: {avg_baac * 100:.2f}%")