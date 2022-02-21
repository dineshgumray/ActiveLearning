#from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from sklearn.metrics import accuracy_score

def evaluate(active_learner, train, test):

    y_pred = active_learner.classifier.predict(train)
    train_acc = accuracy_score(y_pred, train.y)
    # y_pred_prob = active_learner.classifier.predict_proba(train)
    # fpr, tpr, th = metrics.roc_curve(train.y, y_pred_prob[:,1])
    # train_auc = metrics.auc(fpr, tpr)
    # tarin_log_loss = metrics.log_loss(train.y, y_pred_prob)

    y_pred_test = active_learner.classifier.predict(test)
    test_acc = accuracy_score(y_pred_test, test.y)
    # y_pred_test_prob = active_learner.classifier.predict_proba(test)  
    # fpr, tpr, th = metrics.roc_curve(test.y, y_pred_test_prob[:,1])
    # test_auc = metrics.auc(fpr, tpr)
    # test_log_loss = metrics.log_loss(test.y, y_pred_test_prob)

    print('Train accuracy: {:.2f}'.format(train_acc))
    print('Test accuracy: {:.2f}'.format(test_acc))

    return (train_acc, test_acc)


res = tuple()
results = defaultdict(list)
res = evaluate(active_learner, train[labeled_indices], test)
results["train"].append(res[0])
results["test"].append(res[1])

    
