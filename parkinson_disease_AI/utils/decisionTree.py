# Splits
# n_splits_dt = 10

# kf_dt = KFold(n_splits=n_splits_dt, shuffle=True, random_state=13)

# Decision Tree cartc = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={
# 'n_estimators': np.arange(6, 16, 2), 'criterion': ['gini', 'entropy', 'log_loss'],
# 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False], 'max_depth': np.arange(10, 30,
# 5)}, cv=3) with open('cartc.pkl', 'rb') as file: cartc = pickle.load(file)

# Decision Tree
"""
    for train_index, test_index in kf_dt.split(listPerson, y=listDiagnosis):

        x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
        y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

        x_train = ss.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
        x_test = ss.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

        cartc_trained = cartc.fit(x_train, y_train)

        predict = cartc_trained.predict(x_test)

        for i in range(y_test.shape[0]):
            if y_test[i] == 0 and predict[i] == 0:
                TP_TN_FP_FN[1][1] += 1
            elif y_test[i] == 1 and predict[i] == 1:
                TP_TN_FP_FN[1][0] += 1
            elif y_test[i] == 1 and predict[i] == 0:
                TP_TN_FP_FN[1][3] += 1
            elif y_test[i] == 0 and predict[i] == 1:
                TP_TN_FP_FN[1][2] += 1


    X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_dt,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, cartc)
    plt.show()"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_learning_curves
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Read data
dataset = pd.read_csv('../dataset/expanded_data.csv')
listDiagnosis = dataset["Diagnosis"].to_numpy()
listPerson = dataset[["velocityWeighted", "pressureWeighted", "CISP"]]

# Split
n_splits_knn = 10
kf_knn = KFold(n_splits=n_splits_knn, shuffle=True, random_state=13)

# Decision tree grid search
# cartc = GridSearchCV(estimator=DecisionTreeClassifier(),
#                     param_grid={'criterion': ['gini', 'entropy', 'log_loss'],
#                                 'splitter': ['best', 'random'],
#                                 'max_depth': [None],
#                                 'min_samples_split': np.arange(1, 6),
#                                 'min_samples_leaf': np.arange(2, 6),
#                                 'max_features': ['sqrt', 'log2']},
#                     cv=n_splits_knn)
cartc = DecisionTreeClassifier(criterion='gini', max_depth=None, max_features='sqrt',
                               min_samples_leaf=4, min_samples_split=1, splitter='best')

# with open('cartc.pkl', 'rb') as file:
#     cartc = pickle.load(file)

ss = StandardScaler()
TP_TN_FP_FN = np.zeros(4)
predicted = np.empty((listPerson.shape[0], 2))

# Calc
for train_index, test_index in kf_knn.split(listPerson, y=listDiagnosis):

    x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
    y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

    x_train = ss.fit_transform(x_train)
    x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    x_test = ss.transform(x_test)
    x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    cartc_trained = cartc.fit(x_train, y_train)

    predict = cartc_trained.predict(x_test)
    #print(cartc.best_params_)

    for i in range(y_test.shape[0]):
        if y_test[i] == 0 and predict[i] == 0:
            TP_TN_FP_FN[1] += 1
        elif y_test[i] == 1 and predict[i] == 1:
            TP_TN_FP_FN[0] += 1
        elif y_test[i] == 1 and predict[i] == 0:
            TP_TN_FP_FN[3] += 1
        elif y_test[i] == 0 and predict[i] == 1:
            TP_TN_FP_FN[2] += 1

    # predicted[test_index] = cartc_trained.predict_proba(x_test)

# Evaluation
TP, TN, FP, FN = TP_TN_FP_FN

acc = ((TP + TN) / (TP + TN + FN + FP) * 100)
sens = (TP / (TP + FN) * 100)
esp = (TN / (TN + FP) * 100)

print(f'KNN sua acurácia é de {np.round(acc, 2)}%')
print(f'KNN sua sensibilidade é de {np.round(sens, 2)}%')
print(f'KNN sua especificidade é de {np.round(esp, 2)}%')

# Learning curve
X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis,
                                                    test_size=1 / n_splits_knn,
                                                    random_state=13)
X_train = ss.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
X_test = ss.transform(X_test)
X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

plot_learning_curves(X_train, y_train, X_test, y_test, cartc)
plt.show()

# ROC

fpr, tpr, thresholds = metrics.roc_curve(listDiagnosis, predicted[:, 1])
auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='Decision Tree')
display.plot()
plt.show()
