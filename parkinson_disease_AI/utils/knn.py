import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_learning_curves
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Read data
dataset = pd.read_csv('../dataset/testingdataset.csv')
listDiagnosis = dataset["Diagnosis"].to_numpy()
listPerson = dataset[["velocityWeighted", "pressureWeighted", "CISP"]]

# Split
n_splits_knn = 10
kf_knn = KFold(n_splits=n_splits_knn, shuffle=True, random_state=13)

# KNN grid search
# knnc = GridSearchCV(estimator=KNeighborsClassifier(),
#                    param_grid={'n_neighbors': np.arange(1, 101),
#                                'weights': ['uniform'],
#                                'algorithm': ['auto'],
#                                'p': np.arange(1, 11)},
#                    cv=n_splits_knn)
knnc = KNeighborsClassifier(n_neighbors=60, p=2)

# with open('parkinson_disease_AI\knnc.pkl', 'rb') as file:
#    knnc = pickle.load(file)

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

    knnc_trained = knnc.fit(x_train, y_train)

    predict = knnc_trained.predict(x_test)
    # print(knnc.best_params_)

    for i in range(y_test.shape[0]):
        if y_test[i] == 0 and predict[i] == 0:
            TP_TN_FP_FN[1] += 1
        elif y_test[i] == 1 and predict[i] == 1:
            TP_TN_FP_FN[0] += 1
        elif y_test[i] == 1 and predict[i] == 0:
            TP_TN_FP_FN[3] += 1
        elif y_test[i] == 0 and predict[i] == 1:
            TP_TN_FP_FN[2] += 1

    predicted[test_index] = knnc_trained.predict_proba(x_test)

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

plot_learning_curves(X_train, y_train, X_test, y_test, knnc)
plt.show()

# ROC

fpr, tpr, thresholds = metrics.roc_curve(listDiagnosis, predicted[:, 1])
auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name='KNN')
display.plot()
plt.show()
