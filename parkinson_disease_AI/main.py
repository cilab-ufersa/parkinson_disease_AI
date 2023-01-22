import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from mlxtend.plotting import plot_learning_curves
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from utils import addNoise, to_csv

if __name__ == "__main__":

    # Retrieving the dataset
    dataset = pd.read_csv('parkinson_disease_AI\dataset\expanded_data.csv')

    # Saving a new expanded DataFrame with 8000 data at "parkinson_disease_AI\dataset\expanded_data.csv"
    # dataset = addNoise(dataset, 4000)
    # to_csv(dataset)

    listDiagnosis = dataset["Diagnosis"].to_numpy()
    listPerson = dataset[["velocityWeighted", "pressureWeighted", "CISP"]]

    # Balance dataset
    #sm = SMOTE(random_state=13)
    #sm.fit(listPerson, listDiagnosis)
    #listPerson, listDiagnosis = sm.fit_resample(listPerson, listDiagnosis)

    # Splits
    #n_splits_knn = 10
    #n_splits_dt = 10
    #n_splits_rf = 10
    n_splits_svm = 10

    # kf = KFold(n_splits=n_splits)
    #kf_knn = KFold(n_splits=n_splits_knn, shuffle=True, random_state=13)
    #kf_dt = KFold(n_splits=n_splits_dt, shuffle=True, random_state=13)
    #kf_rf = KFold(n_splits=n_splits_rf, shuffle=True, random_state=13)
    kf_svm = KFold(n_splits=n_splits_svm, shuffle=True, random_state=13)

    # KNN
    knnc = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_estimators': np.arange(6, 16, 2), 
                'criterion': ['gini', 'entropy', 'log_loss'],	    'criterion': ['gini', 'entropy', 'log_loss'], 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False], 
                'max_features':  ['sqrt', 'log2'],	    'max_depth': np.arange(10, 30, 5)}, cv=3)
    # with open('parkinson_disease_AI\knnc.pkl', 'rb') as file:
    #     knnc = pickle.load(file)
    
    # Decision Tree
    cartc = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={'n_estimators': np.arange(6, 16, 2), 
                'criterion': ['gini', 'entropy', 'log_loss'],	    'criterion': ['gini', 'entropy', 'log_loss'], 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False], 
                'max_features':  ['sqrt', 'log2'],	    'max_depth': np.arange(10, 30, 5)}, cv=3)
    # with open('cartc.pkl', 'rb') as file:
    #     cartc = pickle.load(file)

    # # Random Forest
    rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid={'n_estimators': np.arange(6, 16, 2), 
                'criterion': ['gini', 'entropy', 'log_loss'],	    'criterion': ['gini', 'entropy', 'log_loss'], 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False], 
                'max_features':  ['sqrt', 'log2'],	    'max_depth': np.arange(10, 30, 5)}, cv=3)
    # with open('rfc_model.pkl', 'rb') as file:
    #     rfc = pickle.load(file)

    # # SVM Poly
    # with open('clf_poly.pkl', 'rb') as file:
    #     clf = pickle.load(file)

    #SVM RBF
    with open('parkinson_disease_AI\clf_rbf.pkl', 'rb') as file:
       clf = pickle.load(file)


    ss = StandardScaler()
    TP_TN_FP_FN = np.zeros((4, 4))

    '''
    # KNN
    for train_index, test_index in kf_knn.split(listPerson, y=listDiagnosis):

        x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
        y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

        x_train = ss.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
        x_test = ss.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

        knnc_trained = knnc.fit(x_train, y_train)

        predict = knnc_trained.predict(x_test)

        for i in range(y_test.shape[0]):
            if y_test[i] == 0 and predict[i] == 0:
                TP_TN_FP_FN[0][1] += 1
            elif y_test[i] == 1 and predict[i] == 1:
                TP_TN_FP_FN[0][0] += 1
            elif y_test[i] == 1 and predict[i] == 0:
                TP_TN_FP_FN[0][3] += 1
            elif y_test[i] == 0 and predict[i] == 1:
                TP_TN_FP_FN[0][2] += 1

    # Decision Tree
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
    
    # Random Forest
    for train_index, test_index in kf_rf.split(listPerson, y=listDiagnosis):

        x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
        y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

        x_train = ss.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
        x_test = ss.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

        rfc_trained = rfc.fit(x_train, y_train)

        predict = rfc_trained.predict(x_test)

        for i in range(y_test.shape[0]):
            if y_test[i] == 0 and predict[i] == 0:
                TP_TN_FP_FN[2][1] += 1
            elif y_test[i] == 1 and predict[i] == 1:
                TP_TN_FP_FN[2][0] += 1
            elif y_test[i] == 1 and predict[i] == 0:
                TP_TN_FP_FN[2][3] += 1
            elif y_test[i] == 0 and predict[i] == 1:
                TP_TN_FP_FN[2][2] += 1
    '''
    predicted = np.empty(listPerson.shape[0])
    # SVM
    for train_index, test_index in kf_svm.split(listPerson, y=listDiagnosis):

        x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
        y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

        x_train = ss.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
        x_test = ss.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

        clf_trained = clf.fit(x_train, y_train)

        # print(clf.best_params_)

        predict = clf_trained.predict(x_test)

        for i, index in enumerate(test_index):
            predicted[index] = clf_trained.decision_function(x_test)[i]

        for i in range(y_test.shape[0]):
            if y_test[i] == 0 and predict[i] == 0:
                TP_TN_FP_FN[3][1] += 1
            elif y_test[i] == 1 and predict[i] == 1:
                TP_TN_FP_FN[3][0] += 1
            elif y_test[i] == 1 and predict[i] == 0:
                TP_TN_FP_FN[3][3] += 1
            elif y_test[i] == 0 and predict[i] == 1:
                TP_TN_FP_FN[3][2] += 1

    # TODO: Save the models

    # Evaluation
    acc = np.empty(4)
    sens = np.empty(4)
    esp = np.empty(4)
    for i in range(4):
        TP, TN, FP, FN = TP_TN_FP_FN[i]
        acc[i] = ((TP + TN) / (TP + TN + FN + FP) * 100)
        sens[i] = (TP / (TP + FN) * 100)
        esp[i] = (TN / (TN + FP) * 100)

    mod = ['KNN', 'DecisionTree', 'RandomForest', 'SVM']
    nam = ['acurácia', 'sensibilidade', 'especificidade']
    for i in range(4):
        print(f'{mod[i]} sua {nam[0]} é de {np.round(acc[i], 2)}%')
    for i in range(4):
        print(f'{mod[i]} sua {nam[1]} é de {np.round(sens[i], 2)}%')
    for i in range(4):
        print(f'{mod[i]} sua {nam[2]} é de {np.round(esp[i], 2)}%')

    # Plots
    '''
    X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_knn,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, knnc)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_dt,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, cartc)
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_rf,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, rfc)
    plt.show()
    '''

    X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_svm,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, clf)
    plt.show()

    #ROC

    fpr, tpr, thresholds = metrics.roc_curve(listDiagnosis, predicted)
    auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name = 'SVM')
    display.plot()
    plt.show()
