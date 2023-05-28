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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from pycaret.classification import * #installed -U --pre pycaret

from utils import addNoise, to_csv

if __name__ == "__main__":

    # Retrieving the dataset
    dataset = pd.read_csv('parkinson_disease_AI/dataset/listPersonRefactor.csv')

    listDiagnosis = dataset["Diagnosis"].to_numpy()
    listPerson = dataset[["velocityWeighted", "pressureWeighted", "CISP"]]

    # Pycaret
    # exp_clf = setup(data=dataset, target='Diagnosis', train_size=0.8, data_split_shuffle=True)
    # compare_models()
    # gbc = create_model('gbc')
    # tuned_gbc = tune_model(gbc)
    # evaluate_model(tuned_gbc)
    # predictions = predict_model(tuned_gbc, data=dataset)

    # Splits
    # n_splits_knn = 10
    # n_splits_dt = 10
    # n_splits_rf = 5
    # n_splits_svm = 3
    n_splits_gbc = 3

    # KFolding
    # kf_knn = KFold(n_splits=n_splits_knn, shuffle=True, random_state=13)
    # kf_dt = KFold(n_splits=n_splits_dt, shuffle=True, random_state=13)
    # kf_rf = KFold(n_splits=n_splits_rf, shuffle=True, random_state=13)
    # kf_svm = KFold(n_splits=n_splits_svm, shuffle=True, random_state=13)
    kf_gbc = KFold(n_splits=n_splits_gbc, shuffle=True, random_state=13)
    
    # KNN
    #knnc = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_estimators': np.arange(6, 16, 2),
    #                                                                  'criterion': ['gini', 'entropy', 'log_loss'], 'max_features': ['sqrt', 'log2'], 'bootstrap': [True, False],
    #                                                                  'max_depth': np.arange(10, 30, 5)}, cv=3)
    # knnc = KNeighborsClassifier()
    # with open('parkinson_disease_AI\knnc.pkl', 'rb') as file:
    #     knnc = pickle.load(file)
    
    # Decision Tree
    # cartc = DecisionTreeClassifier(criterion='gini', max_depth=10, max_features=10, min_samples_leaf=10, min_samples_split=50, splitter='random')
    # cartc = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid={'criterion': ['gini', 'entropy', 'log_loss'],
    #                                                                      'splitter': ['best', 'random'], 
    #                                                                      'max_depth': [1, 10, 25, 50],
    #                                                                      'min_samples_split': [1, 10, 25, 50],
    #                                                                      'min_samples_leaf': [1, 10, 25, 50],
    #                                                                #      'min_weight_fraction_leaf': [0, 1, 5, 10, 50, 100],
    #                                                                      'max_features': ['sqrt', 'log2', 1, 10, 50]
    #                                                                      })
    # with open('cartc.pkl', 'rb') as file:
    #     cartc = pickle.load(file)

    # Random Forest
    # rfc = RandomForestClassifier(criterion='log_loss', max_features='log2', n_estimators=50, class_weight='balanced_subsample', max_depth=10, min_samples_split=20, min_samples_leaf=50)
    # rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid={
    #                                                                     'criterion': ['log_loss'],
    #                                                                     'n_estimators': [50],
    #                                                                     'max_features': ['log2'],
    #                                                                     'class_weight': ['balanced_subsample'],       
    #                                                                     'max_depth': [10],
    #                                                                     'min_samples_split': [1, 2, 10, 50],
    #                                                                     #  'min_samples_leaf': [1, 2, 5, 10, 50],
    #                                                                     #  'bootstrap': [True, False],
    #                                                                     #  'oob_score': [True, False],
    #                                                                     #  'verbose': [0, 1, 3, 5]
    #                                                                 })

    # with open('rfc_model.pkl', 'rb') as file:
    #     rfc = pickle.load(file)

    # # SVM Poly
    #with open('clf_poly.pkl', 'rb') as file:
    #    clf = pickle.load(file)
    #clf = GridSearchCV(estimator=SVC(),
    #                   param_grid={'C': [1500], 'kernel': ['rbf'], 'gamma': [1]},
    #                   cv=n_splits_svm)

    # clf = SVC(C = 2000, kernel= 'poly', gamma= 0.9, coef0=1/2, degree =3)

    #SVM RBF
    #with open('clf_rbf.pkl', 'rb') as file:
    #   clf = pickle.load(file)

    # gbc = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features='sqrt', min_samples_leaf=1, min_samples_split=10, n_estimators=130)
    # gbc = GradientBoostingClassifier(learning_rate=0.07, max_depth=3, max_features='sqrt', min_samples_leaf=4, min_samples_split=2, n_estimators=45)
    gbc = GradientBoostingClassifier(learning_rate=0.04, max_depth=7, max_features='sqrt', min_samples_leaf=10, min_samples_split=6, n_estimators=47)

    # gbc = GridSearchCV(estimator=GradientBoostingClassifier(), param_grid = {
    #                                                                         'n_estimators': [30, 40, 50, 30, 60, 70, 80],
    #                                                                         'learning_rate': [0.1, 0.05, 0.01, 0.5, 0.03, 0.3, 1],
    #                                                                         'max_depth': [1, 2, 3, 4, 5, 6, 7, 10, 12],
    #                                                                         'min_samples_split': [1, 2, 3, 4, 5, 6, 8, 10, 15],
    #                                                                         'min_samples_leaf': [1, 2, 3, 4, 5, 6, 10],
    #                                                                         'max_features': ['sqrt']
    #                                                                          })

    ss = StandardScaler()
    TP_TN_FP_FN = np.zeros((5, 4))


 
    # KNN
    # for train_index, test_index in kf_knn.split(listPerson, y=listDiagnosis):

    #     x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
    #     y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

    #     x_train = ss.fit_transform(x_train)
    #     x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    #     x_test = ss.transform(x_test)
    #     x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    #     knnc_trained = knnc.fit(x_train, y_train)

    #     predict = knnc_trained.predict(x_test)

    #     for i in range(y_test.shape[0]):
    #         if y_test[i] == 0 and predict[i] == 0:
    #             TP_TN_FP_FN[0][1] += 1
    #         elif y_test[i] == 1 and predict[i] == 1:
    #             TP_TN_FP_FN[0][0] += 1
    #         elif y_test[i] == 1 and predict[i] == 0:
    #             TP_TN_FP_FN[0][3] += 1
    #         elif y_test[i] == 0 and predict[i] == 1:
    #             TP_TN_FP_FN[0][2] += 1

    # # Decision Tree
    # for train_index, test_index in kf_dt.split(listPerson, y=listDiagnosis):

    #     x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
    #     y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

    #     x_train = ss.fit_transform(x_train)
    #     x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    #     x_test = ss.transform(x_test)
    #     x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    #     cartc_trained = cartc.fit(x_train, y_train)

    #     predict = cartc_trained.predict(x_test)

    #     for i in range(y_test.shape[0]):
    #         if y_test[i] == 0 and predict[i] == 0:
    #             TP_TN_FP_FN[1][1] += 1
    #         elif y_test[i] == 1 and predict[i] == 1:
    #             TP_TN_FP_FN[1][0] += 1
    #         elif y_test[i] == 1 and predict[i] == 0:
    #             TP_TN_FP_FN[1][3] += 1
    #         elif y_test[i] == 0 and predict[i] == 1:
    #             TP_TN_FP_FN[1][2] += 1

    #         # print(cartc.best_params_)
    
    # # Random Forest
    # for train_index, test_index in kf_rf.split(listPerson, y=listDiagnosis):

    #     x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
    #     y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

    #     x_train = ss.fit_transform(x_train)
    #     x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    #     x_test = ss.transform(x_test)
    #     x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    #     rfc_trained = rfc.fit(x_train, y_train)

    #     predict = rfc_trained.predict(x_test)

    #     for i in range(y_test.shape[0]):
    #         if y_test[i] == 0 and predict[i] == 0:
    #             TP_TN_FP_FN[2][1] += 1
    #         elif y_test[i] == 1 and predict[i] == 1:
    #             TP_TN_FP_FN[2][0] += 1
    #         elif y_test[i] == 1 and predict[i] == 0:
    #             TP_TN_FP_FN[2][3] += 1
    #         elif y_test[i] == 0 and predict[i] == 1:
    #             TP_TN_FP_FN[2][2] += 1

    
    # predicted = np.empty(listPerson.shape[0])
    # # SVM
    # for train_index, test_index in kf_svm.split(listPerson, y=listDiagnosis):

    #     x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
    #     y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

    #     x_train = ss.fit_transform(x_train)
    #     x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    #     x_test = ss.transform(x_test)
    #     x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    #     clf_trained = clf.fit(x_train, y_train)

    #     #print(clf.best_params_)

    #     predict = clf_trained.predict(x_test)

    #     for i, index in enumerate(test_index):
    #         predicted[index] = clf_trained.decision_function(x_test)[i]

    #     for i in range(y_test.shape[0]):
    #         if y_test[i] == 0 and predict[i] == 0:
    #             TP_TN_FP_FN[3][1] += 1
    #         elif y_test[i] == 1 and predict[i] == 1:
    #             TP_TN_FP_FN[3][0] += 1
    #         elif y_test[i] == 1 and predict[i] == 0:
    #             TP_TN_FP_FN[3][3] += 1
    #         elif y_test[i] == 0 and predict[i] == 1:
    #             TP_TN_FP_FN[3][2] += 1

    # GBC
    for train_index, test_index in kf_gbc.split(listPerson, y=listDiagnosis):

        x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
        y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

        x_train = ss.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
        x_test = ss.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

        gbc_trained = gbc.fit(x_train, y_train)
        predict = gbc_trained.predict(x_test)

        # for i, index in enumerate(test_index):
        #     predicted[index] = clf_trained.decision_function(x_test)[i]

        for i in range(y_test.shape[0]):
            if y_test[i] == 0 and predict[i] == 0:
                TP_TN_FP_FN[4][1] += 1
            elif y_test[i] == 1 and predict[i] == 1:
                TP_TN_FP_FN[4][0] += 1
            elif y_test[i] == 1 and predict[i] == 0:
                TP_TN_FP_FN[4][3] += 1
            elif y_test[i] == 0 and predict[i] == 1:
                TP_TN_FP_FN[4][2] += 1   
            
            # print(gbc.best_params_)

    # TODO: Save the models

    # Evaluation
    acc = np.empty(5)
    sens = np.empty(5)
    esp = np.empty(5)
    for i in range(5):
        TP, TN, FP, FN = TP_TN_FP_FN[i]
        acc[i] = ((TP + TN) / (TP + TN + FN + FP) * 100)
        sens[i] = (TP / (TP + FN) * 100)
        esp[i] = (TN / (TN + FP) * 100)

    mod = ['KNN', 'DecisionTree', 'RandomForest', 'SVM', 'GBC']
    nam = ['acurácia', 'sensibilidade', 'especificidade']
    for i in range(5):
        print(f'{mod[i]} sua {nam[0]} é de {np.round(acc[i], 2)}%')
    for i in range(5):
        print(f'{mod[i]} sua {nam[1]} é de {np.round(sens[i], 2)}%')
    for i in range(5):
        print(f'{mod[i]} sua {nam[2]} é de {np.round(esp[i], 2)}%')

    # # Plots
    # X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_knn,
    #                                                     random_state=13)
    # X_train = ss.fit_transform(X_train)
    # X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    # X_test = ss.transform(X_test)
    # X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    # plot_learning_curves(X_train, y_train, X_test, y_test, knnc)
    # plt.show()

    # X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_dt,
    #                                                     random_state=13)
    # X_train = ss.fit_transform(X_train)
    # X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    # X_test = ss.transform(X_test)
    # X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    # plot_learning_curves(X_train, y_train, X_test, y_test, cartc)
    # plt.show()

    # X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_rf,
    #                                                     random_state=13)
    # X_train = ss.fit_transform(X_train)
    # X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    # X_test = ss.transform(X_test)
    # X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    # plot_learning_curves(X_train, y_train, X_test, y_test, rfc)
    # plt.show()
    

    # X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_svm,
    #                                                     random_state=13)
    # X_train = ss.fit_transform(X_train)
    # X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    # X_test = ss.transform(X_test)
    # X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    # plot_learning_curves(X_train, y_train, X_test, y_test, clf)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_gbc,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, gbc)
    plt.show()

    # #ROC
    # fpr, tpr, thresholds = metrics.roc_curve(listDiagnosis, predicted)
    # auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name = 'DecisionTreeClassifier')
    # display.plot()
    # plt.show()

