
#n_splits_svm = 3
#kf_svm = KFold(n_splits=n_splits_svm, shuffle=True, random_state=13)
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
# SVM
'''for train_index, test_index in kf_svm.split(listPerson, y=listDiagnosis):

        x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
        y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]

        x_train = ss.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
        x_test = ss.transform(x_test)
        x_test = pd.DataFrame(x_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

        clf_trained = clf.fit(x_train, y_train)

        #print(clf.best_params_)

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
                TP_TN_FP_FN[3][2] += 1'''

''' X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_svm,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, clf)
    plt.show()'''