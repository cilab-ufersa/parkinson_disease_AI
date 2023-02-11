'''n_splits_rf = 3
kf_rf = KFold(n_splits=n_splits_rf, shuffle=True, random_state=13)
# Random Forest
    rfc = RandomForestClassifier(n_estimators=5, criterion='gini', max_features='sqrt')
    # with open('rfc_model.pkl', 'rb') as file:
    #     rfc = pickle.load(file)
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
X_train, X_test, y_train, y_test = train_test_split(listPerson, listDiagnosis, test_size=1 / n_splits_rf,
                                                        random_state=13)
    X_train = ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, rfc)
    plt.show()'''