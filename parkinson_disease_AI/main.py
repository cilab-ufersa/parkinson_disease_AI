import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_learning_curves
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm
from imblearn.over_sampling import SMOTE



if __name__ == "__main__":

    dataset = pd.read_csv('parkinson_disease_AI\dataset\listPerson.csv')
    listDiagnosis = dataset["Diagnosis"].to_numpy()
    listPerson = dataset[["velocityWeighted", "pressureWeighted", "CISP"]]

    # Balance dataset
    sm = SMOTE(random_state=13)
    sm.fit(listPerson, listDiagnosis)
    listPerson, listDiagnosis = sm.fit_resample(listPerson, listDiagnosis)
    n_splits = 5
    kf = KFold(n_splits=n_splits,shuffle=True)
    



    #SVM
    

    # KNN
    knnc = GridSearchCV(KNeighborsClassifier(), param_grid = [
        {
            'weights':['uniform'],
            'n_neighbors':[i for i in range(1,11)]
        },
        {
            'weights':['distance'],
            'n_neighbors':[i for i in range(1,11)],
            'p':[i for i in range(1,6)]
        }
    ], cv = n_splits)

    # Decision Tree
    cartc = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid = {'criterion': ['gini', 'entropy', 'log_loss'],
                        'splitter': ["best", "random"],
                        'max_depth': np.arange(10, 30, 5)
                        }, cv = n_splits)
    # Random Forest
    rfc = GridSearchCV(estimator = RandomForestClassifier(), param_grid =  {'n_estimators': np.arange(6, 16, 2),
                'criterion': ['gini', 'entropy', 'log_loss'],
                'max_features':  ['sqrt', 'log2'],
                'bootstrap': [True, False],
                'max_depth': np.arange(10, 30, 5)},
                 cv = n_splits)


    ss = StandardScaler()
    TP_TN_FP_FN = np.zeros((3,4))

    diagnosis = []
    guesseddiagnosisknn = []
    guesseddiagnosisdt = []
    guesseddiagnosisrf = []
    guesseddiagnosisvm = []

    for train_index, test_index in kf.split(listPerson,y=listDiagnosis):
        
        x_train, x_test = listPerson.iloc[train_index], listPerson.iloc[test_index]
        y_train, y_test = listDiagnosis[train_index], listDiagnosis[test_index]
        
        x_train= ss.fit_transform(x_train)
        x_train = pd.DataFrame(x_train, columns = ['velocityWeighted','pressureWeighted','CISP'])
        x_test = ss.transform(x_test)
        x_test = pd.DataFrame(x_test, columns = ['velocityWeighted','pressureWeighted','CISP'])      
        
        knnc_trained = knnc.fit(x_train, y_train)
        cartc_trained = cartc.fit(x_train, y_train)
        rfc_trained = rfc.fit(x_train, y_train)  
        #SVM
        clf = svm.SVC()
        clf_trained=clf.fit(x_train, y_train) 
   

        modelsc = [knnc_trained, cartc_trained, rfc_trained,clf_trained]

        j = 0
        diagnosis.append(y_test[:])
        for model in modelsc:
            predict = model.predict(x_test)
            if model == modelsc[0]:
                guesseddiagnosisknn.append(model.predict(x_test))
            elif model == modelsc[1]:
                guesseddiagnosisdt.append(model.predict(x_test))

            elif model == modelsc[2]:
                guesseddiagnosisvm.append(model.predict(x_test))  

            else:
                guesseddiagnosisrf.append(model.predict(x_test))


            for i in range(y_test.shape[0]):
                if y_test[i] == 0 and predict[i] == 0:
                    TP_TN_FP_FN[j][1] +=1
                elif y_test[i] == 1 and predict[i] == 1:
                    TP_TN_FP_FN[j][0] +=1
                elif y_test[i] == 1 and predict[i] == 0:
                    TP_TN_FP_FN[j][3] +=1
                elif y_test[i] == 0 and predict[i] == 1:
                    TP_TN_FP_FN[j][2] +=1
            j+=1
    
    #TODO: Save the models

'''
    # Evaluation
    acc = np.empty(3)
    sens = np.empty(3)
    esp = np.empty(3)
    for i in range(3):
        TP,TN,FP,FN = TP_TN_FP_FN[i]
        acc[i] = ((TP+TN)/(TP+TN+FN+FP)*100)
        sens[i] = ((TP)/(TP+FN)*100)
        esp[i] = ((TN)/(TN+FP)*100)

    mod = ['KNN','DecisionTree','RandomForest',"svm"]
    nam = ['acurácia','sensibilidade','especificidade']
    for i in range(4):
        print(f'{mod[i]} sua {nam[0]} é de {np.round(acc[i],2)}%')
    for i in range(4):
        print(f'{mod[i]} sua {nam[1]} é de {np.round(sens[i],2)}%')
    for i in range(4):
        print(f'{mod[i]} sua {nam[2]} é de {np.round(esp[i],2)}%')
    

    # Plots

    # TODO: Verify theses instructions
    limit = int(((n_splits-1)/n_splits)*listPerson.shape[0])
    X_train, X_test = listPerson.iloc[:limit], listPerson.iloc[limit:]
    y_train, y_test = listDiagnosis[:limit], listDiagnosis[limit:]
    X_train= ss.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns = ['velocityWeighted','pressureWeighted','CISP'])
    X_test = ss.transform(X_test)
    X_test = pd.DataFrame(X_test, columns = ['velocityWeighted','pressureWeighted','CISP'])

    plot_learning_curves(X_train, y_train, X_test, y_test, rfc)
    plt.show()
'''