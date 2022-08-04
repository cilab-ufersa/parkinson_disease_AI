import numpy as np
import pandas as pd


def listPeopleArticle(paths, fs, testID):
    """
    Retorna uma lista de paciêntes em dataFrame com a adição da coluna de velocidades e deslocamento.

            Agrs:
                    paths (list | str): Lista de paths de arquivos csv dos paciêntes.
                    fs (float): Frequência de amostragem.
                    testID (int): Número do teste de interesse.

            Returns: listPerson (list | dataFrame): Lista de pessoas em dataFrame feito a partir de todos os paths.
    """
    listPerson = []

    for path in paths:

        person = pd.read_csv(path, sep=';', names=["x", "y", "z", "Pressure", 'GripAngle', 'Timestamp', 'TestID'])
        df_testID = person[person.TestID == testID].copy()
        if not df_testID.empty:
            createNewColumns(df_testID, fs, testID)
            listPerson.append(df_testID)

    return listPerson


def createNewColumns(df, fs, testID):
    """
    Adiciona as colunas de velocidade e deslocamento para o dataFrame df.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.
                    fs (float): Frequência de amostragem.
                    testID (int): Número do teste de interesse.
    """
    moduleDisplacement(df, testID)
    moduleVelocity(df, fs, testID)


def moduleDisplacement(df, testID):
    """
    Calcula e adiciona a coluna deslocamento ao dataFrame df.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.
                    testID (int): Número do teste de interesse.
    """
    size = df.shape[0]
    module = np.empty(size, dtype=float)
    x = df['x'].values.tolist()
    y = df['y'].values.tolist()
    for i in range(size):
        if i == size - 1:
            module[i] = 0
        else:
            module[i] = ((x[i + 1] - x[i]) ** 2 + (y[i + 1] - y[i]) ** 2) ** (1 / 2)
    df.insert(3, f'displacement_{testID}', module)


def moduleVelocity(df, fs, testID):
    """
    Calcula e adiciona a coluna velocidade ao dataFrame df.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.
                    testID (int): Número do teste de interesse.
    """
    size = df.shape[0]
    module = np.empty(size, dtype=float)
    displacement = df[f"displacement_{testID}"].values.tolist()
    for i in range(size):
        module[i] = displacement[i] * fs

    df[f"velocity_{testID}"] = module


def weightedAverage(valuesList, weightList):
    """
    Retorna o cálculo da média ponderada entre duas listas.

            Agrs:
                    valuesList (list | float): Lista contendo os valores do qual se deseja obter a média.
                    weightList (list | float): Lista contendo os pesos.

            Returns: average (float): Média ponderada calculada a partir das lista valuesList e weightList
    """
    multi = np.dot(valuesList, weightList)
    average = multi / weightList.sum()
    return average


def createFeatures(listPerson, diagnoses, testID):
    """
    Retorna uma tupla com duas posições, na primeira posição tem o dataFrame com os atributos de cada paciente já
    calculados e na segunda posição conte uma lista com o diagnóstico real de cada paciênte.

            Agrs:
                    listPerson (list | dataFrame): Lista de pessoas em dataFrame.
                    diagnoses (int): Diaginóstico real dos paciêntes.
                    testID (int): Número do teste de interesse.

            Returns: infos (tuple): tupla com duas posições, na primeira posição tem o dataFrame com os atributos de
            cada paciente já calculados a partir dos dados da listPerson e na segunda posição conte uma lista com o
            diagnóstico real de cada paciênte passado como argumento pela variavel diagnoses.
    """
    table = np.empty((len(listPerson), 3))
    for i in range(len(listPerson)):
        table[i][0] = weightedAverage(listPerson[i][f'velocity_{testID}'], listPerson[i][f'displacement_{testID}'])
        table[i][1] = weightedAverage(listPerson[i]['Pressure'], listPerson[i][f'displacement_{testID}'])
        table[i][2] = table[i][0] * table[i][1]
    df = pd.DataFrame(data=table, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    infos = df, np.full((len(listPerson)), fill_value=diagnoses)
    return infos
