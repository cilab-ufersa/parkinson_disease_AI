import numpy as np
import pandas as pd


def listPeopleArticle(paths, fs, testID):
    """
    Retorna uma lista de paciêntes em dataFrame com a adição da coluna de velocidades e deslocamento.

            Agrs:
                    paths (list | str): Lista de paths de arquivos csv dos paciêntes.
                    fs (float): Frequência de amostragem.
                    testID (int): Número do teste de interesse.

            Returns: listPerson (list): Lista de pessoas em dataFrame feito a partir de todos os paths.
    """
    listPerson = []

    for path in paths:

        person = pd.read_csv(path, sep=';', names=["x", "y", "z", "Pressure", 'GripAngle', 'Timestamp', 'TestID'])
        df_testID = person[person.TestID == testID].copy()
        if not df_testID.empty:
            infos = getSegmentsInfos(df_testID, fs)
            if len(infos) != 0:
                listPerson.append(infos)

    return listPerson


def getSegmentsInfos(df, fs):
    """
    Obtem informações de distância, velocidade e pressão para cada segmento.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.
                    fs (float): Frequência de amostragem.

            Returns: infos (list): Lista de distância, velocidade média e pressão média para cada segmento.
    """
    segments = segmentsTracker(df)
    infos = np.zeros((len(segments), 3))
    if len(segments) != 0:
        moduleDisplacement(df, infos, segments)
        moduleVelocity(infos, segments, fs)
        modulePresure(df, infos, segments)
    else:
        pass

    infos = deleteShortPath(infos)

    return infos


def segmentsTracker(df):
    """
    Obtem todos os segmentos realizados num teste.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.

            Returns: segments (list): Lista de segmentos informando o index de início e fim de cada segmento.
    """
    indexes = df.loc[df["Pressure"] > 0].index
    size = indexes.shape[0]
    segments = []
    if size == 0:
        return segments
    initialPoint = True
    segment = 0
    for i in range(size):
        if initialPoint:
            segment = indexes[i]
            initialPoint = False
        else:
            if i < size - 1:
                if indexes[i] + 1 != indexes[i + 1]:
                    segments.append((segment, indexes[i]))
                    initialPoint = True
            else:
                segments.append((segment, indexes[i]))
                initialPoint = True

    return segments


def moduleDisplacement(df, infos, segments):
    """
    Obtem a distância total percorrida em cada segmento.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.
                    infos (matrix): Lista de informações(distancia percorrida, velocidade média e pressão média)
                    segments (list | tuple): Lista de tuplas contendo os index de início e fim de um segmento.
    """
    size = len(segments)
    x = df['x']
    y = df['y']
    for i in range(size):
        for j in range(segments[i][0], segments[i][1]):
            infos[i][0] += ((x[j + 1] - x[j]) ** 2 + (y[j + 1] - y[j]) ** 2) ** (1 / 2)


def moduleVelocity(infos, segments, fs):
    """
    Obtem a velocidade média em cada segmento.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.
                    segments (list | tuple): Lista de tuplas contendo os index de início e fim de um segmento.
                    fs (float): Frequência de amostragem.
    """
    size = len(segments)
    for i in range(size):
        infos[i][1] = fs * infos[i][0] / (segments[i][1] - segments[i][0])


def modulePresure(df, infos, segments):
    """
    Obtem a pressão média em cada segmento.

            Agrs:
                    df (dataFrame): dataFrame contendo todas as informações sobre o paciênte.
                    infos (matrix): Lista de informações(distancia percorrida, velocidade média e pressão média)
                    segments (list | tuple): Lista de tuplas contendo os index de início e fim de um segmento.
    """
    size = len(segments)
    pressure = df['Pressure']
    for i in range(size):
        comulative = 0
        for j in range(segments[i][0], segments[i][1] + 1):
            comulative += pressure[j]
        infos[i][2] = comulative / (segments[i][1] - segments[i][0] + 1)


def deleteShortPath(infos):
    """
    Remove segmentos com distância total menor que 0,5 unidades de distância.

            Agrs:
                    infos (matrix): Lista de informações(distancia percorrida, velocidade média e pressão média)
    """
    delete = []
    for i in range(len(infos)):
        if infos[i][0] == 0:
            delete.append(i)
    return np.delete(infos, delete, 0)


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


def createFeatures(listPerson, diagnoses):
    """
    Retorna uma tupla com duas posições, na primeira posição tem o dataFrame com os atributos de cada paciente já
    calculados e na segunda posição conte uma lista com o diagnóstico real de cada paciênte.

            Agrs:
                    listPerson (list | dataFrame): Lista de pessoas em dataFrame.
                    diagnoses (int): Diaginóstico real dos paciêntes.

            Returns: infos (tuple): tupla com duas posições, na primeira posição tem o dataFrame com os atributos de
            cada paciente já calculados a partir dos dados da listPerson e na segunda posição conte uma lista com o
            diagnóstico real de cada paciênte passado como argumento pela variavel diagnoses.
    """
    table = np.empty((len(listPerson), 3))
    for i in range(len(listPerson)):
        transpose = listPerson[i].transpose()
        table[i][0] = weightedAverage(transpose[1][:], transpose[0][:])
        table[i][1] = weightedAverage(transpose[2][:], transpose[0][:])
        table[i][2] = table[i][0] * table[i][1]
    df = pd.DataFrame(data=table, columns=['velocityWeighted', 'pressureWeighted', 'CISP'])
    infos = df, np.full((len(listPerson)), fill_value=diagnoses)
    return infos
