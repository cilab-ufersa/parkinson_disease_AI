from parkinson_disease_AI.utils.utils import listPeopleArticle, createFeatures, to_csv

import pandas as pd

paths_non_parkinson = [
    "../dataset/data_motion/hw_dataset/control/C_0001.txt",
    "../dataset/data_motion/hw_dataset/control/C_0002.txt",
    "../dataset/data_motion/hw_dataset/control/C_0003.txt",
    "../dataset/data_motion/hw_dataset/control/C_0004.txt",
    "../dataset/data_motion/hw_dataset/control/C_0005.txt",
    "../dataset/data_motion/hw_dataset/control/C_0006.txt",
    "../dataset/data_motion/hw_dataset/control/C_0007.txt",
    "../dataset/data_motion/hw_dataset/control/C_0008.txt",
    "../dataset/data_motion/hw_dataset/control/C_0009.txt",
    "../dataset/data_motion/hw_dataset/control/C_0010.txt",
    "../dataset/data_motion/hw_dataset/control/C_0011.txt",
    "../dataset/data_motion/hw_dataset/control/C_0012.txt",
    "../dataset/data_motion/hw_dataset/control/C_0013.txt",
    "../dataset/data_motion/hw_dataset/control/C_0014.txt",
    "../dataset/data_motion/hw_dataset/control/C_0015.txt"
]

info_non_parkinson_0 = listPeopleArticle(paths_non_parkinson, 0)
info_non_parkinson_1 = listPeopleArticle(paths_non_parkinson, 1)
info_non_parkinson_2 = listPeopleArticle(paths_non_parkinson, 2)

df_non_parkinson_0 = createFeatures(info_non_parkinson_0, 0)
df_non_parkinson_1 = createFeatures(info_non_parkinson_1, 0)
df_non_parkinson_2 = createFeatures(info_non_parkinson_2, 0)
df_non_parkinson_0[0]['Diagnosis'] = df_non_parkinson_0[1].tolist()
df_non_parkinson_1[0]['Diagnosis'] = df_non_parkinson_1[1].tolist()
df_non_parkinson_1[0]['Diagnosis'] = df_non_parkinson_1[1].tolist()

paths_parkinson = [
    "../dataset/data_motion/hw_dataset/parkinson/P_02100001.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_02100002.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_05060003.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_05060004.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_09100001.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_09100003.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_09100005.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_11120003.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_11120004.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_11120005.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_12060001.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_12060002.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_16100003.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_16100004.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_23100002.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_23100003.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_26060001.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_26060002.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_26060003.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_26060006.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_26060007.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_27110001.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_27110003.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_30100001.txt",
    "../dataset/data_motion/hw_dataset/parkinson/P_30100002.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0001.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0002.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0003.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0004.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0007.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0008.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_p000-0010.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0011.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0012.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0013.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0014.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0015.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0016.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_p000-0017.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_p000-0018.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0019.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0020.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0021.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0022.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0023.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0024.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0025.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_p000-0028.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0029.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0030.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0031.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0032.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0033.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0034.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0035.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0036.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0037.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0039.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_P000-0040.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_p000-0041.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_p000-0042.txt",
    "../dataset/data_motion/new_dataset/parkinson/H_p000-0043.txt"
]

info_parkinson_0 = listPeopleArticle(paths_parkinson, 0)
info_parkinson_1 = listPeopleArticle(paths_parkinson, 1)
info_parkinson_2 = listPeopleArticle(paths_parkinson, 2)

df_parkinson_0 = createFeatures(info_parkinson_0, 1)
df_parkinson_1 = createFeatures(info_parkinson_1, 1)
df_parkinson_2 = createFeatures(info_parkinson_2, 1)

df_parkinson_0[0]['Diagnosis'] = df_parkinson_0[1].tolist()
df_parkinson_1[0]['Diagnosis'] = df_parkinson_1[1].tolist()
df_parkinson_2[0]['Diagnosis'] = df_parkinson_2[1].tolist()

df = pd.concat(
    [df_non_parkinson_0[0], df_parkinson_0[0], df_non_parkinson_1[0], df_parkinson_1[0], df_non_parkinson_2[0],
     df_parkinson_2[0]])

to_csv(df, "listPersonRefactor")
