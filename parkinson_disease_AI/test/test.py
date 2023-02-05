import statistics

import pandas as pd
from matplotlib import pyplot as plt

from parkinson_disease_AI.utils import addNoise, to_csv, listPeopleArticle

dataset = pd.read_csv('..\dataset\listPerson.csv')

dataset1 = dataset[76:]

listDiagnosis1 = dataset["Diagnosis"].to_numpy()[:15]
listPerson1 = dataset[["velocityWeighted", "pressureWeighted", "CISP"]][:15]

listDiagnosis1p = dataset["Diagnosis"].to_numpy()[15:76]
listPerson1p = dataset[["velocityWeighted", "pressureWeighted", "CISP"]][15:76]

listDiagnosis2 = dataset["Diagnosis"].to_numpy()[76:91]
listPerson2 = dataset[["velocityWeighted", "pressureWeighted", "CISP"]][76:91]

listDiagnosis23p = dataset["Diagnosis"].to_numpy()[91:]
listPerson23p = dataset[["velocityWeighted", "pressureWeighted", "CISP"]][91:]


plt.plot(listPerson1["pressureWeighted"],listPerson1["velocityWeighted"], 'or', label='Test 1')
plt.plot(listPerson2["pressureWeighted"],listPerson2["velocityWeighted"], 'xb', label='Test 2')
plt.legend()
plt.ylabel('Velocity')
plt.xlabel('Pressure')
plt.title('Test 1 and 2 from without PD')
plt.show()

plt.plot(listPerson1p["pressureWeighted"],listPerson1p["velocityWeighted"], 'or', label='Test 1')
plt.plot(listPerson23p["pressureWeighted"],listPerson23p["velocityWeighted"], 'xb', label='Test 2 and 3')
plt.legend()
plt.ylabel('Velocity')
plt.xlabel('Pressure')
plt.title('Test 1, 2 and 3 from with PD')
plt.show()

plt.plot(listPerson23p["pressureWeighted"],listPerson23p["velocityWeighted"], 'or', label='with PD')
plt.plot(listPerson2["pressureWeighted"],listPerson2["velocityWeighted"], 'xb', label='without PD')
plt.legend()
plt.ylabel('Velocity')
plt.xlabel('Pressure')
plt.title('Test 2')
plt.show()

#listPerson2.drop(listPerson2[listPerson2["pressureWeighted"] < 600].index, inplace=True)
#listPerson23p.drop(listPerson23p[(listPerson23p["velocityWeighted"] > 400) | (listPerson23p["velocityWeighted"] < 40)].index, inplace=True)

plt.plot(listPerson23p["pressureWeighted"],listPerson23p["velocityWeighted"], 'or', label='with PD')
plt.plot(listPerson2["pressureWeighted"],listPerson2["velocityWeighted"], 'xb', label='without PD')
plt.legend()
plt.ylabel('Velocity')
plt.xlabel('Pressure')
plt.title('Test 2')
plt.show()

qtd = 2000

dataset1 = dataset1.reset_index(drop=True)

#dataset1.drop(dataset1[(dataset1["pressureWeighted"] < 600) & (dataset1["Diagnosis"] == 0)].index, inplace=True)
#dataset1.drop(dataset1[((dataset1["velocityWeighted"] > 400) | (dataset1["velocityWeighted"] < 40)) & (dataset1["Diagnosis"] == 1)].index, inplace=True)

dataset1 = addNoise(dataset1, qtd)

dataset1 = dataset1.reset_index(drop=True)

listPerson1 = dataset1[["velocityWeighted", "pressureWeighted", "CISP"]][:15]
listPerson11 = dataset1[["velocityWeighted", "pressureWeighted", "CISP"]][107: 107 + qtd - 15]
listPerson1 = pd.concat([listPerson1 , listPerson11])
listPerson1p = dataset1[["velocityWeighted", "pressureWeighted", "CISP"]][15:107]
listPerson11p = dataset1[["velocityWeighted", "pressureWeighted", "CISP"]][107 + qtd - 15:]
listPerson1p = pd.concat([listPerson1p , listPerson11p])


plt.plot(listPerson1p["pressureWeighted"],listPerson1p["velocityWeighted"], 'or', label='with PD')
plt.plot(listPerson1["pressureWeighted"],listPerson1["velocityWeighted"], 'xb', label='without PD')
plt.legend()
plt.ylabel('Velocity')
plt.xlabel('Pressure')
plt.title('Test 1')
plt.show()

to_csv(dataset1, 'testingdataset')