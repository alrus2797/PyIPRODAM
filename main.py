import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates

from mlxtend.data import wine_data
import numpy as np


# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
cols =  ['Class', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols', 
         'Flavanoids', 'Nonflavanoid', 'Proanthocyanins', 'ColorIntensity', 
         'Hue', 'OD280/OD315', 'Proline']
# data = pd.read_csv(url, names=cols)

#print (data['Class'])

X, y = wine_data()
# print (np.unique(y))
# print (np.bincount(y))
# print (X)
# print (type(X))

# for i in range(len(cols)):
#     print cols[i], '\t',
# print 

# for j in range(178):
#     print '\t',
#     for i in range(len(X[0])):
#         print X[j][i], '\t\t',
#     print

animalesFile = open('animales.out','r')
animales = {}
X = []
y = []
for line in animalesFile.readlines():
	
    if line != "igl::eigs RESTART \n":
		l =  line.split(' ')
		try: 
			if l[2:-1][0] == 'nan':
				print "asdsad"
				continue
		except:
			continue
		animal, gps = l[0], map(float,l[2:-1])
		print (animal,len(gps))
		if len(gps) == 20 and gps[1] != 'nan': 
			X.append(gps)
			animales [animal] = gps
			nombre = ''.join([i for i in animal[:-4] if not i.isdigit()])
			y.append(nombre)


X = np.array(X)
y = np.array(y)






animalesList = set()

for animal,gps in animales.items():
    nombre = ''.join([i for i in animal[:-4] if not i.isdigit()])
    animalesList.add(nombre)
    #print nombre

animalesList = list(animalesList)
#print (hola)
print (animalesList)



X_norm = (X - X.min()) / (X.max() - X.min())

lda = LDA(n_components=2) #2-dimensional LDA
lda_transformed = pd.DataFrame(lda.fit_transform(X_norm, y))


print (lda_transformed[y=='cat'])

# Plot all three series

for animal in animalesList:
    plt.scatter(lda_transformed[y==animal][0], lda_transformed[y==animal][1], label=animal)

# Display legend and show plot
plt.legend(loc=3)
plt.show()











for animal, gps in animales.items():
    pass
