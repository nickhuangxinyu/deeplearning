import random
from sklearn import neighbors
import numpy as np
#from numpy import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x1 = np.random.normal(50, 6, 200)
y1 = np.random.normal(5, 0.5, 200)

x2 = np.random.normal(30,6,200)
y2 = np.random.normal(4,0.5,200)

x3 = np.random.normal(45,6,200)
y3 = np.random.normal(2.5, 0.5, 200)

plt.scatter(x1,y1,c='b',marker='s',s=50,alpha=0.8)
plt.scatter(x2,y2,c='r', marker='^', s=50, alpha=0.8)
plt.scatter(x3,y3, c='g', s=50, alpha=0.8)

x_val = np.concatenate((x1,x2,x3))
y_val = np.concatenate((y1,y2,y3))

x_diff = max(x_val)-min(x_val)
y_diff = max(y_val)-min(y_val)

x_normalized = [x/(x_diff) for x in x_val]
y_normalized = [y/(y_diff) for y in y_val]
xy_normalized = zip(x_normalized,y_normalized)

l=[]
for i in xy_normalized:
  l.append([i[0],i[1]])

la = np.array(l)

labels = [1]*200+[2]*200+[3]*200

clf = neighbors.KNeighborsClassifier(5)

clf.fit(la, labels)

nearests = clf.kneighbors([(50/x_diff, 5/y_diff),(30/x_diff, 3/y_diff)], 10, False)

prediction = clf.predict([(50/x_diff, 5/y_diff),(30/x_diff, 3/y_diff)])

x1_test = np.random.normal(50, 6, 100)
y1_test = np.random.normal(5, 0.5, 100)

x2_test = np.random.normal(30,6,100)
y2_test = np.random.normal(4,0.5,100)

x3_test = np.random.normal(45,6,100)
y3_test = np.random.normal(2.5, 0.5, 100)

xy_test_normalized = zip(np.concatenate((x1_test,x2_test,x3_test))/x_diff,\
                        np.concatenate((y1_test,y2_test,y3_test))/y_diff)

labels_test = [1]*100+[2]*100+[3]*100
l_test = []

for i in xy_test_normalized:
  l_test.append([i[0], i[1]])

la_test = np.array(l_test)

score = clf.score(la_test, labels_test)

print('score is ', score)
