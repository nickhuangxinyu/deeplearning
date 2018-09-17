import pickle
import numpy as np
import time
import sys

start_sec = time.time()

line_content=open('train.tsv').read().splitlines()[1:]
test_line_content=open('test.tsv').read().splitlines()[1:]
sents = []
labels = []

for line in line_content:
  v = line.split('\t')
  sents.append(v[-2])
  labels.append(int(v[-1]))

test_sents = []
test_no = []
for line in test_line_content:
  v = line.split('\t')
  test_sents.append(v[-1])
  test_no.append(int(v[0]))

useless_char = [',', '.', ':']
def BuildDict(l):
  d = {}
  count = 0
  for i in l:
    for uc in useless_char:
      i.replace(uc, '')
    iv = i.split(' ')
    for word in iv:
      if word not in d:
        d[word] = count
        count = count + 1
  return d

d = BuildDict(sents+test_sents)
train = np.array([[0]*len(d)]*len(sents))
test = np.array([[0]*len(d)]*len(test_sents))
print('train shape is ', train.shape)
print('test shape is ', test.shape)
for i in range(len(sents)):
  s = sents[i]
  for uc in useless_char:
    s.replace(uc, '')
  iv = s.split(' ')
  for word in iv:
    train[i][d[word]] += 1

for i in range(len(test_sents)):
  s = test_sents[i]
  for uc in useless_char:
    s.replace(uc, '')
  iv = s.split(' ')
  for word in iv:
    test[i][d[word]] += 1


wd = open('word_dict.pickle', 'wb')
pickle.dump(d, wd, protocol=pickle.HIGHEST_PROTOCOL)
wd.close()

sl = open('seq_list.pickle', 'wb')
pickle.dump(sents, sl, protocol=pickle.HIGHEST_PROTOCOL)
sl.close()

lab = open('labels.pickle', 'wb')
pickle.dump(labels, lab, protocol=pickle.HIGHEST_PROTOCOL)
lab.close()

tr = open('train.pickle', 'wb')
pickle.dump(train, tr, protocol=pickle.HIGHEST_PROTOCOL)
tr.close()

te = open('test.pickle', 'wb')
pickle.dump(test, te, protocol=pickle.HIGHEST_PROTOCOL)
te.close()

ten = open('testno.pickle', 'wb')
pickle.dump(test_no, ten, protocol=pickle.HIGHEST_PROTOCOL)
ten.close()

print('running time is ', time.time()-start_sec)
