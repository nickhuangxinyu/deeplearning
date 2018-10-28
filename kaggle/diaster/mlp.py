from handle_data import *

model=Sequential()
model.add(Dense(128, activation='tanh', input_shape=(1, 7)))
model.add(Dense(32, activation='tanh'))
model.add(Dense(2, activation='softmax'))
y = np_utils.to_categorical(labels)
y = y.reshape(y.shape[0], 1, y.shape[1])
data = np.array(train_data).reshape(train_data.shape[0], 1, train_data.shape[1])
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.006),metrics=['accuracy'])
model.fit(data, y, epochs = 60000, batch_size=32)
te = np.array(test_data).reshape(test_data.shape[0], 1, test_data.shape[1])
p=model.predict_classes(te)

output['Survived'] = p.flatten()
output.to_csv('out.csv', index=False)
