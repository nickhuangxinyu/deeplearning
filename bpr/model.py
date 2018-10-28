from handle_data import *
import time

def gen_test():
  t = []
  u_array = []
  view_array = []
  unview_array = []
  for u in user_ratings.keys():
    i = user_ratings_test[u]
    for j in range(1, item_count+1):
      if not (j in user_ratings[u]):
        u_array.append(u)
        view_array.append(i)
        unview_array.append(j)
  return [np.array(u_array), np.array(view_array), np.array(unview_array)]

regularize_rate = 0.001
k = 40
batch_size = 512

model_user = Sequential()
model_item = Sequential()
user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
item_input1 = Input(shape=(1,), dtype='int32', name = 'item_input1')
item_input2 = Input(shape=(1,), dtype='int32', name = 'item_input2')

user_embd = Embedding(user_count+1, k, input_length=1, embeddings_regularizer = regularizers.l2(regularize_rate))
item_embd = Embedding(item_count+1, k, input_length=1, embeddings_regularizer = regularizers.l2(regularize_rate))

user_latent = Flatten()(user_embd(user_input))
item_latent1 = Flatten()(item_embd(item_input1))
item_latent2 = Flatten()(item_embd(item_input2))
item_latent = subtract([item_latent1, item_latent2])

Xuij = multiply([user_latent, item_latent])
op = Lambda(lambda x: K.sum(x))(Xuij)
model = Model(inputs=[user_input, item_input1, item_input2], outputs=op)

def myloss(y_true, y_pred):
  return -K.sum(K.log(K.sigmoid(K.sum(y_pred))))

model.compile(optimizer=Adam(), loss=myloss)
model.summary()

epochs = 10000
for i in range(epochs):
  train_batch = generate_train_batch().transpose()
  a = model.fit([np.array(train_batch[0]).transpose(), np.array(train_batch[1]).transpose() ,np.array(train_batch[2]).transpose()], [1]*batch_size, verbose = 0)
  if i % 100 ==0:
    pred = (model.predict(gen_test()) > 0).mean()
    print str(i) + 'th iteration: acc is ' + str(pred)+' loss is ' + str(a.history['loss'])
