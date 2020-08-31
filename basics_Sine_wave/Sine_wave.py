"""
this is the file that show in detail how rnn works

uncomment print statments to see the data filtration process and
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,LSTM
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=2)

#creating a line space with numpy with linspace
x = np.linspace(0,50,501)
# print(x)
y = np.sin(x)
# print(y)
df = pd.DataFrame(data=y, index=x, columns=['sine'])
# print(f)
# plt.show(plt.plot(y))
test_percentage = 0.1
test_point = np.round(len(df)*test_percentage)
# print(test_point)
test_ind = int(len(df)-test_point)

train = df.iloc[:test_ind]
# print(train)
test = df.iloc[test_ind:]
# print(test)

scaler = MinMaxScaler()

scaler.fit(train)
scaler_train = scaler.transform(train)
scaler_test = scaler.transform(test)

# help(TimeseriesGenerator)
#this is the number of points the code is going to look into to in sequence to predict the
#next point
#it should not to arbetrory it should be based of a certen positional or sesional value
# u can plot the data and check the trends and give the length
#larger the length the longer the trainning time
#when your are passing test data your lenth should be smaller than the train data
main_gen_length = 50
test_gen_length = 49

batch_size = 1
generator = TimeseriesGenerator(scaler_train,scaler_train,length=main_gen_length,batch_size=batch_size)
valadition_gen = TimeseriesGenerator(scaler_test,scaler_test,length=test_gen_length,batch_size=batch_size)
# print(len(scaler_train))
# print(len(generator))
#X is the length and y is the future prediction point that it is going to predict
# X,y = generator[0]
# plt.show(df.plot())
features = 1

# creating you model
model = Sequential()

model.add(SimpleRNN(50, input_shape=(main_gen_length,features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())

model.fit_generator(generator,epochs=20, validation_data=valadition_gen, callbacks=[early_stop])

losses = pd.DataFrame(model.history.history)

#loss_plot = losses.plot()
print(losses)

first_eval_batch = scaler_train[-main_gen_length:]
first_eval_batch=first_eval_batch.reshape(1,main_gen_length,features)
print(model.predict(first_eval_batch))

test_pred = []
first_eval_batch = scaler_train[-main_gen_length:]
curret_batch = first_eval_batch.reshape(1,main_gen_length,features)

for i in range(len(test)):
    curret_pred = model.predict(curret_batch)[0]
    test_pred.append(curret_pred)
    curret_batch= np.append(curret_batch[:,1:,:],[[curret_pred]],axis=1)

# print(test_pred)

test_pred = scaler.inverse_transform(test_pred)
test['lstm_prediction'] = test_pred
print(test)

#to plot the data
# plt.show(test.plot())


