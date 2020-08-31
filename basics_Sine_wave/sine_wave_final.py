import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=2)

x = np.linspace(0,50,501)
y = np.sin(x)
df = pd.DataFrame(data=y, index=x, columns=['sine'])

test_percentage = 0.1
test_point = np.round(len(df)*test_percentage)
test_ind = int(len(df)-test_point)

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

scaler = MinMaxScaler()

scaler.fit(train)
scaler_train = scaler.transform(train)

scaler.fit(train)
full_scaler = scaler.transform(df)
full_scaler_test = scaler.transform(df)

main_gen_length = 50
test_gen_length = 49
batch_size = 1

generator = TimeseriesGenerator(full_scaler,full_scaler,length=main_gen_length,batch_size=batch_size)

features = 1

model = Sequential()
model.add(SimpleRNN(50, input_shape=(main_gen_length,features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())

file_name_today = str(time.time())
checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

checking_user = int(input("----------------------------------------------------------------------------\n"
                          "use the below command to build or load your model \n"
                          "1. build your model (recommended if your are running for the first-time) \n"
                          "2. load your model (recommended if already building is done) \n"
                          "----------------------------------------------------------------------------\n"
                          "Answer : "))

# please run this command for the first time
if checking_user == 1:
    model.fit_generator(generator, epochs=20, callbacks=[cp_callback])
    loss = pd.DataFrame(model.history.history)
    print("loss of the given data : \n",loss,"\n thank-you the model has been built")
elif checking_user == 2:
    model.load_weights(checkpoint_path)
else:
    print("please choose correct option and run again ")

full_pred = []
first_eval_batch = scaler_train[-main_gen_length:]
curret_batch = first_eval_batch.reshape(1,main_gen_length,features)

for i in range(25):
    curret_pred = model.predict(curret_batch)[0]
    full_pred.append(curret_pred)
    curret_batch= np.append(curret_batch[:,1:,:],[[curret_pred]],axis=1)

full_pred = scaler.inverse_transform(full_pred)
full_pred_index = np.arange(50.1,52.6,step=0.1)

userinput = int(input("----------------------------------------------------------------------------\n"
                      "please choose the below options. \n"
                      "1.display the graph of the prediction \n"
                      "        -note: this will show prediction under one graph \n"
                      "2.display the graph of the prediction seperate \n"
                      "        -note: this will show prediction in seperate graphs \n"
                      "3.display every thing in pandas-dataframe\n"
                      "----------------------------------------------------------------------------\n"
                      "Answer : "))
if userinput == 3:
    # run this to print the data
    print(full_pred)
    print(full_pred_index)
    print(df)
elif userinput == 2:
    # run this to plt the data
    a = plt.plot(df)
    b = plt.plot(full_pred_index, full_pred)
    plt.show()
elif userinput == 1:
    # run this to plot the data and join the predicted data
    dull_conc = pd.concat(full_pred_index, full_pred)
    plt.show(full_pred.plot())
    plt.show(test.plot())
else:
    print("please choose correct option and run again ")
