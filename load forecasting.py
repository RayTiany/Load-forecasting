import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# Data preprocessing
df = pd.read_csv('cb.csv')
df["Time"] = pd.to_datetime(df["Time"])
df = df.groupby([pd.Grouper(key='Time',freq="30min")]).sum()
df.reset_index(level=0, inplace=True)
df["week_day"] = df['Time'].dt.day_name()

Monday_set = df[df["week_day"] == "Monday"]
Monday = Monday_set["instant_load"].values

# Setting time stamp
time_stamp = 3

# Normalization
scaler = MinMaxScaler(feature_range=(0, 1))
Monday = scaler.fit_transform(Monday.reshape(-1, 1)).reshape(-1, 48)

# Create features and labels
def create_dataset(dataset, look_back = 1):
    X = []
    Y = []
    for i in range(len(dataset) - look_back):
        part = dataset[i: (i + look_back)]
        X.append(part)
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)

x_mon, y_mon = create_dataset(Monday, time_stamp)

# Model fit
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(x_mon.shape[1:])))
model.add(Dropout(0.2))

model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(48))

model.compile(optimizer="adam", loss='mse',metrics=['accuracy'])

history = model.fit(x_mon, y_mon,
                    batch_size = 1,
                    epochs=100)

plt.plot(history.history['loss'],label='training loss')

# Prediction
MonPredict = model.predict(x_mon)

# Transfrom data back into original form
MonPredict = scaler.inverse_transform(MonPredict)
MonPredict[MonPredict < 0] = 0
MonY = scaler.inverse_transform(y_mon)

# Setting Chinese Character
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Visualization
plt.figure(figsize=(15,6))
plt.style.use('ggplot')
my_xticks = list(range(0,24))
plt.xticks(np.arange(0, len(MonPredict[0])+1, len(MonPredict[0])/len(my_xticks)),my_xticks, rotation=45, ha='right', fontsize='12')
plt.plot(MonPredict[0], color = "tab:purple", label='真实流量')
plt.plot(MonY[0], color = "gold", label='预测流量')
plt.xlabel('当日时间', fontsize='14', ha='center')
plt.ylabel('瞬时流量', fontsize='14', ha='center')
plt.title("周一流量预测：以8/2为例", fontsize='16', fontweight = 'bold')
plt.legend(loc='best')
plt.show()

# Saving model
# tf.saved_model.save(model, "models3")