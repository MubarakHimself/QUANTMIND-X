---
title: Triangular arbitrage with predictions
url: https://www.mql5.com/en/articles/14873
categories: Trading, Trading Systems, Machine Learning
relevance_score: 0
scraped_at: 2026-01-24T13:29:23.764322
---

[![](https://www.mql5.com/ff/sh/zf7a2k61x98jzh89z2/01.png)Speed up your tradingUse our high-speed VPS for MetaTrader 4 and 5Learn more](https://www.mql5.com/ff/go?link=https://www.mql5.com/en/vps&a=qtrrsuiwuicrscmckjynyanztbditglq&s=c617dc80d90cfd3783ec1345eec2b419b281f10fec6eac77b3218984ac337259&uid=&ref=https://www.mql5.com/en/articles/14873&id=bfogggabsofabcpxuzmgaibarmaxasdrj&fz_uniq=5082898769290203394)

MetaTrader 5 / Trading


### Introduction

This article is about the triangular arbitrage strategy. It has an example of two triangular arbitrages which are based on deep learning models. These models and the used EA are available in the attachments to the article. Triangular arbitrage leverages discrepancies in exchange rates to generate risk-free profits.

### What is triangular arbitrage?

Arbitrage is very curious, it's been prohibited from the bookies of sports betting. Imagine you have some winning odds of 1.25 for Real Madrid to win the champions 2024, and Borussia Dortmund has 3.60 odds, that means Madrid has 100/1.25 =  80 % of probabilities to win and Borussia 27.7  % to win. If you add those two, you have 107.7%, that is because bookies want to win money and that over 100 % is their commission. But, imagine you find Bookie number 2 and hey offer odds for Borussia of 19% probabilities to win, odds of 5.26. Then you could bet in Bookie number 1 to Real Madrid and Bookie number 2 for Borussia, and if you bet the appropriate quantity to each team, you will win money in the game, because both add less than 100%. This is a simple way to explain why its prohibited in sports betting and what is arbitrage.

Imagine you are a "legal" person and you don't want to have your sports account closed by doing arbitrage, you know that even if you bet for Madrid, you could do "legal" arbitrage if you waited for minute 70' of the game if draw or wait to Real Madrid to score to have those odds for Borussia and have a win win... this seems a bit risky, but here is where we can take advantage of Deep Learning, we know Real Madrid is gonna score, so you are gonna have those odds with a 98 % of probabilities (we know this with cointegration between the predictions and the real values). This is what's new with Deep Learning and Arbitrage.

So, now that we know what arbitrage and is how we win more with the help of Deep Learning, what is triangular arbitrage? Well, it's the same as arbitrage, but using three pairs. Why? Because it's used in forex and cryptos which use this formula for a symbol A / B, and if you have to solve this you need three equations (A / B ) \* (B/C) \* (C/A), so when this is >1 you multiply the right way, and when <1 the left way.

### Why can or can't you do triangular arbitrage with all accounts?

If you have a zero spread account, the triangular arbitrage would be done in one second or less. If you have spread, it's impossible to beat the spread in such times. But, as I said before, don't worry, this EA is really profitable in both ways. My account is not zero spread, so this article will have an example with spreads.

### What do we need for this EA?

This EA uses predictions made in python to ONNX models to use them in MT5 EA's. This is why I'm going to go over the whole process to keep sure every can use this EA. If you know how to make an ONNX model, you can skip to the EA.

You will have to install for your first time:

\- Python 3.10

You can find this in Microsoft's Store, just click install.

![python 3.10](https://c.mql5.com/2/77/python_redimensionado.png)

\- Visual Studio Code

You can find this in Microsoft's Store, just click install, and it will do everything for you.

![VSC](https://c.mql5.com/2/77/vsc_escalado.png)

After this, you need to install Visual Studio 2019 or C++ from here (it will be asked to be installed with one library of python):

```
https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170#visual-studio-2015-2017-2019-and-2022
```

Once this is done, you have to add the python scripts folder to the path variables. You must also add ".py" to the PATHEXT.

Once all this is done, we can now install the libraries, like this.

Open VSC -> Terminal -> New Terminal.

VSC might ask you to install python extensions (just click OK). And just copy past this (and press Enter):

```
pip install MetaTrader5==5.0.4200
pip install pandas==2.2.1
pip install scipy==1.12.0
pip install statsmodels==0.14.1
pip install numpy==1.26.4
pip install tensorflow==2.15.0
pip install tf2onnx==1.16.1
pip install scikit-learn==1.4.1.post1
pip install keras==2.15.0
pip install matplotlib==3.8.3
```

Should be no errors, if not, you can ask here.

Once all the required parts are installed and have no errors, we can proceed to the .py testing model. I will copy paste this example:

```
# python libraries
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx
from datetime import timedelta, datetime
# input parameters

symbol1 = "EURGBP"
symbol2 = "GBPUSD"
symbol3 = "EURUSD"
sample_size1 = 200000
optional = "_M1_test"
timeframe = mt5.TIMEFRAME_M1

#end_date = datetime.now()
end_date = datetime(2024, 3, 4, 0)

inp_history_size = 120

sample_size = sample_size1
symbol = symbol1
optional = optional
inp_model_name = str(symbol)+"_"+str(optional)+".onnx"

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# we will save generated onnx-file near our script to use as resource
from sys import argv
data_path=argv[0]
last_index=data_path.rfind("\\")+1
data_path=data_path[0:last_index]
print("data path to save onnx model",data_path)

# and save to MQL5\Files folder to use as file
terminal_info=mt5.terminal_info()
file_path=terminal_info.data_path+"\\MQL5\\Files\\"
print("file path to save onnx model",file_path)

# set start and end dates for history data

#end_date = datetime.now()
#end_date = datetime(2024, 5, 1, 0)
start_date = end_date - timedelta(days=inp_history_size*20)

# print start and end dates
print("data start date =",start_date)
print("data end date =",end_date)

# get rates
eurusd_rates = mt5.copy_rates_from(symbol, timeframe , end_date, sample_size )

# create dataframe
df=pd.DataFrame()
df = pd.DataFrame(eurusd_rates)
print(df)
# Extraer los precios de cierre directamente
datas = df['close'].values

"""# Calcular la inversa de cada valor
inverted_data = 1 / datas

# Convertir los datos invertidos a un array de numpy si es necesario
data = inverted_data.values"""

data = datas.reshape(-1,1)
# Imprimir los resultados
"""data = datas"""
# scale data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# training size is 80% of the data
training_size = int(len(scaled_data)*0.80)
print("Training_size:",training_size)
train_data_initial = scaled_data[0:training_size,:]
test_data_initial = scaled_data[training_size:,:1]

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
       # find the end of this pattern
       end_ix = i + n_steps
       # check if we are beyond the sequence
       if end_ix > len(sequence)-1:
          break
       # gather input and output parts of the pattern
       seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
       X.append(seq_x)
       y.append(seq_y)
    return np.array(X), np.array(y)

# split into samples
time_step = inp_history_size
x_train, y_train = split_sequence(train_data_initial, time_step)
x_test, y_test = split_sequence(test_data_initial, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
x_train =x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# define model
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras import callbacks
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu',padding = 'same',input_shape=(inp_history_size,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss= 'mse' , metrics = [rmse()])

# Set up early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

# model training for 300 epochs
history = model.fit(x_train, y_train, epochs = 300 , validation_data = (x_test,y_test), batch_size=32, callbacks=[early_stopping], verbose=2)

# evaluate training data
train_loss, train_rmse = model.evaluate(x_train,y_train, batch_size = 32)
print(f"train_loss={train_loss:.3f}")
print(f"train_rmse={train_rmse:.3f}")

# evaluate testing data
test_loss, test_rmse = model.evaluate(x_test,y_test, batch_size = 32)
print(f"test_loss={test_loss:.3f}")
print(f"test_rmse={test_rmse:.3f}")

# save model to ONNX
output_path = data_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

output_path = file_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

# finish
mt5.shutdown()
#prediction using testing data

#prediction using testing data
test_predict = model.predict(x_test)
print(test_predict)
print("longitud total de la prediccion: ", len(test_predict))
print("longitud total del sample: ", sample_size)

plot_y_test = np.array(y_test).reshape(-1, 1)  # Selecciona solo el último elemento de cada muestra de prueba
plot_y_train = y_train.reshape(-1,1)
train_predict = model.predict(x_train)
#print(plot_y_test)

#calculate metrics
from sklearn import metrics
from sklearn.metrics import r2_score
#transform data to real values
value1=scaler.inverse_transform(plot_y_test)
#print(value1)
# Escala las predicciones inversas al transformarlas a la escala original
value2 = scaler.inverse_transform(test_predict.reshape(-1, 1))
#print(value2)
#calc score
score = np.sqrt(metrics.mean_squared_error(value1,value2))

print("RMSE         : {}".format(score))
print("MSE          :", metrics.mean_squared_error(value1,value2))
print("R2 score     :",metrics.r2_score(value1,value2))

#sumarize model
model.summary()

#Print error
value11=pd.DataFrame(value1)
value22=pd.DataFrame(value2)
#print(value11)
#print(value22)

value111=value11.iloc[:,:]
value222=value22.iloc[:,:]

print("longitud salida (tandas de 1 minuto): ",len(value111) )
#print("en horas son " + str((len(value111))*60*24)+ " minutos")
print("en horas son " + str(((len(value111)))/60)+ " horas")
print("en horas son " + str(((len(value111)))/60/24)+ " dias")

# Calculate error
error = value111 - value222

import matplotlib.pyplot as plt
# Plot error
plt.figure(figsize=(10, 6))
plt.scatter(range(len(error)), error, color='blue', label='Error')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # Línea horizontal en y=0
plt.title('Error de Predicción ' + str(symbol))
plt.xlabel('Índice de la muestra')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig(str(symbol)+str(optional)+'.png')

rmse_ = format(score)
mse_ = metrics.mean_squared_error(value1,value2)
r2_ = metrics.r2_score(value1,value2)

resultados= [rmse_,mse_,r2_]

# Abre un archivo en modo escritura
with open(str(symbol)+str(optional)+"results.txt", "w") as archivo:
    # Escribe cada resultado en una línea separada
    for resultado in resultados:
        archivo.write(str(resultado) + "\n")

# finish
mt5.shutdown()

#show iteration-rmse graph for training and validation
plt.figure(figsize = (18,10))
plt.plot(history.history['root_mean_squared_error'],label='Training RMSE',color='b')
plt.plot(history.history['val_root_mean_squared_error'],label='Validation-RMSE',color='g')
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE" + str(symbol))
plt.legend()
plt.savefig(str(symbol)+str(optional)+'1.png')

#show iteration-loss graph for training and validation
plt.figure(figsize = (18,10))
plt.plot(history.history['loss'],label='Training Loss',color='b')
plt.plot(history.history['val_loss'],label='Validation-loss',color='g')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("LOSS" + str(symbol))
plt.legend()
plt.savefig(str(symbol)+str(optional)+'2.png')

#show actual vs predicted (training) graph
plt.figure(figsize=(18,10))
plt.plot(scaler.inverse_transform(plot_y_train),color = 'b', label = 'Original')
plt.plot(scaler.inverse_transform(train_predict),color='red', label = 'Predicted')
plt.title("Prediction Graph Using Training Data" + str(symbol))
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.savefig(str(symbol)+str(optional)+'3.png')

#show actual vs predicted (testing) graph
plt.figure(figsize=(18,10))
plt.plot(scaler.inverse_transform(plot_y_test),color = 'b',  label = 'Original')
plt.plot(scaler.inverse_transform(test_predict),color='g', label = 'Predicted')
plt.title("Prediction Graph Using Testing Data" + str(symbol))
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.savefig(str(symbol)+str(optional)+'4.png')

################################################################################################ EURJPY 1

# python libraries
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx

# input parameters

inp_history_size = 120

sample_size = sample_size1
symbol = symbol2
optional = optional
inp_model_name = str(symbol)+"_"+str(optional)+".onnx"

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# we will save generated onnx-file near our script to use as resource
from sys import argv
data_path=argv[0]
last_index=data_path.rfind("\\")+1
data_path=data_path[0:last_index]
print("data path to save onnx model",data_path)

# and save to MQL5\Files folder to use as file
terminal_info=mt5.terminal_info()
file_path=terminal_info.data_path+"\\MQL5\\Files\\"
print("file path to save onnx model",file_path)

# set start and end dates for history data
from datetime import timedelta, datetime
#end_date = datetime.now()
#end_date = datetime(2024, 5, 1, 0)
start_date = end_date - timedelta(days=inp_history_size*20)

# print start and end dates
print("data start date =",start_date)
print("data end date =",end_date)

# get rates
eurusd_rates2 = mt5.copy_rates_from(symbol, timeframe ,  end_date, sample_size)
# create dataframe
df=pd.DataFrame()
df2 = pd.DataFrame(eurusd_rates2)
print(df2)
# Extraer los precios de cierre directamente
datas2 = df2['close'].values

"""inverted_data = 1 / datas

# Convertir los datos invertidos a un array de numpy si es necesario
data = inverted_data.values"""
data2 = datas2.reshape(-1,1)

# Convertir los datos invertidos a un array de numpy si es necesario
#data = datas.values

# Imprimir los resultados

# scale data
from sklearn.preprocessing import MinMaxScaler
scaler2=MinMaxScaler(feature_range=(0,1))
scaled_data2 = scaler2.fit_transform(data2)

# training size is 80% of the data
training_size2 = int(len(scaled_data2)*0.80)
print("Training_size:",training_size2)
train_data_initial2 = scaled_data2[0:training_size2,:]
test_data_initial2 = scaled_data2[training_size2:,:1]

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
       # find the end of this pattern
       end_ix = i + n_steps
       # check if we are beyond the sequence
       if end_ix > len(sequence)-1:
          break
       # gather input and output parts of the pattern
       seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
       X.append(seq_x)
       y.append(seq_y)
    return np.array(X), np.array(y)

# split into samples
time_step = inp_history_size
x_train2, y_train2 = split_sequence(train_data_initial2, time_step)
x_test2, y_test2 = split_sequence(test_data_initial2, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
x_train2 =x_train2.reshape(x_train2.shape[0],x_train2.shape[1],1)
x_test2 = x_test2.reshape(x_test2.shape[0],x_test2.shape[1],1)

# define model
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras import callbacks
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu',padding = 'same',input_shape=(inp_history_size,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss= 'mse' , metrics = [rmse()])

# Set up early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

# model training for 300 epochs
history2 = model.fit(x_train2, y_train2, epochs = 300 , validation_data = (x_test2,y_test2), batch_size=32, callbacks=[early_stopping], verbose=2)

# evaluate training data
train_loss2, train_rmse2 = model.evaluate(x_train2,y_train2, batch_size = 32)
print(f"train_loss={train_loss2:.3f}")
print(f"train_rmse={train_rmse2:.3f}")

# evaluate testing data
test_loss2, test_rmse2 = model.evaluate(x_test2,y_test2, batch_size = 32)
print(f"test_loss={test_loss2:.3f}")
print(f"test_rmse={test_rmse2:.3f}")

# save model to ONNX
output_path = data_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

output_path = file_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

# finish
mt5.shutdown()
#prediction using testing data

#prediction using testing data
test_predict2 = model.predict(x_test2)
print(test_predict2)
print("longitud total de la prediccion: ", len(test_predict2))
print("longitud total del sample: ", sample_size)

plot_y_test2 = np.array(y_test2).reshape(-1, 1)  # Selecciona solo el último elemento de cada muestra de prueba
plot_y_train2 = y_train2.reshape(-1,1)
train_predict2 = model.predict(x_train2)
#print(plot_y_test)

#calculate metrics
from sklearn import metrics
from sklearn.metrics import r2_score
#transform data to real values
value12=scaler2.inverse_transform(plot_y_test2)
#print(value1)
# Escala las predicciones inversas al transformarlas a la escala original
value22 = scaler2.inverse_transform(test_predict2.reshape(-1, 1))
#print(value2)
#calc score
score2 = np.sqrt(metrics.mean_squared_error(value12,value22))

print("RMSE         : {}".format(score2))
print("MSE          :", metrics.mean_squared_error(value12,value22))
print("R2 score     :",metrics.r2_score(value12,value22))

#sumarize model
model.summary()

#Print error
value112=pd.DataFrame(value12)
value222=pd.DataFrame(value22)
#print(value11)
#print(value22)

value1112=value112.iloc[:,:]
value2222=value222.iloc[:,:]

print("longitud salida (tandas de 1 min): ",len(value1112) )
#print("en horas son " + str((len(value1112))*60*24)+ " minutos")
print("en horas son " + str(((len(value1112)))/60)+ " horas")
print("en horas son " + str(((len(value1112)))/60/24)+ " dias")

# Calculate error
error2 = value1112 - value2222

import matplotlib.pyplot as plt
# Plot error
plt.figure(figsize=(10, 6))
plt.scatter(range(len(error2)), error2, color='blue', label='Error')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # Línea horizontal en y=0
plt.title('Error de Predicción ' + str(symbol))
plt.xlabel('Índice de la muestra')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig(str(symbol)+str(optional)+'.png')

rmse_2 = format(score2)
mse_2 = metrics.mean_squared_error(value12,value22)
r2_2 = metrics.r2_score(value12,value22)

resultados2= [rmse_2,mse_2,r2_2]

# Abre un archivo en modo escritura
with open(str(symbol)+str(optional)+"results.txt", "w") as archivo:
    # Escribe cada resultado en una línea separada
    for resultado in resultados2:
        archivo.write(str(resultado) + "\n")

# finish
mt5.shutdown()

#show iteration-rmse graph for training and validation
plt.figure(figsize = (18,10))
plt.plot(history2.history['root_mean_squared_error'],label='Training RMSE',color='b')
plt.plot(history2.history['val_root_mean_squared_error'],label='Validation-RMSE',color='g')
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE" + str(symbol))
plt.legend()
plt.savefig(str(symbol)+str(optional)+'1.png')

#show iteration-loss graph for training and validation
plt.figure(figsize = (18,10))
plt.plot(history2.history['loss'],label='Training Loss',color='b')
plt.plot(history2.history['val_loss'],label='Validation-loss',color='g')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("LOSS" + str(symbol))
plt.legend()
plt.savefig(str(symbol)+str(optional)+'2.png')

#show actual vs predicted (training) graph
plt.figure(figsize=(18,10))
plt.plot(scaler2.inverse_transform(plot_y_train2),color = 'b', label = 'Original')
plt.plot(scaler2.inverse_transform(train_predict2),color='red', label = 'Predicted')
plt.title("Prediction Graph Using Training Data" + str(symbol))
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.savefig(str(symbol)+str(optional)+'3.png')

#show actual vs predicted (testing) graph
plt.figure(figsize=(18,10))
plt.plot(scaler2.inverse_transform(plot_y_test2),color = 'b',  label = 'Original')
plt.plot(scaler2.inverse_transform(test_predict2),color='g', label = 'Predicted')
plt.title("Prediction Graph Using Testing Data" + str(symbol))
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.savefig(str(symbol)+str(optional)+'4.png')

##############################################################################################  JPYUSD

# python libraries
import MetaTrader5 as mt5
import tensorflow as tf
import numpy as np
import pandas as pd
import tf2onnx

# input parameters

inp_history_size = 120

sample_size = sample_size1
symbol = symbol3
optional = optional
inp_model_name = str(symbol)+"_"+str(optional)+".onnx"

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# we will save generated onnx-file near our script to use as resource
from sys import argv
data_path=argv[0]
last_index=data_path.rfind("\\")+1
data_path=data_path[0:last_index]
print("data path to save onnx model",data_path)

# and save to MQL5\Files folder to use as file
terminal_info=mt5.terminal_info()
file_path=terminal_info.data_path+"\\MQL5\\Files\\"
print("file path to save onnx model",file_path)

# set start and end dates for history data
from datetime import timedelta, datetime
#end_date = datetime.now()
#end_date = datetime(2024, 5, 1, 0)
start_date = end_date - timedelta(days=inp_history_size*20)

# print start and end dates
print("data start date =",start_date)
print("data end date =",end_date)

# get rates
eurusd_rates3 = mt5.copy_rates_from(symbol, timeframe ,  end_date, sample_size)
# create dataframe
df3=pd.DataFrame()
df3 = pd.DataFrame(eurusd_rates3)
print(df3)
# Extraer los precios de cierre directamente
datas3 = df3['close'].values

"""# Calcular la inversa de cada valor
inverted_data = 1 / datas

# Convertir los datos invertidos a un array de numpy si es necesario
data = inverted_data.values"""
data3 = datas3.reshape(-1,1)
# Imprimir los resultados
"""data = datas"""
# scale data
from sklearn.preprocessing import MinMaxScaler
scaler3=MinMaxScaler(feature_range=(0,1))
scaled_data3 = scaler3.fit_transform(data3)

# training size is 80% of the data
training_size3 = int(len(scaled_data3)*0.80)
print("Training_size:",training_size3)
train_data_initial3 = scaled_data3[0:training_size3,:]
test_data_initial3 = scaled_data3[training_size3:,:1]

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
       # find the end of this pattern
       end_ix = i + n_steps
       # check if we are beyond the sequence
       if end_ix > len(sequence)-1:
          break
       # gather input and output parts of the pattern
       seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
       X.append(seq_x)
       y.append(seq_y)
    return np.array(X), np.array(y)

# split into samples
time_step = inp_history_size
x_train3, y_train3 = split_sequence(train_data_initial3, time_step)
x_test3, y_test3 = split_sequence(test_data_initial3, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
x_train3 =x_train3.reshape(x_train3.shape[0],x_train3.shape[1],1)
x_test3 = x_test3.reshape(x_test3.shape[0],x_test3.shape[1],1)

# define model
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, MaxPooling1D, Dropout, Flatten, LSTM
from keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras import callbacks
model = Sequential()
model.add(Conv1D(filters=256, kernel_size=2, activation='relu',padding = 'same',input_shape=(inp_history_size,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, return_sequences = True))
model.add(Dropout(0.3))
model.add(LSTM(100, return_sequences = False))
model.add(Dropout(0.3))
model.add(Dense(units=1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss= 'mse' , metrics = [rmse()])

# Set up early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
)

# model training for 300 epochs
history3 = model.fit(x_train3, y_train3, epochs = 300 , validation_data = (x_test3,y_test3), batch_size=32, callbacks=[early_stopping], verbose=2)

# evaluate training data
train_loss3, train_rmse3 = model.evaluate(x_train3,y_train3, batch_size = 32)
print(f"train_loss={train_loss3:.3f}")
print(f"train_rmse={train_rmse3:.3f}")

# evaluate testing data
test_loss3, test_rmse3 = model.evaluate(x_test3,y_test3, batch_size = 32)
print(f"test_loss={test_loss3:.3f}")
print(f"test_rmse={test_rmse3:.3f}")

# save model to ONNX
output_path = data_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

output_path = file_path+inp_model_name
onnx_model = tf2onnx.convert.from_keras(model, output_path=output_path)
print(f"saved model to {output_path}")

# finish
mt5.shutdown()
#prediction using testing data

#prediction using testing data
test_predict3 = model.predict(x_test3)
print(test_predict3)
print("longitud total de la prediccion: ", len(test_predict3))
print("longitud total del sample: ", sample_size)

plot_y_test3 = np.array(y_test3).reshape(-1, 1)  # Selecciona solo el último elemento de cada muestra de prueba
plot_y_train3 = y_train3.reshape(-1,1)
train_predict3 = model.predict(x_train3)
#print(plot_y_test)

#calculate metrics
from sklearn import metrics
from sklearn.metrics import r2_score
#transform data to real values
value13=scaler3.inverse_transform(plot_y_test3)
#print(value1)
# Escala las predicciones inversas al transformarlas a la escala original
value23 = scaler3.inverse_transform(test_predict3.reshape(-1, 1))
#print(value2)
#calc score
score3 = np.sqrt(metrics.mean_squared_error(value13,value23))

print("RMSE         : {}".format(score3))
print("MSE          :", metrics.mean_squared_error(value13,value23))
print("R2 score     :",metrics.r2_score(value13,value23))

#sumarize model
model.summary()

#Print error
value113=pd.DataFrame(value13)
value223=pd.DataFrame(value23)
#print(value11)
#print(value22)

value1113=value113.iloc[:,:]
value2223=value223.iloc[:,:]

print("longitud salida (tandas de 1 hora): ",len(value1113) )
#print("en horas son " + str((len(value1113))*60*24)+ " minutos")
print("en horas son " + str(((len(value1113)))/60)+ " horas")
print("en horas son " + str(((len(value1113)))/60/24)+ " dias")

# Calculate error
error3 = value1113 - value2223

import matplotlib.pyplot as plt
# Plot error
plt.figure(figsize=(10, 6))
plt.scatter(range(len(error3)), error3, color='blue', label='Error')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)  # Línea horizontal en y=0
plt.title('Error de Predicción ' + str(symbol))
plt.xlabel('Índice de la muestra')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.savefig(str(symbol)+str(optional)+'.png')

rmse_3 = format(score3)
mse_3 = metrics.mean_squared_error(value13,value23)
r2_3 = metrics.r2_score(value13,value23)

resultados3= [rmse_3,mse_3,r2_3]

# Abre un archivo en modo escritura
with open(str(symbol)+str(optional)+"results.txt", "w") as archivo:
    # Escribe cada resultado en una línea separada
    for resultado in resultados3:
        archivo.write(str(resultado) + "\n")

# finish
mt5.shutdown()

#show iteration-rmse graph for training and validation
plt.figure(figsize = (18,10))
plt.plot(history3.history['root_mean_squared_error'],label='Training RMSE',color='b')
plt.plot(history3.history['val_root_mean_squared_error'],label='Validation-RMSE',color='g')
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.title("RMSE" + str(symbol))
plt.legend()
plt.savefig(str(symbol)+str(optional)+'1.png')

#show iteration-loss graph for training and validation
plt.figure(figsize = (18,10))
plt.plot(history3.history['loss'],label='Training Loss',color='b')
plt.plot(history3.history['val_loss'],label='Validation-loss',color='g')
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("LOSS" + str(symbol))
plt.legend()
plt.savefig(str(symbol)+str(optional)+'2.png')

#show actual vs predicted (training) graph
plt.figure(figsize=(18,10))
plt.plot(scaler3.inverse_transform(plot_y_train3),color = 'b', label = 'Original')
plt.plot(scaler3.inverse_transform(train_predict3),color='red', label = 'Predicted')
plt.title("Prediction Graph Using Training Data" + str(symbol))
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.savefig(str(symbol)+str(optional)+'3.png')

#show actual vs predicted (testing) graph
plt.figure(figsize=(18,10))
plt.plot(scaler3.inverse_transform(plot_y_test3),color = 'b',  label = 'Original')
plt.plot(scaler3.inverse_transform(test_predict3),color='g', label = 'Predicted')
plt.title("Prediction Graph Using Testing Data" + str(symbol))
plt.xlabel("Hours")
plt.ylabel("Price")
plt.legend()
plt.savefig(str(symbol)+str(optional)+'4.png')

################################################################################################
##############################################################################################
"""
import onnxruntime as ort
import numpy as np

# Cargar el modelo ONNX
sesion = ort.InferenceSession("EURUSD_M1_inverse_test.onnx")

# Obtener el nombre de la entrada y la salida del modelo
input_name = sesion.get_inputs()[0].name
output_name = sesion.get_outputs()[0].name

# Crear datos de entrada de prueba como un array de numpy
# Asegúrate de que los datos de entrada coincidan con la forma y el tipo esperado por el modelo
input_data = [1,120] #np.random.rand(1, 10).astype(np.float32)  # Ejemplo: entrada de tamaño [1, 10]

# Realizar la inferencia
result = sesion.run([output_name], {input_name: input_data})

# Imprimir el resultado
print(result)
"""
```

This .py will make three ONNX and also some graphs and data so you can check if everything is all right.

The data comes in a txt file, and each number stands for RMSE, MSE and R2 respectively.

Before running this script, you must specify the symbols, sample size, timeframe, and ending date (from to count backwards periods).

The optional variable is a string where you can add something like M1 Ticks or the end\_date ... whatever you whant to save the onnx files and the graphs and data.

```
symbol1 = "EURGBP"
symbol2 = "GBPUSD"
symbol3 = "EURUSD"
sample_size1 = 200000
optional = "_M1_test"
timeframe = mt5.TIMEFRAME_M1

#end_date = datetime.now()
end_date = datetime(2024, 3, 4, 0)
```

If you want to test in the strategy tester, modify the date as you want. If you want to trade, you just have to use this end\_date.

```
end_date = datetime.now()
```

\\*\\*\\* If you are trading in a zero-spread account, you can try using ticks instead of periods, you just have to change: \*\*\*

```
eurusd_rates = mt5.copy_rates_from(symbol, timeframe ,  end_date, sample_size)
```

with this one:

```
eurusd_rates = mt5.copy_ticks_from(symbol, end_date, sample_size, mt5.COPY_TICKS_ALL)
```

Here you will have Bid and Ask ticks. I think there is a limitation for the number of ticks. If you need more ticks, you can download all ticks from a symbol with this: [Download all data from a symbol](https://www.mql5.com/en/market/product/111572) that is free.

To run the .py file, just open it with VSC and hit RUN -> Run Without Debugging  (while having MT5 opened). Then wait for it to finish.

You will end up with a bunch of graphs, txts and ONNX files. You have to save the ONNX file in the MQL5/Files folder and specify the same path in the EA code.

It will still do that job thanks to this line:

```
# and save to MQL5\Files folder to use as file
terminal_info=mt5.terminal_info()
file_path=terminal_info.data_path+"\\MQL5\\Files\\"
print("file path to save onnx model",file_path)
```

Please note that if you want to have many ONNX files, which you have in other folders, you must specify the path.

This .py exports images like this ones:

![usjpy error](https://c.mql5.com/2/77/USDJPY_M1_test.png)

![usdjpy 1](https://c.mql5.com/2/77/EURJPY_M1_test1.png)

![usdjpy 2](https://c.mql5.com/2/77/EURJPY_M1_test2.png)

![usdjpy 3](https://c.mql5.com/2/77/EURJPY_M1_test3.png)

![usdjpy 4](https://c.mql5.com/2/77/EURJPY_M1_test4.png)

This graphs with the RMSE, MSE and R2 values

```
0.023019903957086384
0.0005299159781934813
0.999707563612641
```

With all this, we can know if our models are overfitted or underfitted.

In this case:

RMSE measures the standard deviation of the residuals (prediction errors). Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit.

A smaller RMSE value indicates better fit. The RMSE value you have is very small, suggesting that the model fits the dataset very well.

MSE is like RMSE but squares the errors before averaging them, which gives higher weight to larger errors. It's another measure of the quality of an estimator—it is always non-negative, and values closer to zero are better.

The very small MSE value further confirms that the model predictions are very close to the actual data points.

R2 is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. An 𝑅2 value of 1 indicates that the regression predictions perfectly fit the data.

Our R2 value is very close to 1, indicating that your model explains almost all the variability around the mean, which is excellent.

Overall, these metrics suggest that your model is performing exceptionally well in predicting or fitting to your dataset.

And to know if it overfitted, we use the graphs, for example in this case, the second one.

Here’s an analysis based on the graph:

1. **Training Loss (Blue Line):**

   - This line shows a steep decline initially, indicating that the model is quickly learning from the training dataset. As the iterations progress, the training loss continues to decrease, but at a slower rate, which is typical as the model starts to converge towards a minimum.
2. **Validation Loss (Green Line):**

   - The validation loss remains extremely low and fairly stable throughout the training process. This suggests that the model is generalizing well and not just memorizing the training data. The small fluctuations indicate variability in the validation set performance across iterations but remain within a very narrow band.

Overall, the graph suggests a very successful training process with excellent convergence and generalization. The low validation loss is particularly promising, as it indicates that the model should perform well on unseen data, assuming the validation set is representative of the general problem space.

Once all this is done, let's pass it to the EA.

### Expert Advisor for Triangular Arbitrage with predictions

```
//+------------------------------------------------------------------+
//|                          ONNX_Triangular EURUSD-USDJPY-EURJPY.mq5|
//|             Copyright 2024, Javier S. Gastón de Iriarte Cabrera. |
//|                      https://www.mql5.com/en/users/jsgaston/news |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2024, Javier S. Gastón de Iriarte Cabrera."
#property link        "https://www.mql5.com/en/users/jsgaston/news"
#property version     "1.04"

#property strict
#include <Trade\Trade.mqh>
#define MAGIC (965334)

#resource "/Files/art/arbitrage triangular/eurusdjpy/EURUSD__M1_test.onnx" as uchar ExtModel[]
#resource "/Files/art/arbitrage triangular/eurusdjpy/USDJPY__M1_test.onnx" as uchar ExtModel2[]
#resource "/Files/art/arbitrage triangular/eurusdjpy/EURJPY__M1_test.onnx" as uchar ExtModel3[]
CTrade ExtTrade;

#define SAMPLE_SIZE 120

input double lotSize = 3.0;
//input double slippage = 3;
// Add these inputs to allow dynamic control over SL and TP distances
input double StopLossPips = 50.0; // Stop Loss in pips
input double TakeProfitPips = 100.0; // Take Profit in pips
//input double maxSpreadPoints = 10.0;

input ENUM_TIMEFRAMES Periodo = PERIOD_CURRENT;

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
string symbol1 = Symbol();
input string symbol2 = "USDJPY";
input string symbol3 = "EURJPY";
int ticket1 = 0;
int ticket2 = 0;
int ticket3 = 0;
int ticket11 = 0;
int ticket22 = 0;
int ticket33 = 0;
input bool isArbitrageActive = true;

double spreads[1000]; // Array para almacenar hasta 1000 spreads
int spreadIndex = 0; // Índice para el próximo spread a almacenar

long     ExtHandle=INVALID_HANDLE;
//int      ExtPredictedClass=-1;
datetime ExtNextBar=0;
datetime ExtNextDay=0;
float    ExtMin=0.0;
float    ExtMax=0.0;

long     ExtHandle2=INVALID_HANDLE;
//int      ExtPredictedClass=-1;
datetime ExtNextBar2=0;
datetime ExtNextDay2=0;
float    ExtMin2=0.0;
float    ExtMax2=0.0;

long     ExtHandle3=INVALID_HANDLE;
//int      ExtPredictedClass=-1;
datetime ExtNextBar3=0;
datetime ExtNextDay3=0;
float    ExtMin3=0.0;
float    ExtMax3=0.0;

float predicted=0.0;
float predicted2=0.0;
float predicted3=0.0;
float predicted2i=0.0;
float predicted3i=0.0;

float   lastPredicted1=0.0;
float   lastPredicted2=0.0;
float   lastPredicted3=0.0;
float   lastPredicted2i=0.0;
float   lastPredicted3i=0.0;

int Order=0;

input double targetProfit = 100.0; // Eur benefit goal
input double maxLoss = -50.0;      // Eur max loss

input double perVar = 0.005; // Percentage of variation to make orders

ulong tickets[6]; // Array para almacenar los tickets de las órdenes

double sl=0.0;
double tp=0.0;

int Abrir = 0;
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {

   ExtTrade.SetExpertMagicNumber(MAGIC);
   Print("EA de arbitraje ONNX iniciado");

//--- create a model from static buffer
   ExtHandle=OnnxCreateFromBuffer(ExtModel,ONNX_DEFAULT);
   if(ExtHandle==INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the input tensor we must set them explicitly
//--- first index - batch size, second index - series size, third index - number of series (only Close)
   const long input_shape[] = {1,SAMPLE_SIZE,1};
   if(!OnnxSetInputShape(ExtHandle,ONNX_DEFAULT,input_shape))
     {
      Print("OnnxSetInputShape error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the output tensor we must set them explicitly
//--- first index - batch size, must match the batch size of the input tensor
//--- second index - number of predicted prices (we only predict Close)
   const long output_shape[] = {1,1};
   if(!OnnxSetOutputShape(ExtHandle,0,output_shape))
     {
      Print("OnnxSetOutputShape error ",GetLastError());
      return(INIT_FAILED);
     }

////////////////////////////////////////////////////////////////////////////////////////

//--- create a model from static buffer
   ExtHandle2=OnnxCreateFromBuffer(ExtModel2,ONNX_DEFAULT);
   if(ExtHandle2==INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the input tensor we must set them explicitly
//--- first index - batch size, second index - series size, third index - number of series (only Close)
   const long input_shape2[] = {1,SAMPLE_SIZE,1};
   if(!OnnxSetInputShape(ExtHandle2,ONNX_DEFAULT,input_shape2))
     {
      Print("OnnxSetInputShape error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the output tensor we must set them explicitly
//--- first index - batch size, must match the batch size of the input tensor
//--- second index - number of predicted prices (we only predict Close)
   const long output_shape2[] = {1,1};
   if(!OnnxSetOutputShape(ExtHandle2,0,output_shape2))
     {
      Print("OnnxSetOutputShape error ",GetLastError());
      return(INIT_FAILED);
     }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//--- create a model from static buffer
   ExtHandle3=OnnxCreateFromBuffer(ExtModel3,ONNX_DEFAULT);
   if(ExtHandle3==INVALID_HANDLE)
     {
      Print("OnnxCreateFromBuffer error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the input tensor we must set them explicitly
//--- first index - batch size, second index - series size, third index - number of series (only Close)
   const long input_shape3[] = {1,SAMPLE_SIZE,1};
   if(!OnnxSetInputShape(ExtHandle3,ONNX_DEFAULT,input_shape3))
     {
      Print("OnnxSetInputShape error ",GetLastError());
      return(INIT_FAILED);
     }

//--- since not all sizes defined in the output tensor we must set them explicitly
//--- first index - batch size, must match the batch size of the input tensor
//--- second index - number of predicted prices (we only predict Close)
   const long output_shape3[] = {1,1};
   if(!OnnxSetOutputShape(ExtHandle3,0,output_shape3))
     {
      Print("OnnxSetOutputShape error ",GetLastError());
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(ExtHandle!=INVALID_HANDLE)
     {
      OnnxRelease(ExtHandle);
      ExtHandle=INVALID_HANDLE;
     }

   if(ExtHandle2!=INVALID_HANDLE)
     {
      OnnxRelease(ExtHandle2);
      ExtHandle2=INVALID_HANDLE;
     }

   if(ExtHandle3!=INVALID_HANDLE)
     {
      OnnxRelease(ExtHandle3);
      ExtHandle3=INVALID_HANDLE;
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {

   SymbolProcessor processor; // Crear instancia de SymbolProcessor

   string A = processor.GetFirstThree(symbol1);
   string B = processor.GetLastThree(symbol1);

   string C = processor.GetFirstThree(symbol2);
   string D = processor.GetLastThree(symbol2);

   string E = processor.GetFirstThree(symbol3);
   string F = processor.GetLastThree(symbol3);

   if((A != E) || (B != C) || (D != F))
     {
      Print("Wrongly selected symbols");
      return;
     }

//--- check new day
   if(TimeCurrent()>=ExtNextDay)
     {
      GetMinMax();
      GetMinMax2();
      GetMinMax3();
      //--- set next day time
      ExtNextDay=TimeCurrent();
      ExtNextDay-=ExtNextDay%PeriodSeconds(Periodo);
      ExtNextDay+=PeriodSeconds(Periodo);

     }

//--- check new bar
   if(TimeCurrent()<ExtNextBar)
     {

      return;
     }
//--- set next bar time
   ExtNextBar=TimeCurrent();
   ExtNextBar-=ExtNextBar%PeriodSeconds();
   ExtNextBar+=PeriodSeconds();
//--- check min and max
   float close=(float)iClose(symbol1,Periodo,0);
   if(ExtMin>close)
      ExtMin=close;
   if(ExtMax<close)
      ExtMax=close;
   float close2=(float)iClose(symbol2,Periodo,0);
   if(ExtMin2>close2)
      ExtMin2=close2;
   if(ExtMax2<close2)
      ExtMax2=close2;
   float close3=(float)iClose(symbol3,Periodo,0);
   if(ExtMin3>close3)
      ExtMin3=close3;
   if(ExtMax3<close3)
      ExtMax3=close3;

   lastPredicted1=predicted;
   lastPredicted2=predicted2;
   lastPredicted3=predicted3;
   lastPredicted2i=predicted2i;
   lastPredicted3i=predicted3i;

//--- predict next price
   PredictPrice();
   PredictPrice2();
   PredictPrice3();

   /*   */
   double price1 = SymbolInfoDouble(symbol1, SYMBOL_BID);/////////////////
   double price2 = SymbolInfoDouble(symbol2, SYMBOL_BID);
   double price2i = 1/price2;
   double price3 = SymbolInfoDouble(symbol3, SYMBOL_ASK);
   double price3i = 1/price3;

   double price11 = SymbolInfoDouble(symbol1, SYMBOL_ASK);/////////////////
   double price22 = SymbolInfoDouble(symbol2, SYMBOL_ASK);
   double price22i = 1/price22;
   double price33 = SymbolInfoDouble(symbol3, SYMBOL_BID);
   double price33i = 1/price33;

   predicted2i = 1/predicted2;
   predicted3i = 1/predicted3;

//double lotSize = 1.0; // Lote base
   double lotSize2 = lotSize * predicted / predicted2;   /// tengo dudas con usar el invertido o no invertido
   double lotSize3 = lotSize * predicted / predicted3;

   double lotSize22 = lotSize * predicted / predicted2;  /// tengo dudas con usar el invertido o no invertido
   double lotSize33 = lotSize * predicted / predicted3;

// Redondear lotes a un múltiplo aceptable por tu bróker
   lotSize2 = NormalizeDouble(lotSize2, 2); // Asume 2 decimales para lotes
   lotSize3 = NormalizeDouble(lotSize3, 2);

   lotSize22 = NormalizeDouble(lotSize22, 2); // Asume 2 decimales para lotes
   lotSize33 = NormalizeDouble(lotSize33, 2);

   int totalPositions = PositionsTotal();

   if(Order==1 || Order==2)
     {
      // Verificar y cerrar órdenes si se cumplen las condiciones
      Print("Verificar y cerrar órdenes si se cumplen las condiciones");
      CheckAndCloseOrders();

     }

   if(!isArbitrageActive || ArePositionsOpen())
     {
      Print("Arbitraje inactivo o ya hay posiciones abiertas.");
      return;
     }
   double varia11 = 100.0 - (close*100/predicted);
   double varia21 = 100.0 - (close2*100/predicted2);
   double varia31 = 100.0 - (predicted3*100/close3);
   double varia12 = 100.0 - (predicted*100/close);
   double varia22 = 100.0 - (predicted2*100/close2);
   double varia32 = 100.0 - (close3*100/predicted3);

   if((varia11 > perVar)   && (varia21 > perVar) && (varia31 > perVar))
     {
      Print("se debería proceder a apertura de ordenes de derechas");
      Abrir = 1;
     }
   if((varia12 > perVar)   && (varia22 > perVar) && (varia32 > perVar))
     {
      Print("se debería proceder a apertura de ordenes de izquierdas");
      Abrir = 2;
     }
   if(Abrir == 1 && (predicted*predicted2*predicted3i>1))
     {
      Print("orden derecha");
      // Inicia el arbitraje si aún no está activo
      if(isArbitrageActive)
        {

         if((ticket1 == 0 && ticket2 == 0 && ticket3 ==0) && (Order==0) &&  totalPositions ==0)
           {
            Print("Preparando para abrir órdenes");
            Order = 1;

            MqlTradeRequest request;
            MqlTradeResult  result;

              {

               MqlRates rates[];
               ArraySetAsSeries(rates,true);
               int copied=CopyRates(symbol1,0,0,1,rates);

               CalculateSL(true,rates[0].close,symbol1);
               CalculateTP(true,rates[0].close,symbol1);

               if(ExtTrade.Buy(lotSize, symbol1, rates[0].close, sl, tp, "Arbitraje"))
                 {
                  tickets[0] = ExtTrade.ResultDeal();  // Getting the ticket of the last trade
                  Print("Order placed with ticket: ", tickets[0]);
                 }
               else
                 {
                  Print("Failed to place order: ", GetLastError());
                 }
              }

              {
               MqlRates rates[];
               ArraySetAsSeries(rates,true);
               int copied=CopyRates(symbol2,0,0,1,rates);

               CalculateSL(true,rates[0].close,symbol2);
               CalculateTP(true,rates[0].close,symbol2);

               if(ExtTrade.Buy(lotSize2, symbol2, rates[0].close, sl, tp, "Arbitraje"))
                 {
                  tickets[1] = ExtTrade.ResultDeal();  // Getting the ticket of the last trade
                  Print("Order placed with ticket: ", tickets[1]);
                 }
               else
                 {
                  Print("Failed to place order: ", GetLastError());
                 }
              }

              {
               MqlRates rates[];
               ArraySetAsSeries(rates,true);
               int copied=CopyRates(symbol3,0,0,1,rates);

               CalculateSL(false,rates[0].close,symbol3);
               CalculateTP(false,rates[0].close,symbol3);

               if(ExtTrade.Sell(lotSize3, symbol3, rates[0].close, sl, tp, "Arbitraje"))
                 {
                  tickets[2] = ExtTrade.ResultDeal();  // Getting the ticket of the last trade
                  Print("Order placed with ticket: ", tickets[2]);
                 }
               else
                 {
                  Print("Failed to place order: ", GetLastError());
                 }
              }

            ticket1=1;
            ticket2=1;
            ticket3=1;
            Abrir=0;

            return;
           }
         else
           {
            Print(" no se puede abrir ordenes");
           }
        }
     }

   if(Abrir == 2 && (predicted*predicted2*predicted3i<1))
     {
      Print("Orden Inversa");
      // Inicia el arbitraje si aún no está activo
      if(isArbitrageActive)
        {

         if((ticket11 == 0 && ticket22 == 0 && ticket33 ==0) && (Order==0) && totalPositions==0)
           {
            Print("Preparando para abrir órdenes");
            Order = 2;

            MqlTradeRequest request;
            MqlTradeResult  result;

              {
               MqlRates rates[];
               ArraySetAsSeries(rates,true);
               int copied=CopyRates(symbol1,0,0,1,rates);

               CalculateSL(false,rates[0].close,symbol1);
               CalculateTP(false,rates[0].close,symbol1);

               if(ExtTrade.Sell(lotSize, symbol1, rates[0].close, sl, tp, "Arbitraje"))
                 {
                  tickets[3] = ExtTrade.ResultDeal();  // Getting the ticket of the last trade
                  Print("Order placed with ticket: ", tickets[3]);
                 }
               else
                 {
                  Print("Failed to place order: ", GetLastError());
                 }
              }

              {
               MqlRates rates[];
               ArraySetAsSeries(rates,true);
               int copied=CopyRates(symbol2,0,0,1,rates);
               CalculateSL(false,rates[0].close,symbol2);
               CalculateTP(false,rates[0].close,symbol2);

               if(ExtTrade.Sell(lotSize2, symbol2, rates[0].close, sl, tp, "Arbitraje"))
                 {
                  tickets[4] = ExtTrade.ResultDeal();  // Getting the ticket of the last trade
                  Print("Order placed with ticket: ", tickets[4]);
                 }
               else
                 {
                  Print("Failed to place order: ", GetLastError());
                 }
              }

              {
               MqlRates rates[];
               ArraySetAsSeries(rates,true);
               int copied=CopyRates(symbol3,0,0,1,rates);

               CalculateSL(true,rates[0].close,symbol3);
               CalculateTP(true,rates[0].close,symbol3);

               if(ExtTrade.Buy(lotSize3, symbol3, rates[0].close, sl, tp, "Arbitraje"))
                 {
                  tickets[5] = ExtTrade.ResultDeal();  // Getting the ticket of the last trade
                  Print("Order placed with ticket: ", tickets[5]);
                 }
               else
                 {
                  Print("Failed to place order: ", GetLastError());
                 }
              }

            ticket11=1;
            ticket22=1;
            ticket33=1;
            Abrir=0;

            return;
           }
         else
           {
            Print(" no se puede abrir ordenes");
           }
        }
     }

  }

//+------------------------------------------------------------------+
//| Postions are open function                                       |
//+------------------------------------------------------------------+
bool ArePositionsOpen()
  {
// Check for positions on symbol1
   if(PositionSelect(symbol1) && PositionGetDouble(POSITION_VOLUME) > 0)
      return true;
// Check for positions on symbol2
   if(PositionSelect(symbol2) && PositionGetDouble(POSITION_VOLUME) > 0)
      return true;
// Check for positions on symbol3
   if(PositionSelect(symbol3) && PositionGetDouble(POSITION_VOLUME) > 0)
      return true;

   return false;
  }
//+------------------------------------------------------------------+
//| Price prediction function                                        |
//+------------------------------------------------------------------+
void PredictPrice(void)
  {
   static vectorf output_data(1);            // vector to get result
   static vectorf x_norm(SAMPLE_SIZE);       // vector for prices normalize

//--- check for normalization possibility
   if(ExtMin>=ExtMax)
     {
      Print("ExtMin>=ExtMax");
      //ExtPredictedClass=-1;
      return;
     }
//--- request last bars
   if(!x_norm.CopyRates(_Symbol,Periodo,COPY_RATES_CLOSE,1,SAMPLE_SIZE))
     {
      Print("CopyRates ",x_norm.Size());
      //ExtPredictedClass=-1;
      return;
     }
   float last_close=x_norm[SAMPLE_SIZE-1];
//--- normalize prices
   x_norm-=ExtMin;
   x_norm/=(ExtMax-ExtMin);
//--- run the inference
   if(!OnnxRun(ExtHandle,ONNX_NO_CONVERSION,x_norm,output_data))
     {
      Print("OnnxRun");
      //ExtPredictedClass=-1;
      return;
     }
//--- denormalize the price from the output value
   predicted=output_data[0]*(ExtMax-ExtMin)+ExtMin;
//return predicted;
  }

//+------------------------------------------------------------------+
//| Price prediction function                                        |
//+------------------------------------------------------------------+
void PredictPrice2(void)
  {
   static vectorf output_data2(1);            // vector to get result
   static vectorf x_norm2(SAMPLE_SIZE);       // vector for prices normalize

//--- check for normalization possibility
   if(ExtMin2>=ExtMax2)
     {
      Print("ExtMin2>=ExtMax2");
      //ExtPredictedClass=-1;
      return;
     }
//--- request last bars
   if(!x_norm2.CopyRates(symbol2,Periodo,COPY_RATES_CLOSE,1,SAMPLE_SIZE))
     {
      Print("CopyRates ",x_norm2.Size());
      //ExtPredictedClass=-1;
      return;
     }
   float last_close2=x_norm2[SAMPLE_SIZE-1];
//--- normalize prices
   x_norm2-=ExtMin2;
   x_norm2/=(ExtMax2-ExtMin2);
//--- run the inference
   if(!OnnxRun(ExtHandle2,ONNX_NO_CONVERSION,x_norm2,output_data2))
     {
      Print("OnnxRun");
      //ExtPredictedClass=-1;
      return;
     }
//--- denormalize the price from the output value
   predicted2=output_data2[0]*(ExtMax2-ExtMin2)+ExtMin2;
//--- classify predicted price movement
//return predicted2;
  }

//+------------------------------------------------------------------+
//| Price prediction function                                        |
//+------------------------------------------------------------------+
void PredictPrice3(void)
  {
   static vectorf output_data3(1);            // vector to get result
   static vectorf x_norm3(SAMPLE_SIZE);       // vector for prices normalize

//--- check for normalization possibility
   if(ExtMin3>=ExtMax3)
     {
      Print("ExtMin3>=ExtMax3");
      //ExtPredictedClass=-1;
      return;
     }
//--- request last bars
   if(!x_norm3.CopyRates(symbol3,Periodo,COPY_RATES_CLOSE,1,SAMPLE_SIZE))
     {
      Print("CopyRates ",x_norm3.Size());
      //ExtPredictedClass=-1;
      return;
     }
   float last_close3=x_norm3[SAMPLE_SIZE-1];
//--- normalize prices
   x_norm3-=ExtMin3;
   x_norm3/=(ExtMax3-ExtMin3);
//--- run the inference
   if(!OnnxRun(ExtHandle3,ONNX_NO_CONVERSION,x_norm3,output_data3))
     {
      Print("OnnxRun");
      //ExtPredictedClass=-1;
      return;
     }
//--- denormalize the price from the output value
   predicted3=output_data3[0]*(ExtMax3-ExtMin3)+ExtMin3;
//--- classify predicted price movement
//return predicted2;
  }

//+------------------------------------------------------------------+
//| Get minimal and maximal Close for last 120 values                |
//+------------------------------------------------------------------+
void GetMinMax(void)
  {
   vectorf closeMN;
   closeMN.CopyRates(symbol1,Periodo,COPY_RATES_CLOSE,0,SAMPLE_SIZE);
   ExtMin=closeMN.Min();
   ExtMax=closeMN.Max();
  }

//+------------------------------------------------------------------+
//| Get minimal and maximal Close for last 120 values                |
//+------------------------------------------------------------------+
void GetMinMax2(void)
  {
   vectorf closeMN2;
   closeMN2.CopyRates(symbol2,Periodo,COPY_RATES_CLOSE,0,SAMPLE_SIZE);
   ExtMin2=closeMN2.Min();
   ExtMax2=closeMN2.Max();
  }

//+------------------------------------------------------------------+
//| Get minimal and maximal Close for last 120 values                |
//+------------------------------------------------------------------+
void GetMinMax3(void)
  {
   vectorf closeMN3;
   closeMN3.CopyRates(symbol3,Periodo,COPY_RATES_CLOSE,0,SAMPLE_SIZE);
   ExtMin3=closeMN3.Min();
   ExtMax3=closeMN3.Max();
  }

//+------------------------------------------------------------------+
//| Symbols class returns both pairs of a symbol                     |
//+------------------------------------------------------------------+
class SymbolProcessor
  {
public:
   // Método para obtener los primeros tres caracteres de un símbolo dado
   string            GetFirstThree(string symbol)
     {
      return StringSubstr(symbol, 0, 3);
     }

   // Método para obtener los últimos tres caracteres de un símbolo dado
   string            GetLastThree(string symbol)
     {
      if(StringLen(symbol) >= 3)
         return StringSubstr(symbol, StringLen(symbol) - 3, 3);
      else
         return ""; // Retorna un string vacío si el símbolo es demasiado corto
     }
  };

//+------------------------------------------------------------------+
//| Calculate total profit from all open positions for the current symbol
//+------------------------------------------------------------------+
double CalculateCurrentArbitrageProfit()
  {
   double totalProfit = 0.0;
   int totalPositions = PositionsTotal(); // Get the total number of open positions

// Loop through all open positions
   for(int i = 0; i < totalPositions; i++)
     {
      // Get the ticket of the position at index i
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))    // Select the position by its ticket
        {
         // Add the profit of the current position to the total profit
         totalProfit += PositionGetDouble(POSITION_PROFIT);
         //Print("totalProfit ", totalProfit);
        }
     }

   return totalProfit; // Return the total profit of all open positions
  }
// Función para cerrar todas las órdenes
void CloseAllOrders()
  {
   string symbols[] = {symbol1, symbol2, symbol3};
   for(int i = 0; i < ArraySize(symbols); i++)
     {
      if(ExtTrade.PositionClose(symbols[i], 3))
         Print("Posición cerrada correctamente para ", symbols[i]);
      else
         Print("Error al cerrar posición para ", symbols[i], ": Error", GetLastError());
     }

// Resetea tickets y ordenes
   ticket1 = 0;
   ticket2 = 0;
   ticket3 = 0;
   ticket11 = 0;
   ticket22 = 0;
   ticket33 = 0;
   Order = 0;

   Print("Todas las órdenes están cerradas");
  }

//+------------------------------------------------------------------+
//| Check and close orders funcion                                   |
//+------------------------------------------------------------------+
// Función para verificar y cerrar órdenes
void CheckAndCloseOrders()
  {
   double currentProfit = CalculateCurrentArbitrageProfit();

// Condiciones para cerrar las órdenes
   if((currentProfit >= targetProfit || currentProfit <= maxLoss))
     {
      CloseAllOrders();  // Cierra todas las órdenes
      Print("Todas las órdenes cerradas. Beneficio/Pérdida actual: ", currentProfit);
     }
  }

//+------------------------------------------------------------------+
//| Get order volume function                                        |
//+------------------------------------------------------------------+
double GetOrderVolume(int ticket)
  {
   if(PositionSelectByTicket(ticket))
     {
      double volume = PositionGetDouble(POSITION_VOLUME);
      return volume;
     }
   else
     {
      Print("No se pudo seleccionar la posición con el ticket: ", ticket);
      return 0; // Retorna 0 si no se encuentra la posición
     }
  }
//+------------------------------------------------------------------+
// Function to get the price and calculate SL dynamically

double CalculateSL(bool isBuyOrder,double entryPrice,string simbolo)
  {

   double pointSize = SymbolInfoDouble(simbolo, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(simbolo, SYMBOL_DIGITS);
   double pipSize = pointSize * 10;

   if(isBuyOrder)
     {
      sl = NormalizeDouble(entryPrice - StopLossPips * pipSize, digits);
      tp = NormalizeDouble(entryPrice + TakeProfitPips * pipSize, digits);
     }
   else
     {
      sl = NormalizeDouble(entryPrice + StopLossPips * pipSize, digits);
      tp = NormalizeDouble(entryPrice - TakeProfitPips * pipSize, digits);
     }
   return sl;
  }
//+------------------------------------------------------------------+
// Function to get the price and calculate TP dynamically
double CalculateTP(bool isBuyOrder,double entryPrice, string simbolo)
  {

   double pointSize = SymbolInfoDouble(simbolo, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(simbolo, SYMBOL_DIGITS);
   double pipSize = pointSize * 10;

   if(isBuyOrder)
     {
      sl = NormalizeDouble(entryPrice - StopLossPips * pipSize, digits);
      tp = NormalizeDouble(entryPrice + TakeProfitPips * pipSize, digits);
     }
   else
     {
      sl = NormalizeDouble(entryPrice + StopLossPips * pipSize, digits);
      tp = NormalizeDouble(entryPrice - TakeProfitPips * pipSize, digits);
     }
   return tp;
  }
//+------------------------------------------------------------------+
// Function to handle errors and retry
bool TryOrderSend(MqlTradeRequest &request, MqlTradeResult &result)
  {
   for(int attempts = 0; attempts < 5; attempts++)
     {
      if(OrderSend(request, result))
        {
         return true;
        }
      else
        {
         Print("Failed to send order on attempt ", attempts + 1, ": Error ", GetLastError());
         Sleep(1000); // Pause before retrying to avoid 'context busy' errors
        }
     }
   return false;
  }
//+------------------------------------------------------------------+
```

### Explanation of the Expert Advisor

### The Strategy

We all now know what triangular arbitrage is, but, I've added to the code, a minimum amount difference between the prediction and the actual close value,   this difference is the percentage rate change between those two values, and you can modify that amount with this input:

```
input double perVar = 0.005; // Percentage of variation to make orders
```

\\*\\*\\* Notice in the code that is has this logic: \*\*\*

```
EUR       USD     | EUR  |^(-1)
----   x  ---  x  | ---- |
USD       JPY     | JPY  |
```

Everything is adapted to this logic, so if you use another pairs, everything must be modified.

I will attach another example (EURUSD - GBPUSD - EURGBP) so you can see the changes. It uses this logic:

```
EUR      | GBP |^(-1)    | EUR  |^(-1)
----   x | --- |      x  | ---- |
USD      | USD |         | GBP  |
```

The whole strategy is based on that when you multiply that logic, if its >1 you can multiply in the right direction, and if its <1 you can multiply on the left direction.

But the strategy of this Expert Advisor is that instead of using the actual prices we will use the predicted prices.

This logic means that if you multiply on the right, you multiply prices, and if you multiply by the inverse, it's a division. On the left, it's the other way around. You can observe these changes in the code.

The lot sizes must be selected according to the highest price, this is why, in this EUR-USD-JPY the minimum lot is around 2 or three lots.

The logic for the lot sizes is this:

```
   double lotSize2 = lotSize * predicted / predicted2;
   double lotSize3 = lotSize * predicted / predicted3;
```

where predicted is the predicted price for EURUSD, predicted2 is the price for USDJPY and predicted3 is the price for EURJPY.

The last part of this, is to normalize the lot size to the brokers requirements.

```
lotSize2 = NormalizeDouble(lotSize2, 2);
lotSize3 = NormalizeDouble(lotSize3, 2);
```

### Example

For this example, we will use EUR-USD-JPY pairs.

With this logic

```
EUR       USD     | EUR  |^(-1)
----   x  ---  x  | ---- |
USD       JPY     | JPY  |
```

We will train and test with 200000 minute period of time (sample size), till the 3rd of April. This will give us around 17 days of predictions.

So, the test in the strategy tester will run from the 3rd of April to the 21st of the same month and we will select the 1 Minute timeframe.

For the first test, we will use this inputs and settings (select carefully the symbols as they are added to the EA (OONX)):

![Input](https://c.mql5.com/2/78/alineadito_inputs.png)

![Settings](https://c.mql5.com/2/78/settings.png)

And these are the results:

![Graph](https://c.mql5.com/2/78/graph__2.png)

![Back-testing](https://c.mql5.com/2/78/backtesting__2.png)

This is what an EA resumes for the results of this back-testing:

This backtesting report provides a detailed analysis of a trading strategy's performance over a given period using historical data. The strategy started with an initial deposit of $100,000 and ended with a total net profit of $395.72, despite a significant gross loss of $1,279.06 compared to the gross profit of $1,674.78. The profit factor of 1.31 indicates that the gross profit exceeded the gross loss by 31%, showcasing the strategy's ability to generate profit over loss.

The strategy executed a total of 96 trades, with a roughly equal split between winning and losing trades, as indicated by the percentages of short trades won (38.30%) and long trades won (46.94%). The strategy had more losing trades overall (55 or 57.29%), emphasizing a need for improvement in trade selection or exit strategies.

The recovery factor of 0.84 suggests moderate risk, with the strategy recovering 84% of the maximum drawdown. Additionally, the maximum drawdown was relatively high at $358.78 (0.36% of the account), indicating that while the strategy was profitable, it also faced significant declines from which it had to recover.

The back-test also showed substantial drawdowns in terms of equity, with a maximal equity drawdown of 0.47% of the account. This, coupled with the sharp ratio of 21.21, suggests that the returns were considerably higher than the risk taken, which is positive. However, the low average consecutive wins (3 trades) versus higher average consecutive losses (2 trades) suggest that the strategy might benefit from refining its approach to maintain consistency in winning trades.

### Conclusion

In this article, we break down the exciting concept of triangular arbitrage using predictions, all through the user-friendly MT5 platform and Python programming. Imagine having a secret formula that lets you play a smart game of currency exchange, spinning dollars into euros, then yen, and back to dollars, aiming to end up with more than you started. This isn't magic; it's all about using special predictive models called ONNX and triangular arbitrage, which learn from past currency prices to predict future ones, guiding your trading moves.

The piece is super helpful for setting everything up, showing you how to install all the tools you'll need, like Python and Visual Studio Code, and how to get your computer ready to start testing. The article explains in simple terms, making sure you know how to adjust the strategy whether your trading account is basic or fancy.

Overall, this article is a fantastic resource for anyone looking to get into the forex trading game, using some of the smartest tech available. It guides you through the nitty-gritty of setting up and running your trading system, so you can try your hand at trading with an edge, thanks to the latest in artificial intelligence and machine learning. Whether you're a newbie to coding or trading, this guide has your back, showing you step by step how to make the digital leap into automated trading.

I hope you enjoid reading this article as I did making it.

**Attached files** \|


[Download ZIP](https://www.mql5.com/en/articles/download/14873.zip "Download all attachments in the single ZIP archive")

[ONNX.eurusdjpy.triangular\_test.120.Training\_\_mod.py](https://www.mql5.com/en/articles/download/14873/onnx.eurusdjpy.triangular_test.120.training__mod.py "Download ONNX.eurusdjpy.triangular_test.120.Training__mod.py")(26.86 KB)

[EURJPY\_\_M1\_test.onnx](https://www.mql5.com/en/articles/download/14873/eurjpy__m1_test.onnx "Download EURJPY__M1_test.onnx")(884.33 KB)

[EURUSD\_\_M1\_test.onnx](https://www.mql5.com/en/articles/download/14873/eurusd__m1_test.onnx "Download EURUSD__M1_test.onnx")(884.75 KB)

[USDJPY\_\_M1\_test.onnx](https://www.mql5.com/en/articles/download/14873/usdjpy__m1_test.onnx "Download USDJPY__M1_test.onnx")(884.83 KB)

[Triangular\_Arbitrage\_with\_ONNX\_predictions\_EURUSDGBP\_v6.mq5](https://www.mql5.com/en/articles/download/14873/triangular_arbitrage_with_onnx_predictions_eurusdgbp_v6.mq5 "Download Triangular_Arbitrage_with_ONNX_predictions_EURUSDGBP_v6.mq5")(55.3 KB)

[Triangular\_Arbitrage\_with\_ONNX\_predictions\_EURUSDJPY\_v6.mq5](https://www.mql5.com/en/articles/download/14873/triangular_arbitrage_with_onnx_predictions_eurusdjpy_v6.mq5 "Download Triangular_Arbitrage_with_ONNX_predictions_EURUSDJPY_v6.mq5")(57.53 KB)

**Warning:** All rights to these materials are reserved by MetaQuotes Ltd. Copying or reprinting of these materials in whole or in part is prohibited.

This article was written by a user of the site and reflects their personal views. MetaQuotes Ltd is not responsible for the accuracy of the information presented, nor for any consequences resulting from the use of the solutions, strategies or recommendations described.

#### Other articles by this author

- [Integrating Discord with MetaTrader 5: Building a Trading Bot with Real-Time Notifications](https://www.mql5.com/en/articles/16682)
- [Trading Insights Through Volume: Trend Confirmation](https://www.mql5.com/en/articles/16573)
- [Trading Insights Through Volume: Moving Beyond OHLC Charts](https://www.mql5.com/en/articles/16445)
- [From Python to MQL5: A Journey into Quantum-Inspired Trading Systems](https://www.mql5.com/en/articles/16300)
- [Example of new Indicator and Conditional LSTM](https://www.mql5.com/en/articles/15956)
- [Scalping Orderflow for MQL5](https://www.mql5.com/en/articles/15895)
- [Using PSAR, Heiken Ashi, and Deep Learning Together for Trading](https://www.mql5.com/en/articles/15868)

**Last comments \|**
**[Go to discussion](https://www.mql5.com/en/forum/467321)**
(14)


![Matteo Busacca](https://c.mql5.com/avatar/2023/4/642ec9d7-cf7b.png)

**[Matteo Busacca](https://www.mql5.com/en/users/matteobusacca)**
\|
14 Oct 2024 at 14:16

Hi, I'm trying your model, but I don't get a calculation. Why should I open [EURUSD](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: technical analysis") with 3 lots and then open the other two pairs with practically 0.02 lots, what's the point?


![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
19 Oct 2024 at 18:23

**Aleksey Vyazmikin [#](https://www.mql5.com/en/forum/467321#comment_54685439):**

In addition to the translation of the text from MQ, we need a minimal audit of the description of the actions described by the authors in their articles.

As I understood from the code, although I do not know this programming language sufficiently, the author normalises the whole dataset and then divides it into two subsamples.

Trains models that contain indirectly information about the future, and gives a praiseworthy evaluation of the result. I hope the author does not do this intentionally, otherwise it is already falsification, which will lead to the drain of readers' deposits.

I didn't understand from the article how the issue of price of one pip in the [deposit currency](https://www.mql5.com/en/docs/constants/environment_state/accountinformation#enum_account_info_string "MQL5 Documentation: Account Information") is solved. Why did they put VS?

Minmax what it does is normalize (means: instead of using values of price, it uses values from a to b) between 0 and 1.

I don't understand the other q, please explain with details (it's been a while since I made this article).

![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
19 Oct 2024 at 18:27

**Matteo Busacca [#](https://www.mql5.com/en/forum/467321#comment_54829766):**

Hi, I'm trying your model, but I don't get a calculation. Why should I open [EURUSD](https://www.mql5.com/en/quotes/currencies/eurusd "EURUSD chart: technical analysis") with 3 lots and then open the other two pairs with practically 0.02 lots, what's the point?

Hi,

I you could paste details of the q, I could solve them faster (made the article a long time ago).

From what I remember, you can't use the same lot size. The reason for that is that the pairs used have different price.

![Aleksey Vyazmikin](https://c.mql5.com/avatar/2024/6/6678986f-2caa.png)

**[Aleksey Vyazmikin](https://www.mql5.com/en/users/-aleks-)**
\|
19 Oct 2024 at 18:53

**Javier Santiago Gaston De Iriarte Cabrera [#](https://www.mql5.com/ru/forum/473771/page2#comment_54879597):**

Minmax does what it does - normalises (means: uses values from a to b instead of price values) between 0 and 1.

I don't understand the other q's, please explain more (it's been a long time since I made this article).

What is the need to install Visual Studio Code?

![Javier Santiago Gaston De Iriarte Cabrera](https://c.mql5.com/avatar/2024/11/672e255d-38a9.jpg)

**[Javier Santiago Gaston De Iriarte Cabrera](https://www.mql5.com/en/users/jsgaston)**
\|
24 Oct 2024 at 14:44

**Aleksey Vyazmikin [#](https://www.mql5.com/en/forum/467321/page2#comment_54879736):**

What is the need to install Visual Studio Code?

for first users ... its a good ide

![Learn how to trade the Fair Value Gap (FVG)/Imbalances step-by-step: A Smart Money concept approach](https://c.mql5.com/2/78/Learn_how_to_trade_the_Fair_Value_Gap____LOGO__1.png)[Learn how to trade the Fair Value Gap (FVG)/Imbalances step-by-step: A Smart Money concept approach](https://www.mql5.com/en/articles/14261)

A step-by-step guide to creating and implementing an automated trading algorithm in MQL5 based on the Fair Value Gap (FVG) trading strategy. A detailed tutorial on creating an expert advisor that can be useful for both beginners and experienced traders.

![Neural networks made easy (Part 69): Density-based support constraint for the behavioral policy (SPOT)](https://c.mql5.com/2/63/midjourney_image_13954_55_495__1-logo__1.png)[Neural networks made easy (Part 69): Density-based support constraint for the behavioral policy (SPOT)](https://www.mql5.com/en/articles/13954)

In offline learning, we use a fixed dataset, which limits the coverage of environmental diversity. During the learning process, our Agent can generate actions beyond this dataset. If there is no feedback from the environment, how can we be sure that the assessments of such actions are correct? Maintaining the Agent's policy within the training dataset becomes an important aspect to ensure the reliability of training. This is what we will talk about in this article.

![Population optimization algorithms: Binary Genetic Algorithm (BGA). Part II](https://c.mql5.com/2/65/Population_optimization_algorithms__Binary_Genetic_Algorithm_gBGAm___Part_2____LOGO.png)[Population optimization algorithms: Binary Genetic Algorithm (BGA). Part II](https://www.mql5.com/en/articles/14040)

In this article, we will look at the binary genetic algorithm (BGA), which models the natural processes that occur in the genetic material of living things in nature.

![Developing an MQL5 RL agent with RestAPI integration (Part 3): Creating automatic moves and test scripts in MQL5](https://c.mql5.com/2/61/RestAPI_Parte_3_-_Criando_jogadas_automuticas_e_Scripts_de_Teste_em_MQL5__LOGO.png)[Developing an MQL5 RL agent with RestAPI integration (Part 3): Creating automatic moves and test scripts in MQL5](https://www.mql5.com/en/articles/13813)

This article discusses the implementation of automatic moves in the tic-tac-toe game in Python, integrated with MQL5 functions and unit tests. The goal is to improve the interactivity of the game and ensure the reliability of the system through testing in MQL5. The presentation covers game logic development, integration, and hands-on testing, and concludes with the creation of a dynamic game environment and a robust integrated system.

[![](https://www.mql5.com/ff/sh/bhdtjfb1zry09943z2/267b575d2182c180804d340af38ce02c.jpg)\\
Trade from your iPhone or Android device\\
\\
You only need an internet connection to use the new powerful MetaTrader 5 Web terminal\\
\\
Learn more](https://www.mql5.com/ff/go?link=https://trade.metatrader5.com/&a=wtigumvtenarnsocpyfoqnanxrilnbxx&s=ec8c539e52b83881ff2d16eaff6913b25803952eb277cac55f670a102b2edc1f&uid=&ref=https://www.mql5.com/en/articles/14873&id=wdausxxqrpvhekbwjrjlhqjghyhesrqqau&fz_uniq=5082898769290203394)

This website uses cookies. Learn more about our [Cookies Policy](https://www.mql5.com/en/about/cookies).

![close](https://c.mql5.com/i/close.png)

![MQL5 - Language of trade strategies built-in the MetaTrader 5 client terminal](https://c.mql5.com/i/registerlandings/logo-2.png)

You are missing trading opportunities:

- Free trading apps
- Over 8,000 signals for copying
- Economic news for exploring financial markets

RegistrationLog in

latin characters without spaces

a password will be sent to this email

An error occurred


- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup&amp;reg=1)

You agree to [website policy](https://www.mql5.com/en/about/privacy) and [terms of use](https://www.mql5.com/en/about/terms)

If you do not have an account, please [register](https://www.mql5.com/en/auth_register)

Allow the use of cookies to log in to the MQL5.com website.

Please enable the necessary setting in your browser, otherwise you will not be able to log in.

[Forgot your login/password?](https://www.mql5.com/en/auth_forgotten?return=popup)

- [Log in With Google](https://www.mql5.com/en/auth_oauth2?provider=Google&amp;return=popup)