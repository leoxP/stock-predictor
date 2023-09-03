import numpy as np
import pandas as pd
import matplotlib.pyplot as plotter
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential

start='2010-01-01'
end=dt.datetime.now()

df=yf.download('AAPL',start,end)

df=df.reset_index() #adding indexing to rows

#dropping data and adj close columns, axis=1 (dropping columns)
df=df.drop(['Date','Adj Close'],axis=1) 

#moving average 100 days
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()

# plotter.figure(figsize=(12,6))
# plotter.plot(df.Close)
# plotter.plot(ma100,'red')
# plotter.plot(ma200,'green')
# plotter.show()

#Splitting Data into Training and Testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

#Scaling the data down to range bw 0 and 1
scaler=MinMaxScaler(feature_range=(0,1))
data_training_array=scaler.fit_transform(data_training)

x_train=[]
y_train=[]

for i in range(100,data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i]) #first 100 rows become columns
    y_train.append(data_training_array[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

#ML Model
model=Sequential()
model.add(LSTM(units=50,
               activation='relu',
               return_sequences=True,
               input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))


model.add(LSTM(units=60,
               activation='relu',
               return_sequences=True))
model.add(Dropout(0.3))


model.add(LSTM(units=80,
               activation='relu',
               return_sequences=True))
model.add(Dropout(0.4))


model.add(LSTM(units=120,activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1)) #Connects all 4 layers of LSTM model

# model.compile(optimizer='adam',loss='mean_squared_error')
# model.fit(x_train,y_train,epochs=50)
# model.save('keras_model.h5')

past_100_days=data_training.tail(100)
final_df=pd.concat([past_100_days,data_testing],ignore_index=True)
input_data=scaler.fit_transform(final_df)
            
x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    

x_test = np.array(x_test)
y_test = np.array(y_test)

# Making Predictions
from keras.models import load_model
model = load_model('keras_model.h5')

y_predicted=model.predict(x_test)
scale_val=scaler.scale_[0]
 
scale_factor=1/scale_val
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

# y_test = y_test.reshape(-1,1)

plotter.figure(figsize=(12,6))
plotter.plot(y_test,'b',label='Original Price')
plotter.plot(y_predicted,'r',label='Predicted Price')
plotter.xlabel('Time')
plotter.ylabel('Price')
plotter.legend()
plotter.show()