# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import read_traffic_flow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# parameters settings
files_dir = "./traffic_flow_data/traffic data2017.9.28/Traffic flow/"
with_weekends = True
epoch_num = 2

data_train, data_test = \
    read_traffic_flow.read_traffic_flow(files_dir, with_weekends)

data_train_ori = data_train.reshape(data_train.shape[0], -1).astype('float32')
data_test = data_test.reshape(data_test.shape[0], -1).astype('float32')
# plt.figure(1)
# plt.plot(data_train)
# plt.figure(2)
# plt.plot(data_validate)
# plt.figure(3)
# plt.plot(data_test)
# plt.show()


def create_dataset(dataset, look_back=1):
    data_x, data_y = [], []
    for i in range(len(dataset)-look_back-1):
        data_x.append(dataset[i:(i+look_back)])
        data_y.append(dataset[i + look_back])
    return np.array(data_x), np.array(data_y)


# fix random seed for reproducibility
np.random.seed(7)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
data_train = scaler.fit_transform(data_train_ori)
data_test = scaler.fit_transform(data_test)

look_back = 1
train_x, train_y = create_dataset(data_train, look_back)
test_x, test_y = create_dataset(data_test, look_back)

train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1])
test_x = test_x.reshape(test_x.shape[0], 1, test_x.shape[1])

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=epoch_num, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(train_x)
testPredict = model.predict(test_x)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
train_y = scaler.inverse_transform([train_y.ravel()])
testPredict = scaler.inverse_transform(testPredict)
test_y = scaler.inverse_transform([test_y.ravel()])
train_y = train_y.T
test_y = test_y.T
# print(trainPredict.shape, train_y.shape, testPredict.shape, test_y.shape)
# print(train_y.shape, trainPredict.shape)
# print("PAUSE"); input()
trainScore = math.sqrt(mean_absolute_error(train_y, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_absolute_error(test_y, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

l = data_train.shape[0] + data_test.shape[0]
# # shift train predictions for plotting
trainPredictPlot = np.empty((l, 1))
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# # shift test predictions for plotting
testPredictPlot = np.empty((l, 1))
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+look_back*3:l-1, :] = testPredict

test_y = test_y + 1

loss_test = np.sum(np.abs(testPredict - test_y) / test_y) / test_y.shape[0]

# plot baseline and predictions
plt.plot(scaler.inverse_transform(
    np.vstack((data_train, data_test))), 'yellow')
plt.plot(trainPredictPlot, 'red')
plt.plot(testPredictPlot, 'blue')
plt.title("yellow: total; red: trainPred;" +
          " blue: testPred\n" +
          "If take weekends into account = {}\n".format(with_weekends) +
          "After {} epochs".format(epoch_num))
# print("shape of total:", scaler.inverse_transform(
#     np.vstack((data_train, data_test))).shape)
# print("trainPredict.shape:", trainPredict.shape)
# print("testPredict.shape:", testPredict.shape)
print("Loss of data_test = {}".format(loss_test))
loss_lst = np.abs(testPredict - test_y).ravel()
# print("loss_lst:", loss_lst)
# for i in range(len(loss_lst)):
#     if loss_lst[i] > 100:
#         print('{} = |{} - {}|'.format(loss_lst[i],
#               testPredict.ravel().tolist()[i], test_y.ravel().tolist()[i]))
with open('./loss.txt', 'w') as fout:
    fout.writelines('\n'.join(loss_lst.astype(str).tolist()))

plt.show()
