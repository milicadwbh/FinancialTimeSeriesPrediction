import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow messages
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)  # suppress tensorflow messages

# Import dependencies
import pandas as pd
import numpy as np
from numpy import array
from numpy import hstack
from scipy.ndimage.interpolation import shift
from IPython.display import display
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import tkinter as tk
from tkinter import ttk
from tkcalendar import Calendar, DateEntry
import datetime
import glob
import time
from keras import backend as K


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    x, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)


def predict(cal):
    date = cal.get_date()
    print(date)

    minPurchases = 20

    i = 0
    avgError = 0
    count = 0
    countPredictedPurchase = 0
    countPredictedNoPurchase = 0
    countRealizedPurchases = 0
    countNoRealizedPurchases = 0
    countPredictedPurchaseCorrect = 0
    countPredictedNoPurchaseCorrect = 0
    countLackTrainingData = 0
    countLackEvaluationData = 0

    path = 'purchaseDataCustomers'
    dir = os.path.join(os.getcwd(), path)
    resultsPath = 'predictionResults'
    resultsDir = os.path.join(os.getcwd(), resultsPath)
    timestamp = time.time()

    resultsFilename = os.path.join(resultsDir, "predictionResultsForDate" + date.strftime("%Y-%m-%d") + "-Min" + str(
        minPurchases) + "Purchases-" + str(timestamp) + "AdditionalFeatures.txt")
    resultsFile = open(resultsFilename, 'w')

    for filename in glob.glob(os.path.join(dir, '*.csv')):
        if "PurchasePeriodsCustNo" in filename:

            startIndex = filename.index("CustNo") + 6
            endIndex = filename.index(".csv")
            custNo = filename[startIndex:endIndex]

            i = i + 1
            line = "Iteration " + str(i) + " Customer " + custNo
            print(line)
            resultsFile.write(line + "\n")

            # Read data from csv file
            df = pd.read_csv(filename)

            # days from last purchase
            allX1 = df['DayDiff'].to_numpy()

            # days from two purchases ago
            allX2 = df['DayDiff2'].to_numpy()
            # days from three purchases ago
            allX3 = df['DayDiff3'].to_numpy()

            allOutput = shift(allX1,-1, cval=np.nan)

            # transform string date to date and then compare it to entered date
            df['InvDate'] = df['InvDate'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())

            df = df.loc[df['InvDate'] <= date]

            # Remove unnecessary columns
            # df.drop(['CustNo', 'ItemNo', 'InvDateCurr', 'InvDatePrior'], axis=1, inplace=True)
            # display(df[0:5])

            # Convert to numpy array
            trainX1 = df['DayDiff'].to_numpy()
            trainX2 = df['DayDiff2'].to_numpy()
            trainX3 = df['DayDiff3'].to_numpy()
            output = allOutput[0:len(trainX1)]

            # Convert to one dimensional array
            trainX1 = trainX1.reshape(len(trainX1), 1)
            trainX2 = trainX2.reshape(len(trainX2), 1)
            trainX3 = trainX3.reshape(len(trainX3), 1)
            output = output.reshape(len(output), 1)

            dataset = hstack((trainX1, trainX2, output))

            # number of time steps
            n_steps = 3

            if (len(trainX1) > n_steps) & (len(trainX1) >= minPurchases):

                # Split into samples
                x, y = split_sequences(dataset, n_steps)
                #print("Split sequence:")
                #for i in range(len(x)):
                    #print(x[i], y[i])

                # reshape from [samples, timesteps] into [samples, timesteps, features]
                n_features = x.shape[2]
                x = x.reshape((x.shape[0], x.shape[1], n_features))

                # define model
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                # fit model
                model.fit(x, y, epochs=200, verbose=0)
                # demonstrate prediction
                x_input = hstack((trainX1[len(trainX1) - 3:], trainX2[len(trainX2) - 3:]))
                line = "x = [" + str(x_input[0, 0]) + ", " + str(x_input[0, 1]) + "\n" \
                       + str(x_input[1, 0]) + ", " + str(x_input[1, 1]) + "\n" \
                       + str(x_input[2, 0]) + ", " + str(x_input[2, 1]) + "]"
                print(line)
                resultsFile.write(line + "\n")
                x_input = x_input.reshape((1, n_steps, n_features))
                yhat = model.predict(x_input, verbose=0)
                line = "y = " + str("{:.2f}".format(yhat[0][0]))
                print(line)
                resultsFile.write(line + "\n")

                if yhat <= 7:
                    line = "Purchase in the next 7 days is expected."
                    print(line)
                    resultsFile.write(line + "\n")
                else:
                    line = "Purchase in the next 7 days is NOT expected."
                    print(line)
                    resultsFile.write(line + "\n")

                K.clear_session()

                if len(allX1) > len(trainX1):
                    actual = allX1[len(trainX1)]
                    line = "actualY = " + str(actual)
                    print(line)
                    resultsFile.write(line + "\n")
                    if yhat <= 7 and actual <= 7:
                        line = "Prediction is correct."
                        print(line)
                        resultsFile.write(line + "\n")
                        countPredictedPurchaseCorrect = countPredictedPurchaseCorrect + 1
                        countRealizedPurchases = countRealizedPurchases + 1
                        countPredictedPurchase = countPredictedPurchase + 1
                    elif yhat > 7 and actual > 7:
                        line = "Prediction is correct."
                        print(line)
                        resultsFile.write(line + "\n")
                        countPredictedNoPurchaseCorrect = countPredictedNoPurchaseCorrect + 1
                        countNoRealizedPurchases = countNoRealizedPurchases + 1
                        countPredictedNoPurchase = countPredictedNoPurchase + 1
                    else:
                        line = "Prediction is NOT correct."
                        print(line)
                        resultsFile.write(line + "\n")
                        if (actual <= 7):
                            countRealizedPurchases = countRealizedPurchases + 1
                            countPredictedNoPurchase = countPredictedNoPurchase + 1
                        else:
                            countNoRealizedPurchases = countNoRealizedPurchases + 1
                            countPredictedPurchase = countPredictedPurchase + 1
                else:
                    line = "Prediction evaluation not possible due to lack of data."
                    print(line)
                    resultsFile.write(line + "\n")
                    countLackEvaluationData = countLackEvaluationData + 1
            else:
                line = "Not enough data for training"
                print(line)
                resultsFile.write(line + "\n")
                countLackTrainingData = countLackTrainingData + 1

    line = "Purchases prediction precision: " + str(countPredictedPurchaseCorrect) + " out of " + str(
        countPredictedPurchase) \
           + " purchase predictions = " + str(
        "{:.2f}".format(countPredictedPurchaseCorrect / countPredictedPurchase * 100)) + "%"
    print(line)
    resultsFile.write(line + "\n")
    line = "Purchases prediction coverage: " + str(countPredictedPurchaseCorrect) + " out of " + str(
        countRealizedPurchases) \
           + " = " + str("{:.2f}".format(countPredictedPurchaseCorrect / countRealizedPurchases * 100)) + "%"
    print(line)
    resultsFile.write(line + "\n")

    line = "Predicted purchases: " + str(countPredictedPurchase) + "\nPredicted no purchase: " + str(
        countPredictedNoPurchase) \
           + "\nRealized purchases: " + str(countRealizedPurchases) + "\nNo realized purchases: " + str(
        countNoRealizedPurchases) + \
           "\nCorrectly predicted purchases: " + str(
        countPredictedPurchaseCorrect) + "\nCorrectly predicted no purchase: " + str(countPredictedNoPurchaseCorrect) \
           + "\nIterations with no evaluation data: " + str(
        countLackEvaluationData) + "\nIterations with no training data: " + str(countLackTrainingData)
    print(line)
    resultsFile.write(line + "\n")
    resultsFile.close()


master = tk.Tk()
ttk.Label(master, text='Choose date').pack(padx=10, pady=10)
cal = DateEntry(master, width=12, background='grey', foreground='white', borderwidth=2, year=2017,
                showweeknumbers=False, date_pattern='dd.mm.yyyy')
cal.pack(padx=10, pady=10)
ttk.Button(master, text='Predict', command=lambda: predict(cal)).pack(padx=10, pady=10)

master.mainloop()
