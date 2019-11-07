import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #suppress tensorflow messages
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) #suppress tensorflow messages

# Import dependencies
import pandas as pd
import numpy as np
from numpy import array
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
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		x.append(seq_x)
		y.append(seq_y)
	return array(x), array(y)


def predict(cal):
	date = cal.get_date()
	print(date)

	minPurchases = 100

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

	resultsFilename = os.path.join(resultsDir, "predictionResultsForDate" + date.strftime("%Y-%m-%d") + "-Min" + str(minPurchases) + "Purchases-" + str(timestamp) + ".txt")
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
			# display(df[0:5])

			allX = df['DayDiff'].to_numpy()
			#print(allX)

			# transform string date to date and then compare it to entered date
			df['InvDate'] = df['InvDate'].map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").date())
			#display(df[0:5])

			df = df.loc[df['InvDate'] <= date]
			#display(df[:])

			# Remove unnecessary columns
			#df.drop(['CustNo', 'ItemNo', 'InvDateCurr', 'InvDatePrior'], axis=1, inplace=True)
			# display(df[0:5])

			# Convert to numpy array
			trainX = df['DayDiff'].to_numpy()
			# print(x)

			# Convert to one dimensional array
			trainX = np.reshape(trainX, (len(trainX)))
			#print(trainX)

			# number of time steps
			n_steps = 3

			if (len(trainX) > n_steps) & (len(trainX) >= minPurchases):

				# Split into samples
				x, y = split_sequence(trainX, n_steps)
				# print("Split sequence:")
				# print("x = ", x)
				# print("y = ", y)

				# reshape from [samples, timesteps] into [samples, timesteps, features]
				n_features = 1
				x = x.reshape((x.shape[0], x.shape[1], n_features))

				# define model
				model = Sequential()
				model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
				model.add(Dense(1))
				model.compile(optimizer='adam', loss='mse')
				# fit model
				model.fit(x, y, epochs=200, verbose=0)
				# demonstrate prediction
				x_input = trainX[len(trainX) - 3:]
				line = "x = [" + str(x_input[0]) + ", " + str(x_input[1]) + ", " + str(x_input[2]) + "]"
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

				if len(allX) > len(trainX):
					actual = allX[len(trainX)]
					line = "actualY = " + str(actual)
					print(line)
					resultsFile.write(line + "\n")
					'''
					error = abs(yhat- actual)/actual * 100
					avgError = avgError + error
					count = count + 1
					line = "Prediction error is " + str(error) + " percent"
					print(line)
					resultsFile.write(line + "\n")
					'''
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
	'''
	avgError = avgError / count
	line = "Average error is " + str(avgError)
	'''

	line = "Purchases prediction precision: " + str(countPredictedPurchaseCorrect) + " out of " + str(countPredictedPurchase) \
		   + " purchase predictions = " + str("{:.2f}".format(countPredictedPurchaseCorrect/countPredictedPurchase * 100)) + "%"
	print(line)
	resultsFile.write(line + "\n")
	line = "Purchases prediction coverage: " + str(countPredictedPurchaseCorrect) + " out of " + str(countRealizedPurchases) \
		   + " = " + str("{:.2f}".format(countPredictedPurchaseCorrect / countRealizedPurchases * 100)) + "%"
	print(line)
	resultsFile.write(line + "\n")

	line = "Predicted purchases: " + str(countPredictedPurchase) + "\nPredicted no purchase: " + str(countPredictedNoPurchase) \
		   + "\nRealized purchases: " + str(countRealizedPurchases) + "\nNo realized purchases: " + str(countNoRealizedPurchases) + \
			"\nCorrectly predicted purchases: " + str(countPredictedPurchaseCorrect) + "\nCorrectly predicted no purchase: " + str(countPredictedNoPurchaseCorrect) \
		   + "\nIterations with no evaluation data: " + str(countLackEvaluationData) + "\nIterations with no training data: " + str(countLackTrainingData)
	print(line)
	resultsFile.write(line + "\n")
	resultsFile.close()


master = tk.Tk()
ttk.Label(master, text='Choose date').pack(padx=10, pady=10)
cal = DateEntry(master, width=12, background='grey', foreground='white', borderwidth=2, year=2017, showweeknumbers=False, date_pattern='dd.mm.yyyy')
cal.pack(padx=10, pady=10)
ttk.Button(master, text='Predict', command=lambda: predict(cal)).pack(padx=10, pady=10)

master.mainloop()

'''
master = tk.Tk()
tk.Label(master, text="First Name").grid(row=0)
tk.Label(master, text="Last Name").grid(row=1)

e1 = tk.Entry(master)
e2 = tk.Entry(master)

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)

master.mainloop()
'''



