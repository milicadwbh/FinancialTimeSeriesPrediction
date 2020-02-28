import os
import glob
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, StringVar

master = tk.Tk()
l1 = tk.Label(master, text="Please select file for processing", justify=tk.LEFT).grid(row=0, column=1, padx=5, pady=10)
l2 = ttk.Label(master, text='File path').grid(row=1, column=0, padx=5, pady=5)
path = StringVar()
e1 = tk.Entry(master, textvariable=path, width=50)
e1.grid(row=1, column=1, padx=5, pady=5)


def browseCallback():
    filename = askopenfilename()
    path.set(filename)


b1 = tk.Button(master, text="Browse", width=10, command=browseCallback)
b1.grid(row=1, column=2, padx=5, pady=5)


def processCallback():
    filepath = path.get()
    if filepath != "":
        if os.path.exists(filepath):
            print(filepath)
            file = open(filepath, 'r')
            line = file.readline()
            sumMAPE = 0
            sumMAE = 0
            sumMSE = 0
            count = 0
            while line:
                if line.startswith("y ="):
                    y = float(line[(line.index("=") + 2):])
                if line.startswith("actualY = "):
                    actualy = float(line[(line.index("=") + 2):])
                    #print("y = " + str(y))
                    #print("actualY = " + str(actualy))
                    #print("diffY = " + str((actualy - y)/actualy))
                    count = count + 1
                    sumMAPE = sumMAPE + abs((actualy-y)/actualy)
                    sumMAE = sumMAE + abs(actualy-y)
                    sumMSE = sumMSE + (actualy-y) ** 2
                line = file.readline()
            file.close()
            mape = sumMAPE/count * 100
            mae = sumMAE/count
            mse = sumMSE/count
            rmse = mse ** 0.5
            print("MAPE = " + str(mape) + "%")
            print("MAE = " + str(mae))
            print("MSE = " + str(mse))
            print("RMSE = " + str(rmse))

b2 = tk.Button(master, text="Process", width=10, command=processCallback)
b2.grid(row=2, column=1, padx=5, pady=10)

master.mainloop()
