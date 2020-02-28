import os
import glob
import pandas as pd

custNo = ''
counter = 0
path = 'purchaseDataCustomers2019'
dir = os.path.join(os.getcwd(), path)
filename = os.path.join(dir, 'PurchasesMin4AdditionalFeatures2019 - NoInvDateDuplicates.csv')
currentFilename = ''
file = open(filename, 'r')
line = file.readline()
if line:
    header = line
    line = file.readline()
while line:
    list = line.split(',')
    if list[1] == custNo:
        currentFile.write(line)
    else:
        if os.path.exists(currentFilename):
            currentFile.close()
        custNo = list[1]
        name = "PurchasePeriodsCustNo" + custNo + ".csv"
        currentFilename = os.path.join(dir, name)
        counter = counter + 1
        print(str(counter) + ". Creating file " + name)
        currentFile = open(currentFilename, 'w')
        currentFile.write(header)
        currentFile.write(line)
    line = file.readline()

if os.path.exists(currentFilename):
    currentFile.close()
file.close()
