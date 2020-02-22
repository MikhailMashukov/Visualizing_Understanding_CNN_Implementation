# Should be run with C:\Program Files\Python35

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re

def getLogAsTable(srcFileName):
    table = []
    fieldNames = ['trainLoss', 'trainAcc', 'valLoss', 'valAcc']
    with open(srcFileName, 'r') as file:
        for line in file:
            # Parsing "- 9s - loss: 9.9986e-04 - accuracy: 0.0000e+00 - val_loss: 9.9930e-04 - val_accuracy: 0.0000e+00"
            match = re.match(r'\s*- .+?s - loss\: (\d.*?) - accuracy\: (\d.*?)'
                             ' - val_loss: (\d.*?) - val_accuracy\: (\d.*)', line)
            if match:
                # trainLossStr, trainAccStr = match.groups()
                # row = [float(trainLossStr), float(trainAccStr)]
                row = [float(valStr) for valStr in match.groups()]
                if len(row) != len(fieldNames):
                    raise Exception('Value count mismatch (%s)' % line)
                table.append(row)

    return pd.DataFrame(table, columns=fieldNames)

def analyzeColumn(logTable, fieldName):
    xs = np.arange(1, logTable.shape[0] + 1)
    ys = logTable[fieldName]
    plt.plot(xs, ys, label=fieldName, alpha=0.7)

    polyCoeffs = np.polyfit(xs, ys, 3)
    print(polyCoeffs)
    projXs = np.arange(1, logTable.shape[0] * 4)
    polyYs = np.polyval(polyCoeffs, projXs)
    plt.plot(projXs, polyYs, label=fieldName + 'Proj',
             alpha=0.7)

    polyCoeffs = np.polyfit(xs, ys, 4)
    print(polyCoeffs)
    projXs = np.arange(1, logTable.shape[0] * 4)
    polyYs = np.polyval(polyCoeffs, projXs)
    plt.plot(projXs, polyYs, label=fieldName + 'Proj',
             linestyle=':', alpha=0.7)


if __name__ == '__main__':
    logTable = getLogAsTable('log')
    # print(logTable)

    analyzeColumn(logTable, 'trainLoss')
    analyzeColumn(logTable, 'valLoss')
    plt.show()

    pass
