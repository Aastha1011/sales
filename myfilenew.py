import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def predict_data(filen, mnth, yrs):
    months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
    years = [2013, 2014, 2015, 2016]
    month_arr = ["Jan", "Feb", "March", "April", "May", "June", "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
    a = [[0 for x in range(4)] for y in range(12)]  # Five rows 12 for years, columns for months
    avg = list(range(4))  # for average of total (jan to dec)12 months in 5 years average
    total_annual_s = list(range(4))
    total_annual_sale = 0.0
    ys = []
    frcst = []
    temp = 0.0
    s = 0
    # add daily sale to find monthly sale then store into array
    for m in range(0, 12):

        for i in range(0, 4):

            csv_file = csv.reader(open(filen, "r"), delimiter=",")

            sum = 0.0
            # loop through csv list
            for row in csv_file:
                arr = row[1].split("-")

                if arr[0] == str(years[i]) and arr[1] == months[m]:
                    sum = sum + float(row[4])
            a[m][i] = sum

    # Linear Regression to find predict sale of entered Month in entered year
    if mnth in str(month_arr):
        indx = month_arr.index(mnth)
        for val in range(0, 4):
            s = a[indx][val]
            ys.append(s)
            if len(ys) == len(years):
                x = np.array(years)
                y = np.array(ys)
                m = (((np.mean(x) * np.mean(y)) - np.mean(x * y)) / ((np.mean(x) * np.mean(x)) - np.mean(x * x)))
                m = round(m, 2)
                b = (np.mean(y) - np.mean(x) * m)
                b = round(b, 2)
                reg_line = [(m * x) + b for x in years]
                predict = (m * yrs) + b
    print("Actual sales of " + str(mnth) + " " + str(years))
    print(ys)
    print("Predicted value of " + mnth + " in " + str(yrs) + " using Simple Linear Regrission =" + str(predict))

    if len(ys) == len(years):
        x = np.array(years)
        y = np.array(ys)
        m = (((np.mean(x) * np.mean(y)) - np.mean(x * y)) / ((np.mean(x) * np.mean(x)) - np.mean(x * x)))
        m = round(m, 2)
        b = (np.mean(y) - np.mean(x) * m)
        b = round(b, 2)
        reg_line = [round((m * x) + b) for x in years]
        print("predicted values using Linear Regression")
        print(reg_line)


    # finding Coefficients,finding Mean squared error and root Mean squared error
    reg = linear_model.LinearRegression()
    y = np.array(ys).reshape((len(ys), 1))
    x = np.array(reg_line).reshape((len(reg_line), 1))
    reg.fit(x, y)
    print('Coefficients: \n', reg.coef_)

    xTest = np.array(reg_line).reshape((len(reg_line), 1))
    ytest = np.array(ys).reshape((len(ys), 1))

    preds = reg.predict(xTest)
    print("R2 score : %.2f" % r2_score(ytest, preds))
    MSE = mean_squared_error(ytest, preds) / 100
    print("Mean squared error: %.2f" % MSE)
    rmse = math.sqrt(MSE)
    print('RMSE:% 3f' % rmse)
    er = []
    g = 0
    for i in range(len(ytest)):
        print("actual=", ytest[i], " observed=", preds[i])
        x = (ytest[i] - preds[i]) ** 2
        er.append(x)
        g = g + int(x)
    m = np.mean(ytest)
    print("average of observed values", m)
    x = 0
    for i in range(len(er)):
        x = x + er[i]

    print("MSE", x / len(er))

    v = np.var(er)
    print("variance", v)

    print("average of errors ", np.mean(er))

    m = np.mean(ytest)
    print("average of observed values", m)

    y = 0
    for i in range(len(ytest)):
        y = y + ((ytest[i] - m) ** 2)

    print("total sum of squares", y)
    print("ẗotal sum of residuals ", g)
    print("r2 calculated", 1 - (g / y))
    plt.scatter(years, ys)
    plt.plot(years, reg_line)
    plt.xlabel("Independent Varialble")
    plt.ylabel("Dependent Variable")
    plt.title(" Sale Graph of " + mnth)
    plt.show()













