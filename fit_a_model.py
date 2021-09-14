import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

'''get data from file'''
def get_data():
    data = pd.read_excel('dane.xlsx', sheet_name='data', engine='openpyxl')
    global x_raw, y_raw, x_to_predict
    x_raw = np.array(data['x']).reshape(((-1,1)))
    y_raw = np.array(data['y'])
    x_to_predict = np.linspace(x_raw[0], x_raw[-1], 100)



'''plot data'''
def plot_data():
    plt.scatter(x_raw,y_raw)
    plt.show()

'''plot data and model'''
def plot_data_and_model():
    plt.scatter(x_raw,y_raw)
    # prediction based on model
    plt.plot(x_to_predict, y_predicted, color='orange')
    plt.show()


'''linear model'''
def linear_model():
    x = x_raw
    y = y_raw
    global model
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('R^2:', r_sq)
    print('a_0:', model.intercept_)
    print('a_1:', model.coef_)

    global y_predicted
    y_predicted = model.predict(x_to_predict)


''' model wykładniczy '''
def exp_model():
    v = np.log(y_raw)
    global model
    model = LinearRegression().fit(x_raw, v)
    r_sq = model.score(x_raw, v)
    print('coefficient of determination:', r_sq)

    a_0 = np.exp(model.intercept_)
    print('a_0:', a_0) #a_0

    print('a_1:', model.coef_[0]) #a_1
    print('Oszacowany model Y=', a_0, 'exp(', model.coef_[0], 't', ')')

    #prediction based on model
    global y_predicted
    y_predicted = np.exp(model.predict(x_to_predict))

'''hiperbolic model 1/x'''
def hiperbolic_model():
    x = 1/x_raw
    y = y_raw
    global model
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('R^2:', r_sq)
    print('a_0:', model.intercept_)
    print('a_1:', model.coef_)

    global y_predicted
    y_predicted = model.predict(1/x_to_predict)

'''hiperbolic model 1/y'''
def hiperbolic_model2():
    x = x_raw
    y = 1/y_raw
    global model
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print('R^2:', r_sq)
    print('a_0:', model.intercept_)
    print('a_1:', model.coef_)

    global y_predicted
    y_predicted = model.predict(x_to_predict)
    y_predicted = 1/y_predicted



def main_menu():
    print('Wybierz akcję:')
    print('1. Załaduj dane')
    print('2. Wykres danych')
    print('3. Dopasuj model liniowy')
    print('4. Dopasuj model wykładniczy')
    print('5. Dopasuj model hiperboliczny 1/x')
    print('6. Dopasuj model hiperboliczny 1/y')
    print('9. Wyświetl model i dane')
    a = int(input('Wprowadź numer czynności:'))

    if a==1:
        get_data()
    if a==2:
        plot_data()
    if a==3:
        linear_model()
    if a==4:
        exp_model()
    if a==5:
        hiperbolic_model()
    if a==6:
        hiperbolic_model2()
    if a==9:
        plot_data_and_model()



status_active = True
while status_active:
    main_menu()



