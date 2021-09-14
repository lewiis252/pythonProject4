import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import PySimpleGUI as sg
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
import random
import tensorflow

sb.set()

x_raw = []
print = sg.Print
'''get data from file'''


def get_data():
    global data
    data = pd.read_excel('dane.xlsx', sheet_name='data', engine='openpyxl')
    global x_raw, y_raw, x_to_predict
    x_raw = np.array(data['x']).reshape((-1, 1))
    x_raw = np.vstack(x_raw)

    y_raw = np.array(data['y'])

    x_to_predict = np.linspace(min(x_raw), max(x_raw), 500)


'''plot data'''


def plot_data():
    if len(x_raw) > 0:
        sb.scatterplot(data=data, x='x', y='y', color='blue')
        plt.show()
    else:
        print('Nie załadowano danych!')


'''plot data and model'''


def plot_data_and_model():
    if len(x_raw) > 0:
        sb.scatterplot(data=data, x='x', y='y', color='blue')
        # prediction based on model
        plt.plot(x_to_predict, y_predicted, color='orange')
        plt.show()
    else:
        print('Nie załadowano danych!')


def random_data():
    global data
    n = random.randint(1, 100)
    # n=6
    # x = [random.random() * 1000 for i in range(0, n)]
    x = np.linspace(0.01, 1000, n)

    y = [random.random() * 100]
    for i in range(1, n):
        y_ = y[i - 1] + (np.sin(random.random() * 2 * np.pi)) * 10
        y.append(y_)

    d = {'x': x, 'y': y}
    data = pd.DataFrame(data=d)
    global x_raw, y_raw, x_to_predict
    x_raw = np.array(data['x']).reshape(((-1, 1)))
    y_raw = np.array(data['y'])
    x_to_predict = np.linspace(min(x_raw), max(x_raw), 500)


'''linear model'''


def linear_model():
    if len(x_raw) > 0:
        x = x_raw
        y = y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Model liniowy')
        print('Y=', model.intercept_, '+', model.coef_, 'x')
        print('R^2:', r_sq)

        global y_predicted
        y_predicted = model.predict(x_to_predict)
    else:
        print('Nie załadowano danych!')

'''Ridge and Lasso model works but of course there is no point to use them to fit such a simple regression'''
# '''linear Ridge'''
# def ridge_lin_model():
#     if len(x_raw) > 0:
#         x = x_raw
#         y = y_raw
#         global model
#         model = Ridge().fit(x, y)
#         r_sq = model.score(x, y)
#         print('')
#         print('Model liniowy Ridge')
#         print('Y=', model.intercept_, '+', model.coef_, 'x')
#         print('R^2:', r_sq)
#
#         global y_predicted
#         y_predicted = model.predict(x_to_predict)
#     else:
#         print('Nie załadowano danych!')
#
#
# '''linear Lasso'''
#
#
# def lasso_lin_model():
#     if len(x_raw) > 0:
#         x = x_raw
#         y = y_raw
#         global model
#         model = Lasso().fit(x, y)
#         r_sq = model.score(x, y)
#         print('')
#         print('Model liniowy Ridge')
#         print('Y=', model.intercept_, '+', model.coef_, 'x')
#         print('R^2:', r_sq)
#
#         global y_predicted
#         y_predicted = model.predict(x_to_predict)
#     else:
#         print('Nie załadowano danych!')


''' exponential model '''


def exp_model():
    if len(x_raw) > 0:
        v = np.log(y_raw)

        global model
        model = LinearRegression().fit(x_raw, v)
        r_sq = model.score(x_raw, v)
        print('')
        print('Model wykładniczy')
        print('Y=', np.exp(model.intercept_), 'exp(', model.coef_[0], 't', ')')
        print('R^2:', r_sq)

        # prediction based on model
        global y_predicted
        y_predicted = np.exp(model.predict(x_to_predict))
    else:
        print('Nie załadowano danych!')


''' power model '''


def exp_model2():
    if len(x_raw) > 0:
        v = np.log(y_raw)
        z = np.log(x_raw)

        global model
        model = LinearRegression().fit(z, v)
        r_sq = model.score(z, v)
        print('')
        print('Model potęgowy ')
        a_0 = np.exp(model.intercept_)
        print('Y=', a_0, 'x^(', model.coef_[0], ')')
        print('R^2:', r_sq)

        # prediction based on model
        global y_predicted
        y_predicted = np.exp(model.predict(np.log(x_to_predict)))
    else:
        print('Nie załadowano danych!')


'''hiperbolic model 1/x'''


def hiperbolic_model():
    if len(x_raw) > 0:
        x = 1 / x_raw
        y = y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Model hiperboliczny 1/x')
        print('Y=', model.intercept_, '+', model.coef_, '/x')
        print('R^2:', r_sq)

        global y_predicted
        y_predicted = model.predict(1 / x_to_predict)
    else:
        print('Nie załadowano danych!')


'''hiperbolic model 1/y'''


def hiperbolic_model2():
    if len(x_raw) > 0:
        x = x_raw
        y = 1 / y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Model hiperboliczny 1/y')
        print('Y=1/[', model.intercept_, '+', model.coef_, 'x]')
        print('R^2:', r_sq)

        global y_predicted
        y_predicted = model.predict(x_to_predict)
        y_predicted = 1 / y_predicted
    else:
        print('Nie załadowano danych!')


'''logarithmic model'''


def logarithmic_model():
    if len(x_raw) > 0:
        x = np.log(x_raw)
        y = y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Model logarytmiczny')
        print('Y=', model.intercept_, '+ ln(', model.coef_, 'x)')
        print('R^2:', r_sq)

        global y_predicted
        y_predicted = model.predict(np.log(x_to_predict))

    else:
        print('Nie załadowano danych!')


'''2 degree polynomial model '''


def polynomial2_model():
    if len(x_raw) > 0:
        x = x_raw
        transformer = PolynomialFeatures(degree=2, include_bias=False)
        transformer.fit(x)
        x = transformer.transform(x)

        y = y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Wielomian 2 stopnia')
        print('Y=', model.intercept_, '+', model.coef_[1], 'x', '+', model.coef_[0], 'x^2')
        print('R^2:', r_sq)

        global y_predicted

        y_predicted = model.predict(PolynomialFeatures(degree=2, include_bias=False).fit_transform(x_to_predict))

    else:
        print('Nie załadowano danych!')


'''3 degree polynomial model '''


def polynomial3_model():
    if len(x_raw) > 0:
        x = x_raw
        transformer = PolynomialFeatures(degree=3, include_bias=False)
        transformer.fit(x)
        x = transformer.transform(x)

        y = y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Wielomian 3 stopnia')
        print('Y=', model.intercept_, '+', model.coef_[2], 'x', '+', model.coef_[1], 'x^2', '+', model.coef_[0], 'x^3')
        print('R^2:', r_sq)

        global y_predicted

        y_predicted = model.predict(PolynomialFeatures(degree=3, include_bias=False).fit_transform(x_to_predict))

    else:
        print('Nie załadowano danych!')


'''4 degree polynomial model '''


def polynomial4_model():
    if len(x_raw) > 0:
        x = x_raw
        transformer = PolynomialFeatures(degree=4, include_bias=False)
        transformer.fit(x)
        x = transformer.transform(x)

        y = y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Wielomian 4 stopnia')
        print('Y=', model.intercept_, '+', model.coef_[3], 'x', '+', model.coef_[2], 'x^2', '+', model.coef_[1], 'x^3',
              '+', model.coef_[0], 'x^4')
        print('R^2:', r_sq)

        global y_predicted

        y_predicted = model.predict(PolynomialFeatures(degree=4, include_bias=False).fit_transform(x_to_predict))

    else:
        print('Nie załadowano danych!')


'''5 degree polynomial model '''


def polynomial5_model():
    if len(x_raw) > 0:
        x = x_raw
        transformer = PolynomialFeatures(degree=5, include_bias=False)
        transformer.fit(x)
        x = transformer.transform(x)

        y = y_raw
        global model
        model = LinearRegression().fit(x, y)
        r_sq = model.score(x, y)
        print('')
        print('Wielomian 5 stopnia')
        print('Y=', model.intercept_, '+', model.coef_[4], 'x', '+', model.coef_[3], 'x^2', '+', model.coef_[2], 'x^3',
              '+', model.coef_[1], 'x^4', '+', model.coef_[0], 'x^5')
        print('R^2:', r_sq)

        global y_predicted

        y_predicted = model.predict(PolynomialFeatures(degree=5, include_bias=False).fit_transform(x_to_predict))

    else:
        print('Nie załadowano danych!')


def best_poly_model():
    if len(x_raw) > 0:
        def PolynomialRegression(degree=2, **kwargs):
            return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

        param_grid = {'polynomialfeatures__degree': np.arange(5),
                      'linearregression__fit_intercept': [True, False],
                      'linearregression__normalize': [True, False]}
        split = int(np.ceil(len(x_raw) / 3))

        x = np.array(x_raw).reshape((-1, 1))
        y = np.array(y_raw)
        grid = GridSearchCV(PolynomialRegression(), param_grid, cv=split)
        grid.fit(x, y)
        print(grid.best_params_)

        model = grid.best_estimator_
        r_sq = model.score(x, y)
        print('R^2:', r_sq)

        global x_to_predict, y_predicted
        x_to_predict = np.linspace(min(x), max(x), 1000).reshape((-1, 1))
        y_predicted = model.fit(x, y).predict(x_to_predict)

    else:
        print('Nie załadowano danych!')


def poly_model(deg):
    if len(x_raw) > 0 and int(values[0]) >= 1:
        x = np.array(x_raw).reshape((-1, 1))
        y = np.array(y_raw)

        x_ = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x)

        model = LinearRegression().fit(x_, y)
        r_sq = model.score(x_, y)
        print('R^2:', r_sq)
        print('intercept:', model.intercept_)
        print('slope:', model.coef_)

        global x_to_predict, y_predicted
        x_to_predict = np.linspace(min(x), max(x), 1000).reshape((-1, 1))
        x_to_predict_ = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x_to_predict)
        y_predicted = (model.predict((x_to_predict_)))

    else:
        print('Nie załadowano danych lub podano niepoprawny stopień wielomianu - wymagana liczba naturalna.')


'''gui'''

sg.theme('DarkBrown1')  # Add a touch of color
# All the stuff inside your window.
layout = [[sg.Text('Wybierz akcję')],
          [sg.Button('Załaduj dane')],
          [sg.Button('Wylosuj dane')],
          [sg.Button('Wykres danych')],
          [sg.Button('Dopasuj model liniowy')],
          # [sg.Button("Dopasuj model liniowy Ridge'a")],
          # [sg.Button("Dopasuj model liniowy Lasso")],
          [sg.Button('Dopasuj model wykładniczy')],
          [sg.Button('Dopasuj model logarytmiczny')],
          [sg.Button('Dopasuj model potęgowy')],
          [sg.Button('Dopasuj model hiperboliczny 1/x')],
          [sg.Button('Dopasuj model hiperboliczny 1/y')],
          [sg.Button('Dopasuj wielomian 2 stopnia')],
          [sg.Button('Dopasuj wielomian 3 stopnia')],
          [sg.Button('Dopasuj wielomian 4 stopnia')],
          [sg.Button('Dopasuj wielomian 5 stopnia')],
          [sg.Button('Znajdz wielomian o najlepszych parametrach')],
          [sg.Button('Dopasuj wielomian stopnia:')],
          [sg.InputText()],
          [sg.Button('Wyświetl model i dane')],
          [sg.Button('Wyjście')]]

# Create the Window
window = sg.Window('Fit model', layout)
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    if event == 'Załaduj dane':
        get_data()
    if event == 'Wykres danych':
        plot_data()
    if event == 'Dopasuj model liniowy':
        linear_model()
    # if event == "Dopasuj model liniowy Ridge'a":
    #     ridge_lin_model()
    # if event == "Dopasuj model liniowy Lasso":
    #     lasso_lin_model()
    if event == 'Dopasuj model wykładniczy':
        exp_model()
    if event == 'Dopasuj model potęgowy':
        exp_model2()
    if event == 'Dopasuj model hiperboliczny 1/x':
        hiperbolic_model()
    if event == 'Dopasuj model hiperboliczny 1/y':
        hiperbolic_model2()
    if event == 'Dopasuj wielomian 2 stopnia':
        polynomial2_model()
    if event == 'Dopasuj wielomian 3 stopnia':
        polynomial3_model()
    if event == 'Dopasuj wielomian 4 stopnia':
        polynomial4_model()
    if event == 'Dopasuj wielomian 5 stopnia':
        polynomial5_model()
    if event == 'Dopasuj model logarytmiczny':
        logarithmic_model()
    if event == 'Znajdz wielomian o najlepszych parametrach':
        best_poly_model()
    if event == 'Dopasuj wielomian stopnia:':
        poly_model(int(values[0]))
    if event == 'Wylosuj dane':
        random_data()
    if event == 'Wyświetl model i dane':
        plot_data_and_model()
    if event == sg.WIN_CLOSED or event == 'Wyjście':  # if user closes window or clicks cancel
        break
