import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''Kreator testów statystycznych dla dwóch grup, skala pomiarowa zmiennej zależnej ilościowa.'''

grupujaca = 'Płeć'
zalezna = 'Tg'

def get_data():
    global data
    data = pd.read_excel('dane.xlsx', sheet_name='grupy')
    # data = data.dropna(subset=[zalezna]) # z jakiegoś powodu wyrzuca błędy

    data = data.fillna(data[zalezna].mean())
    # print(data.to_string())
    pd.set_option('display.max_columns', None)
    # print(data.head())



def violin_plot():
    sns.violinplot(x=grupujaca, y=zalezna, data=data,
                   palette=["lightblue", "lightpink"])
    plt.show()


def box_plot():
    sns.boxplot(x=grupujaca, y=zalezna, data=data,
                   palette=["lightblue", "lightpink"])
    plt.show()

def plot_histograms():
    sns.displot(data, x=zalezna, col=grupujaca, multiple="dodge")
    plt.show()


def split_data_by_groups():
    global grupy, g1, g2, g
    grupy = []
    for i in data[grupujaca]:
        if i not in grupy:
            grupy.append(i)
    #print(grupy)

    g1 = []
    g2 = []

    for i in range(len(data[grupujaca])):
        if data[grupujaca][i] == grupy[0]:
            g1.append(data[zalezna][i])
        else:
            g2.append(data[zalezna][i])

    g = [g1, g2]




def shapiro_wilk_test():
    from scipy.stats import shapiro
    global isnormal
    for i in range(2):
        stat, p = shapiro(g[i])
        print('Dla grupy:', grupy[i])
        print('stat=%.3f, p=%.3f' % (stat, p))
        if p > 0.05:
            print('Rozkład jest normalny \n')
            isnormal = True
        else:
            print('Rozkład nie jest normalny \n')
            isnormal = False



def variance_bartlett_test():
    from scipy.stats import bartlett
    global eqvar
    stat, p = bartlett(g1,g2)
    print('stat=%.3f, p=%.3f' % (stat, p))

    if p > 0.05:
        print('Wariancje są jednorodne \n')
        eqvar = True
    else:
        print('Wariancje są jednorodne \n')
        eqvar = False

def variance_levene_test():
    from scipy.stats import levene
    stat, p = levene(g1,g2)
    print('levene')
    print('stat=%.3f, p=%.3f' % (stat, p))

    if p > 0.05:
        print('Wariancje są jednorodne \n')
    else:
        print('Wariancje są jednorodne \n')

def mean_t_test():
    from scipy.stats import ttest_ind

    stat, p = ttest_ind(g1, g2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Średnie są równe')
    else:
        print('Średnie nie są równe')

def mean_Welch_test():
    from scipy.stats import ttest_ind

    stat, p = ttest_ind(g1, g2, equal_var=False)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Średnie są równe')
    else:
        print('Średnie nie są równe')

def Mann_Whitney():
    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(g1, g2)
    print('stat=%.3f, p=%.3f' % (stat, p))
    if p > 0.05:
        print('Probably the same distribution')
    else:
        print('Probably different distributions')

def test_tree():
    get_data()

    split_data_by_groups()

    shapiro_wilk_test()

    if isnormal:
        variance_bartlett_test()
        if eqvar:
            mean_t_test()
        else: mean_Welch_test()
    else: Mann_Whitney()


get_data()
violin_plot()
box_plot()
plot_histograms()
test_tree()
