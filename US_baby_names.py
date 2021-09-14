import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names1880 = pd.read_csv('names/yob1880.txt', names=['name', 'sex', 'births'])

# group and sum by sex
print(names1880.groupby('sex').births.sum())

years = range(1880, 2020)

# a list of every names
pieces = []

columns = ['name', 'sex', 'births']

for year in years:
    path = 'names/yob%d.txt' %year
    frame = pd.read_csv(path, names=columns)

    frame['year'] = year
    pieces.append(frame)


# concantenate into single dataframe
names = pd.concat(pieces, ignore_index=True)


total_births = names.pivot_table('births', index = ['year'], columns = ['sex'], aggfunc=sum)
print(total_births.tail())

def add_prop(group):

    births = group.births.astype(float)

    group['prop'] = births / births.sum()
    return group

total_births.plot(title='Total births by sex and year')

names = names.groupby(['year', 'sex']).apply(add_prop)
print(names.tail())

# check if sum of % is 1
print(np.allclose(names.groupby(['year', 'sex']).prop.sum(),1))

def get_top1000(group):
    return group.sort_index(by='births', ascending=False)[:1000]

grouped = names.groupby(['year', 'sex'])
top1000 = grouped.apply(get_top1000)
plt.show()

