import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split #splitting data into training and test sets

#assume that there is some pre-formatted data
#should include 1) different filter values, 2) stellar mass, 3) dust mass, 4) sfr, 5) metallicity

labels = ['stellar_mass','dust_mass','sfr100','metallicity']

data_config = 'train_test_config']

data_dict = {'train':[],'test':[],'both':[]}
data_cats = ['train','test','both']
current_cat = None

with open('train_test_config') as f:
    for line in f:
        if(line == 'train' or line == 'test' or line == 'both'):
            current_cat = line
        elif(current_cat != None):
            data_dict[current_cat].append(line)

train_data_x = []
test_data_x = []
overlap_x = []

all_x = {'train':train_dat_x,'test':test_data_x,'both':overlap_x}

train_data_y = []
test_data_y = []
overlap_y = []

all_y = {'train':train_dat_y,'test':test_data_y,'both':overlap_y}

for cat in data_cats:
    if(len(data_dict[cat]) > 0):
    for loc in data_dict[cat]:
        temp = np.load(loc)
        temp_x = [f_val for f_val in temp['filter']]
        all_x[cat] = all_x[cat] + temp_x
        for i in len(temp[labels[0]):
            temp_y = [temp[label][i] for label in temp[label]]
            all_y[cat] = all_y[cat]+temp_y


if(len(data_dict['both']) > 0):
    train_X, test_X, train_y, test_y = train_test_split(all_x['both'], all_y['both'], test_size=0.3)

train_x_all = all_x['train'] + train_X
train_y_all = all_y['train'] + train_y

test_x_all = all_x['test'] + test_X
test_y_all = all_y['test'] + test_y

np.save('train_data_x.npy',train_x_all)
np.save('train_data_y.npy',train_y_all)
np.save('test_data_x.npy',test_x_all)
np.save('test_data_y.npy',test_y_all)
