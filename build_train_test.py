import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split #splitting data into training and test sets
import sklearn
from sklearn.model_selection import GridSearchCV


def hyp_opt(train_X,train_y,model,param_grid):
    grid = GridSearchCV(model,param_grid,return_train_score=True,cv=5,verbose=0) # can change verbose to other values for more printing
    grid.fit(train_X,train_y)
    print("Best parameters: {}".format(grid.best_params_))



#assume that there is some pre-formatted data
#should include 1) different filter values, 2) stellar mass, 3) dust mass, 4) sfr, 5) metallicity

labels = ['stellar_mass','dust_mass','sfr']#,'metallicity','ages']

data_config = 'train_test_config.txt'

data_dict = {'train':[],'test':[],'both':[]}
data_cats = ['train','test','both']
current_cat = None

with open('train_test_config.txt') as f:
    for l in f:
        line = l.rstrip()
        if(line == 'train' or line == 'test' or line == 'both'):
            current_cat = line
        elif(current_cat != None):
            data_dict[current_cat].append(line)

print(data_dict)

train_data_x = []
test_data_x = []
overlap_x = []

all_x = {'train':train_data_x,'test':test_data_x,'both':overlap_x}

train_data_y = []
test_data_y = []
overlap_y = []

all_y = {'train':train_data_y,'test':test_data_y,'both':overlap_y}

for cat in data_cats:
    if(len(data_dict[cat]) > 0):
        for loc in data_dict[cat]:
            temp = np.load(loc)
            temp_x = [f_val for f_val in temp['phot_data']]
            all_x[cat] = all_x[cat] + temp_x
            for i in range(len(temp[labels[0]])):
                temp_y = [[temp[label][i] for label in labels]]
                all_y[cat] = all_y[cat]+temp_y



if(len(data_dict['both']) > 0):
    train_X, test_X, train_y, test_y = train_test_split(all_x['both'], all_y['both'], test_size=0.3)
else:
    train_X = []
    test_X = []
    train_y = []
    test_y = []


train_x_all = all_x['train'] + train_X
train_y_all = all_y['train'] + train_y

test_x_all = all_x['test'] + test_X
test_y_all = all_y['test'] + test_y

print(np.shape(train_x_all))
print(np.shape(train_y_all))
print(np.shape(test_x_all))
print(np.shape(test_y_all))

np.save('train_data_x.npy',train_x_all)
np.save('train_data_y.npy',train_y_all)
np.save('test_data_x.npy',test_x_all)
np.save('test_data_y.npy',test_y_all)
