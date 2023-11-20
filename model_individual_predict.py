import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
from ngboost import NGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


import warnings
warnings.filterwarnings("ignore")


#import model_predict_v0


train_x = np.log10(np.load('train_data_x.npy'))
train_stel = np.log10(np.load('train_data_y.npy')[:,0])
train_z = np.array([np.load('train_data_z.npy')])


test_x = np.log10(np.load('test_data_x.npy'))
#test_stel = np.log10(np.load('test_data_y.npy')[:,0])
#test_z = np.load('test_data_z.npy')

perm = np.random.permutation(len(test_x))[:20]
test_x = np.log10(np.load('test_data_x.npy'))[perm]
test_stel = np.log10(np.load('test_data_y.npy')[perm,0])
test_z = np.array([np.load('test_data_z.npy')[perm]])
test_x[test_x < -20] = -20

data_all = np.load('/orange/narayanan/d.zimmerman/simba/ml_data/all_filter_simba_snap305_m25n512.npz')
filter_wav = data_all['wav_filt'] 

#xscaler,imputer,model = model_predict_v0.get_model_info()

combined_train_x = np.concatenate((train_x,train_z.T),axis=1)
combined_test_x = np.concatenate((test_x,test_z.T),axis=1)

combined_train_x[combined_train_x < -20] = -20
combined_test_x[combined_test_x < -20] = -20

xscaler = StandardScaler()
xscaler.fit(combined_train_x)
train_x_scaled = xscaler.transform(combined_train_x)
test_x_scaled = xscaler.transform(combined_test_x)
zscaler = StandardScaler().fit(train_z.T)

knn_imputer = KNNImputer(n_neighbors = 10,weights = 'distance')
knn_imputer.fit(train_x_scaled[:,:-1])

knn_z_imputer = KNNImputer(n_neighbors = 10,weights = 'distance')
knn_z_imputer.fit(train_x_scaled)

phot_z = NGBRegressor()
print('nhgb phot')
print(np.shape(train_z.T))
phot_z.fit(train_x_scaled[:,:-1],train_z.T)
print('phot trained')

model = NGBRegressor()
model.fit(train_x_scaled,train_stel)

# detailed exploration of what is going on with the first few
filters_keep = [5,10,20]
filters_remove = [0,1,2,3,4,6,7,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]

iterations = 100#0
snr = 5

gal_phot = test_x
gal_mass = test_stel
data_imputed = []
stel_mass_out_imp = []
stel_mass_out_phot = []
z_knn = []
z_photz = []
for i in range(iterations):
    phot_noise = np.random.normal(loc=1.0,scale = 1.0/snr,size=np.shape(gal_phot))
    phot_scaled = xscaler.transform(np.concatenate((np.log10(phot_noise)+gal_phot,test_z.T),axis = 1))
    phot_trimmed = np.copy(phot_scaled)
    #print(np.shape(phot_trimmed))
    phot_trimmed[:,filters_remove] = np.nan
    # up to here should be the same as previous versions of the code
    # impute missing filters (and z)
    imputed_phot = knn_z_imputer.transform(phot_trimmed)
    imputed_phot_old = imputed_phot[:,:-1]
    imputed_phot_z = xscaler.inverse_transform(imputed_phot)[:,-1]
    z_knn.append(imputed_phot_z)
    #
    #xscaler.inverse_transform(imputed_phot_old)
    data_imputed.append(xscaler.inverse_transform(imputed_phot)[:,:-1])

    z_dist = phot_z.pred_dist(imputed_phot_old)
    mean = z_dist.params['loc']
    std = z_dist.params['scale']
    z_res = np.random.normal(loc=mean,scale=std)
    z_photz.append(z_res)

    
    pred_mass = model.pred_dist(imputed_phot)
    mean = pred_mass.params['loc']
    std = pred_mass.params['scale']
    stel_mass_out_imp.append(np.random.normal(loc=mean,scale=std))


    pred_mass = model.pred_dist(np.concatenate((imputed_phot_old,zscaler.transform(np.array([z_res]).T)),axis=1))
    mean = pred_mass.params['loc']
    std = pred_mass.params['scale']
    stel_mass_out_phot.append(np.random.normal(loc=mean,scale=std))


    #if(i%50==0):
    print(i)

print(np.shape(stel_mass_out_imp))
print(np.shape(data_imputed))

stel_med_imp = np.median(stel_mass_out_imp,axis=0)
stel_16_imp = np.quantile(stel_mass_out_imp,0.16,axis=0)
stel_84_imp = np.quantile(stel_mass_out_imp,0.84,axis=0)

stel_med_phot = np.median(stel_mass_out_phot,axis=0)
stel_16_phot = np.quantile(stel_mass_out_phot,0.16,axis=0)
stel_84_phot = np.quantile(stel_mass_out_phot,0.84,axis=0)

print(np.shape(stel_med_imp))
print(np.shape(stel_16_imp))

phot_med = np.median(data_imputed,axis=0)
phot_16 = np.quantile(data_imputed,0.16,axis=0)
phot_84 = np.quantile(data_imputed,0.84,axis=0)
print(np.shape(phot_med))
print(np.shape(phot_16))
print(np.array(data_imputed)[:,0,8])

print(np.shape(z_knn))
zk_med = np.median(z_knn,axis=0)
zk_16 = np.quantile(z_knn,0.16,axis=0)
zk_84 = np.quantile(z_knn,0.84,axis=0)

print(np.shape(z_photz))
zp_med = np.median(z_photz,axis=0)
zp_16 = np.quantile(z_photz,0.16,axis=0)
zp_84 = np.quantile(z_photz,0.84,axis=0)

for i in range(20):
    plt.plot(filter_wav,gal_phot[i],color='black',label='true')
    plt.scatter(filter_wav[filters_keep],gal_phot[i][filters_keep],color='black',label='phot given',alpha=0.5)
    plt.plot(filter_wav,phot_med[i],color = 'blue',label = 'median KNN',alpha=0.8)
    plt.fill_between(filter_wav,phot_16[i],phot_84[i],color='blue',alpha=0.5)
    plt.xscale('log')
    plt.xlabel('$\\lambda$')
    plt.ylabel('log flux')
    plt.title(f'True M*:{np.round(gal_mass[i],2)}, M* (imp):{np.round(stel_med_imp[i],2)}-{np.round(stel_med_imp[i]-stel_16_imp[i],2)}+{np.round(stel_84_imp[i]-stel_med_imp[i],2)},M* (phot-z):{np.round(stel_med_phot[i],2)}-{np.round(stel_med_phot[i]-stel_16_phot[i],2)}+{np.round(stel_84_phot[i]-stel_med_phot[i],2)}\ntrue z: {test_z[0,i]},imputed z: [{np.round(zk_16[i],2)},{np.round(zk_med[i],2)},{np.round(zk_84[i],2)}],phot z: [{np.round(zp_16[i],2)},{np.round(zp_med[i],2)},{np.round(zp_84[i],2)}]')
    plt.legend()
    plt.savefig(f'model_v0_outfigs/spec_z_gal{i}.png')
    plt.close()

