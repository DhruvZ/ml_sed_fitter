import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
from ngboost import NGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

import model_predict_v0

test_x = np.load('test_data_x.npy')
test_y = np.load('test_data_y.npy')


data_all = np.load('/orange/narayanan/d.zimmerman/simba/ml_data/all_filter_simba_snap305_m25n512.npz')
filter_wav = data_all['wav_filt'] 

xscaler,imputer,model = model_predict_v0.get_model_info()

# detailed exploration of what is going on with the first few
filters_keep = [5,10,20]
filters_remove = [0,1,2,3,4,6,7,8,9,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

iterations = 1000
snr = 5



gal_phot = test_x
gal_mass = test_y[:,0]
data_imputed = []
stel_mass_out = []
for i in range(iterations):
    phot_noise = gal_phot*np.random.normal(loc=1.0,scale = 1.0/snr,size=np.shape(gal_phot))
    phot_scaled = xscaler.transform(np.log10(phot_noise))
    phot_trimmed = np.copy(phot_scaled)
    #print(np.shape(phot_trimmed))
    phot_trimmed[:,filters_remove] = np.nan
    imputed_phot = imputer.transform(phot_trimmed)
    data_imputed.append(xscaler.inverse_transform(imputed_phot))
    pred_mass = model.pred_dist(imputed_phot)
    mean = pred_mass.params['loc']
    std = pred_mass.params['scale']
    stel_mass_out.append(np.random.normal(loc=mean,scale=std))
    if(i%50==0):
        print(i)

print(np.shape(stel_mass_out))
print(np.shape(data_imputed))

stel_med = np.median(stel_mass_out,axis=0)
stel_16 = np.quantile(stel_mass_out,0.16,axis=0)
stel_84 = np.quantile(stel_mass_out,0.84,axis=0)
print(np.shape(stel_med))
print(np.shape(stel_16))

phot_med = np.median(data_imputed,axis=0)
phot_16 = np.quantile(data_imputed,0.16,axis=0)
phot_84 = np.quantile(data_imputed,0.84,axis=0)
print(np.shape(phot_med))
print(np.shape(phot_16))


for i in range(20):
    plt.plot(filter_wav,np.log10(gal_phot[i]),color='black',label='true')
    plt.scatter(filter_wav[filters_keep],np.log10(gal_phot[i][filters_keep]),color='black',label='phot given',alpha=0.5)
    plt.plot(filter_wav,phot_med[i],color = 'blue',label = 'median KNN',alpha=0.8)
    plt.fill_between(filter_wav,phot_16[i],phot_84[i],color='blue',alpha=0.5)
    plt.xscale('log')
    plt.xlabel('$\\lambda$')
    plt.ylabel('log flux')
    plt.title(f'True M*:{np.round(np.log10(gal_mass[i]),2)}, Predicted M*:{np.round(stel_med[i],2)}-{np.round(stel_med[i]-stel_16[i],2)}+{np.round(stel_84[i]-stel_med[i],2)}')
    plt.legend()
    plt.savefig(f'model_v0_outfigs/spec_gal{i}.png')
    plt.close()

