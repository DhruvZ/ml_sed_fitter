import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import joblib
from ngboost import NGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


test_x = np.load('test_data_x.npy')
test_y = np.load('test_data_y.npy')

def get_model_info():
    xscaler = joblib.load('model_dir/scaler_v0.joblib')
    imputer = joblib.load('model_dir/knn_v0.joblib')
    model = joblib.load('model_dir/stel_ng_v0.joblib')
    return xscaler,imputer,model


xscaler,imputer,model = get_model_info()


test_x_scaled = xscaler.transform(np.log10(test_x))

snr = 5
noise = np.random.normal(loc=1.0,scale=1.0/snr,size = np.shape(test_x))


test_x_noise = test_x*noise
test_x_noise = np.log10(test_x_noise)
test_x_noise_scaled = xscaler.transform(test_x_noise)


ng_test_full_dist = model.pred_dist(test_x_scaled)
ng_test_mean = ng_test_full_dist.params['loc']
ng_test_std = ng_test_full_dist.params['scale']
#print(ng_test_mean)
#print(ng_test_std)


filter_remove = (np.random.randint(low=0,high=8,size = np.shape(test_x_noise_scaled))>0)
#print(filter_remove)
test_x_noise_scaled[filter_remove] = np.nan
#print(test_x_noise_scaled)
#print(np.mean(filter_remove))

test_x_noise_imputed = imputer.transform(test_x_noise_scaled)

filter_count = np.sum(~filter_remove,axis=1)
print(np.shape(filter_count))

ng_test_noise_dist = model.pred_dist(test_x_noise_imputed)
ng_test_noise_mean = ng_test_noise_dist.params['loc']
ng_test_noise_std = ng_test_noise_dist.params['scale']

stel_mass_test = np.log10(test_y[:,0])



oto = np.linspace(7,12,100)
plt.plot(oto,oto,linestyle = '--',color = 'black')
plt.errorbar(stel_mass_test,ng_test_mean,yerr = ng_test_std,fmt='o',label='NG',alpha=0.5,markersize=3)
#plt.errorbar(stel_mass_test,ng_test_noise_mean,yerr = ng_test_noise_std,fmt='o',label='NG imperfect',alpha=0.5,markersize=3)
plt.scatter(stel_mass_test,ng_test_noise_mean,c=filter_count,label='NG imperfect',s=3)
plt.colorbar()

plt.xlabel('log true M*')
plt.ylabel('log predicted M*')
plt.xlim(7.2,12)
plt.ylim(7.2,12)
plt.title(f'Full run test 0')
plt.legend()
plt.savefig('model_test0.png')
plt.close()

sig_offset_full = np.abs((ng_test_mean-stel_mass_test)/ng_test_std)
sig_offset_noise = np.abs((ng_test_noise_mean-stel_mass_test)/ng_test_noise_std)


plt.hist(sig_offset_full,density = True,cumulative=True,label='full',alpha=0.5)
plt.hist(sig_offset_noise,density = True,cumulative=True,label='imperfect',alpha=0.5)
plt.legend()
plt.xlim(right=8)
plt.xlabel('sigma of prediction from true value')
plt.savefig('error_test0.png')
plt.close()

