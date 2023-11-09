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
    model_stel = joblib.load('model_dir/stel_ng_v0.joblib')
    model_dust = joblib.load('model_dir/dust_ng_v0.joblib')
    model_sfr = joblib.load('model_dir/sfr_ng_v0.joblib')
    model_metal = joblib.load('model_dir/metal_ng_v0.joblib')
    model_age = joblib.load('model_dir/age_ng_v0.joblib')
    return xscaler,imputer,model_stel,model_dust,model_sfr,model_metal,model_age


def process_error(data_val,pred_med,pred16,pred84):
    
    #print(np.shape(pred_med))
    #print(np.shape(data_val))

    err_abs = pred_med-data_val
    #print(err_abs[:5])
    err_dev = err_abs*0
    #print(err_dev[:5])
    err_dev[err_abs <= 0] = err_abs[err_abs <= 0]/(pred84-pred_med)[err_abs <= 0]
    #print(err_dev[:5])
    err_dev[err_abs > 0] = err_abs[err_abs > 0]/(pred_med-pred16)[err_abs > 0]
    #print(err_dev[:5])    
    return np.abs(err_dev)

xscaler,imputer,model_stel,model_dust,model_sfr,model_metal,model_age = get_model_info()
test_x_scaled = xscaler.transform(np.log10(test_x))

stel_full = model_stel.pred_dist(test_x_scaled)
stel_full_mean = stel_full.params['loc']
stel_full_std = stel_full.params['scale']

dust_full = model_dust.pred_dist(test_x_scaled)
dust_full_mean = dust_full.params['loc']
dust_full_std = dust_full.params['scale']

sfr_full = model_sfr.pred_dist(test_x_scaled)
sfr_full_mean = sfr_full.params['loc']
sfr_full_std = sfr_full.params['scale']

#metal_full = model_metal.pred_dist(test_x_scaled)
#metal_full_mean = metal_full.params['loc']
#metal_full_std = metal_full.params['scale']

#age_full = model_age.pred_dist(test_x_scaled)
#age_full_mean = age_full.params['loc']
#age_full_std = age_full.params['scale']

# start copy
iterations = 100#0
snr = 5

filter_count = np.random.randint(len(test_x[0])-7,len(test_x[0]),size = len(test_x))

filter_arr = np.full(np.shape(test_x),1)
for i in range(len(filter_count)):
    count = filter_count[i]
    perm = np.random.permutation(len(test_x[0]))
    perm = perm[:count]
    filter_arr[i,perm] = 0
#print(filter_arr)
filter_remove = (filter_arr == 0)
#print(filter_remove)

#filter_count = np.sum(~filter_remove,axis=1)
#print(filter_count)

gal_phot = test_x
gal_stel = np.log10(test_y[:,0])
gal_dust = np.log10(test_y[:,1])
gal_sfr = np.log10(test_y[:,2]+1)
#gal_metal = np.log10(test_y[:,3])
#gal_age = test_y[:,4]

data_imputed = []
stel_mass_out = []
dust_mass_out = []
sfr_out = []
metal_out = []
age_out = []

for i in range(iterations):
    phot_noise = gal_phot*np.random.normal(loc=1.0,scale = 1.0/snr,size=np.shape(gal_phot))
    phot_scaled = xscaler.transform(np.log10(phot_noise))
    phot_trimmed = np.copy(phot_scaled)
    #print(np.shape(phot_trimmed))
    phot_trimmed[filter_remove] = np.nan
    #print(phot_trimmed)
    #raise Exception()
    imputed_phot = imputer.transform(phot_trimmed)
    data_imputed.append(xscaler.inverse_transform(imputed_phot))
    
    pred_stel = model_stel.pred_dist(imputed_phot)
    mean = pred_stel.params['loc']
    std = pred_stel.params['scale']
    stel_mass_out.append(np.random.normal(loc=mean,scale=std))
    
    pred_dust = model_dust.pred_dist(imputed_phot)
    mean = pred_dust.params['loc']
    std = pred_dust.params['scale']
    dust_mass_out.append(np.random.normal(loc=mean,scale=std))

    pred_sfr = model_sfr.pred_dist(imputed_phot)
    mean = pred_sfr.params['loc']
    std = pred_sfr.params['scale']
    sfr_out.append(np.random.normal(loc=mean,scale=std))

    #pred_metal = model_metal.pred_dist(imputed_phot)
    #mean = pred_metal.params['loc']
    #std = pred_metal.params['scale']
    #metal_out.append(np.random.normal(loc=mean,scale=std))

    #pred_age = model_age.pred_dist(imputed_phot)
    #mean = pred_age.params['loc']
    #std = pred_age.params['scale']
    #age_out.append(np.random.normal(loc=mean,scale=std))
    
    #if(i%50==0):
    print(i)

print(np.shape(stel_mass_out))
print(np.shape(data_imputed))

stel_med = np.median(stel_mass_out,axis=0)
stel_16 = np.quantile(stel_mass_out,0.16,axis=0)
stel_84 = np.quantile(stel_mass_out,0.84,axis=0)

#print(np.shape(stel_med))
#print(np.shape(stel_16))

dust_med = np.median(dust_mass_out,axis=0)
dust_16 = np.quantile(dust_mass_out,0.16,axis=0)
dust_84 = np.quantile(dust_mass_out,0.84,axis=0)

sfr_med = np.median(sfr_out,axis=0)
sfr_16 = np.quantile(sfr_out,0.16,axis=0)
sfr_84 = np.quantile(sfr_out,0.84,axis=0)

#metal_med = np.median(metal_out,axis=0)
#metal_16 = np.quantile(metal_out,0.16,axis=0)
#metal_84 = np.quantile(metal_out,0.84,axis=0)

#age_med = np.median(age_out,axis=0)
#age_16 = np.quantile(age_out,0.16,axis=0)
#age_84 = np.quantile(age_out,0.84,axis=0)


phot_med = np.median(data_imputed,axis=0)
phot_16 = np.quantile(data_imputed,0.16,axis=0)
phot_84 = np.quantile(data_imputed,0.84,axis=0)
#print(np.shape(phot_med))
#print(np.shape(phot_16))

# end copy


oto = np.linspace(7,12.5,100)
plt.plot(oto,oto,linestyle = '--',color = 'black')
plt.errorbar(gal_stel,stel_full_mean,yerr = stel_full_std,fmt='o',label='NG',alpha=0.5,markersize=3)
plt.xlabel('log true M*')
plt.ylabel('log predicted M*')
plt.xlim(7.2,12.5)
plt.ylim(7.2,12.5)
plt.title(f'Full run test 0.1')
plt.savefig('model_v0_outfigs/model_test0.1_stel_full.png')
plt.errorbar(gal_stel,stel_med,yerr = [stel_med-stel_16,stel_84-stel_med],fmt='o',label='NG imperfect',alpha=0.5,markersize=3)
plt.legend()
plt.savefig('model_v0_outfigs/model_test0.1_stel.png')
plt.close()



oto = np.linspace(2,9,100)
plt.plot(oto,oto,linestyle = '--',color = 'black')
plt.errorbar(gal_dust,dust_full_mean,yerr = dust_full_std,fmt='o',label='NG',alpha=0.5,markersize=3)
plt.xlabel('log true M_dust')
plt.ylabel('log predicted M_dust')
plt.xlim(2,9)
plt.ylim(2,9)
plt.title(f'Full run test 0.1')
plt.savefig('model_v0_outfigs/model_test0.1_dust_full.png')
plt.errorbar(gal_dust,dust_med,yerr = [dust_med-dust_16,dust_84-dust_med],fmt='o',label='NG imperfect',alpha=0.5,markersize=3)
plt.legend()
plt.savefig('model_v0_outfigs/model_test0.1_dust.png')
plt.close()


oto = np.linspace(-0.1,2.5,100)
plt.plot(oto,oto,linestyle = '--',color = 'black')
plt.errorbar(gal_sfr,sfr_full_mean,yerr = sfr_full_std,fmt='o',label='NG',alpha=0.5,markersize=3)
plt.xlabel('true log (sfr+1)')
plt.ylabel('predicted log (sfr+1)')
plt.xlim(-0.1,1.2)
plt.ylim(-0.1,1.2)
plt.title(f'Full run test 0.1')
plt.savefig('model_v0_outfigs/model_test0.1_sfr_full.png')
plt.errorbar(gal_sfr,sfr_med,yerr = [sfr_med-sfr_16,sfr_84-sfr_med],fmt='o',label='NG imperfect',alpha=0.5,markersize=3)
plt.legend()
plt.savefig('model_v0_outfigs/model_test0.1_sfr.png')
plt.close()


#oto = np.linspace(-3.4,1.5,100)
#plt.plot(oto,oto,linestyle = '--',color = 'black')
#plt.errorbar(gal_metal,metal_full_mean,yerr = metal_full_std,fmt='o',label='NG',alpha=0.5,markersize=3)
#plt.scatter(gal_metal,metal_full_mean,alpha=0.5,s=3)
#plt.xlabel('log true stellar metallicity')
#plt.ylabel('predicted log stellar metallicity')
#plt.xlim(-3.3,-1.6)
#plt.ylim(-3.3,-1.6)
#plt.title(f'Full run test 0.1')
#plt.savefig('model_v0_outfigs/model_test0.1_metal_full.png')
#plt.errorbar(gal_metal,metal_med,yerr = [metal_med-metal_16,metal_84-metal_med],fmt='o',label='NG imperfect',alpha=0.5,markersize=3)
#plt.legend()
#plt.savefig('model_v0_outfigs/model_test0.1_metal.png')
#plt.close()


#oto = np.linspace(1,14,100)
#plt.plot(oto,oto,linestyle = '--',color = 'black')
#plt.errorbar(gal_age,age_full_mean,yerr = age_full_std,fmt='o',label='NG',alpha=0.5,markersize=3)
#plt.scatter(gal_age,age_full_mean,label='NG',alpha=0.5,s=3)
#plt.xlabel('true ages (Gyr)')
#plt.ylabel('predicted age')
#plt.xlim(0,14)
#plt.ylim(0,14)
#plt.title(f'Full run test 0.1')
#plt.savefig('model_v0_outfigs/model_test0.1_age_full.png')
#plt.errorbar(gal_age,age_med,yerr = [age_med-age_16,age_84-age_med],fmt='o',label='NG imperfect',alpha=0.5,markersize=3)
#plt.legend()
#plt.savefig('model_v0_outfigs/model_test0.1_age.png')
#plt.close()


sig_off_stel = np.abs((stel_full_mean-gal_stel)/stel_full_std)
sig_off_stel_noise = process_error(gal_stel,stel_med,stel_16,stel_84)

sig_off_dust = np.abs((dust_full_mean-gal_dust)/dust_full_std)
sig_off_dust_noise = process_error(gal_dust,dust_med,dust_16,dust_84)

sig_off_sfr = np.abs((sfr_full_mean-gal_sfr)/sfr_full_std)
sig_off_sfr_noise = process_error(gal_sfr,sfr_med,sfr_16,sfr_84)

#sig_off_metal = np.abs((metal_full_mean-gal_metal)/metal_full_std)
#sig_off_metal_noise = process_error(gal_metal,metal_med,metal_16,metal_84)

#sig_off_age = np.abs((age_full_mean-gal_age)/age_full_std)
#sig_off_age_noise = process_error(gal_age,age_med,age_16,age_84)



#print()
#print(gal_stel[:5])
#print()
#print(stel_full_mean[:5])
#print(stel_full_std[:5])
#print(sig_off_stel[:5])
#print()
#print(stel_med[:5])
#print((stel_med-stel_16)[:5])
#print((stel_84-stel_med)[:5])
#print(sig_off_noise[:5])


#t1 = np.array(range(10))
#true_te = np.repeat([5.0],10)
#lowt = t1-1.0
#hight = t1+0.5
#print()
#res_test = process_error(true_te,t1,lowt,hight)
#print(t1)
#print(true_te)
#print(lowt)
#print(hight)
#print(res_test)
#process_error(data_val,pred_med,pred16,pred84)
bins = np.arange(0,10,0.5)
plt.hist(sig_off_stel,bins=bins,density = True,cumulative=True,label='full',alpha=0.5)
plt.hist(sig_off_stel_noise,bins=bins,density = True,cumulative=True,label='imperfect',alpha=0.5)
plt.legend()
plt.xlim(right=4)
plt.xlabel('sigma of stel prediction from true value')
plt.savefig('model_v0_outfigs/error_test_stel_0.1.png')
plt.close()


plt.hist(sig_off_dust,bins=bins,density = True,cumulative=True,label='full',alpha=0.5)
plt.hist(sig_off_dust_noise,bins=bins,density = True,cumulative=True,label='imperfect',alpha=0.5)
plt.legend()
plt.xlim(right=4)
plt.xlabel('sigma of dust prediction from true value')
plt.savefig('model_v0_outfigs/error_test_dust_0.1.png')
plt.close()



plt.hist(sig_off_sfr,bins=bins,density = True,cumulative=True,label='full',alpha=0.5)
plt.hist(sig_off_sfr_noise,bins=bins,density = True,cumulative=True,label='imperfect',alpha=0.5)
plt.legend()
plt.xlim(right=4)
plt.xlabel('sigma of sfr prediction from true value')
plt.savefig('model_v0_outfigs/error_test_sfr_0.1.png')
plt.close()


#plt.hist(sig_off_metal,bins=bins,density = True,cumulative=True,label='full',alpha=0.5)
#plt.hist(sig_off_metal_noise,bins=bins,density = True,cumulative=True,label='imperfect',alpha=0.5)
#plt.legend()
#plt.xlim(right=4)
#plt.xlabel('sigma of metal prediction from true value')
#plt.savefig('model_v0_outfigs/error_test_metal_0.1.png')
#plt.close()

##

#plt.hist(sig_off_age,bins=bins,density = True,cumulative=True,label='full',alpha=0.5)
#plt.hist(sig_off_age_noise,bins=bins,density = True,cumulative=True,label='imperfect',alpha=0.5)
#plt.legend()
#plt.xlim(right=4)
#plt.xlabel('sigma of age prediction from true value')
#plt.savefig('model_v0_outfigs/error_test_age_0.1.png')
#plt.close()




