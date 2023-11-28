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
test_z = np.load('test_data_z.npy')
iterations = 100#0
snr = 5

def get_model_info():
    xscaler = joblib.load('model_dir/scaler_v0.1.joblib')
    zscaler = joblib.load('model_dir/zscaler_v0.1.joblib')
    cscaler = joblib.load('model_dir/cscaler_v0.1.joblib')
    imputer = joblib.load('model_dir/knn_v0.1.joblib')
    model_stel = joblib.load('model_dir/stel_ng_v0.1.joblib')
    model_dust = joblib.load('model_dir/dust_ng_v0.1.joblib')
    model_sfr = joblib.load('model_dir/sfr_ng_v0.1.joblib')
    model_metal = joblib.load('model_dir/metal_ng_v0.1.joblib')
    model_age = joblib.load('model_dir/age_ng_v0.1.joblib')
    model_z = joblib.load('model_dir/z_ng_v0.1.joblib')
    return xscaler,zscaler,cscaler,imputer,model_stel,model_dust,model_sfr,model_metal,model_age,model_z

def eval_model_with_z(model,imputed_photometry,photo_z):

    pred_imp = model.pred_dist(imputed_photometry)
    mean_imp = pred_imp.params['loc']
    std_imp = pred_imp.params['scale']
    res_imp = np.random.normal(loc=mean_imp,scale=std_imp)
    
    photo_z_scaled = zscaler.transform(photo_z.reshape(-1,1))
    x_photo = np.concatenate((imputed_photometry[:,:-1],photo_z_scaled),axis = 1)
    
    pred_photz = model.pred_dist(x_photo)
    mean_photz = pred_imp.params['loc']
    std_photz = pred_imp.params['scale']
    res_photz = np.random.normal(loc=mean_photz,scale=std_photz)

    return res_imp,res_photz


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
    return err_dev

xscaler,zscaler,cscaler,imputer,model_stel,model_dust,model_sfr,model_metal,model_age,model_z = get_model_info()


test_x_scaled = np.log10(test_x)
test_x_scaled[np.isnan(test_x_scaled) | (test_x_scaled < -50)] = -20
test_x_scaled = xscaler.transform(test_x_scaled)

test_z_scaled = zscaler.transform(test_z.reshape(-1,1))

test_x_combined = np.concatenate((test_x_scaled,test_z_scaled),axis = 1)

stel_full = model_stel.pred_dist(test_x_combined)
stel_full_mean = stel_full.params['loc']
stel_full_std = stel_full.params['scale']

dust_full = model_dust.pred_dist(test_x_combined)
dust_full_mean = dust_full.params['loc']
dust_full_std = dust_full.params['scale']

sfr_full = model_sfr.pred_dist(test_x_combined)
sfr_full_mean = sfr_full.params['loc']
sfr_full_std = sfr_full.params['scale']

metal_full = model_metal.pred_dist(test_x_combined)
metal_full_mean = metal_full.params['loc']
metal_full_std = metal_full.params['scale']

age_full = model_age.pred_dist(test_x_combined)
age_full_mean = age_full.params['loc']
age_full_std = age_full.params['scale']

z_full = model_z.pred_dist(test_x_combined)
z_full_mean = z_full.params['loc']
z_full_std = z_full.params['loc']


# start copy

filter_count = np.random.randint(len(test_x_combined[0])-6,len(test_x_combined[0])-1,size = len(test_x_combined))

filter_arr = np.full(np.shape(test_x_combined),1)
for i in range(len(filter_count)):
    count = filter_count[i]
    perm = np.random.permutation(len(test_x_combined[0])-1)
    perm = perm[:count]
    filter_arr[i,perm] = 0
    filter_arr[i,-1] = 0
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
gal_z = test_z

z_imp = []
z_photz = []

data_imputed = []
stel_mass_impz = []
dust_mass_impz = []
sfr_impz = []
metal_impz = []
age_impz = []

stel_mass_photz = []
dust_mass_photz = []
sfr_photz = []
metal_photz = []
age_photz = []

for i in range(iterations):
    phot_noise = gal_phot*np.random.normal(loc=1.0,scale = 1.0/snr,size=np.shape(gal_phot))
    phot_scaled = np.log10(phot_noise)
    phot_scaled[(phot_scaled < -50)] = -20
    phot_scaled = xscaler.transform(phot_scaled)
    phot_trimmed = np.concatenate((phot_scaled,np.full((len(phot_scaled),1),np.nan)),axis = 1)
    #print(np.shape(phot_trimmed))
    phot_trimmed[filter_remove] = np.nan
    #print(phot_trimmed)
    #raise Exception()
    #print('impute')
    imputed_phot = imputer.transform(phot_trimmed)
    data_imputed.append(xscaler.inverse_transform(imputed_phot[:,:-1]))
    
    zi = zscaler.inverse_transform(imputed_phot[:,-1].reshape(-1,1))

    z_dist = model_z.pred_dist(imputed_phot[:,:-1])
    zp = np.random.normal(loc=z_dist.params['loc'],scale=z_dist.params['scale'])

    z_imp.append(np.ravel(zi))
    z_photz.append(zp)
    
    
    
    #print('imputed')
    #print('eval')
    stel_imp,stel_photz = eval_model_with_z(model_stel,imputed_phot,zp)
    stel_mass_impz.append(stel_imp)
    stel_mass_photz.append(stel_photz)
    
    dust_imp,dust_photz = eval_model_with_z(model_dust,imputed_phot,zp)
    dust_mass_impz.append(dust_imp)
    dust_mass_photz.append(dust_photz)

    sfr_imp,sfr_phot = eval_model_with_z(model_sfr,imputed_phot,zp)
    sfr_impz.append(sfr_imp)
    sfr_photz.append(sfr_phot)

    metal_imp,metal_phot = eval_model_with_z(model_metal,imputed_phot,zp)
    metal_impz.append(metal_imp)
    metal_photz.append(metal_phot)

    age_imp,age_phot = eval_model_with_z(model_age,imputed_phot,zp)
    age_impz.append(age_imp)
    age_photz.append(age_phot)

    #print('evaluated')

    #if(i%50==0):
    print(i)

print(np.shape(stel_mass_impz))
print(np.shape(data_imputed))

stel_med_impz = np.median(stel_mass_impz,axis=0)
stel_16_impz = np.quantile(stel_mass_impz,0.16,axis=0)
stel_84_impz = np.quantile(stel_mass_impz,0.84,axis=0)

stel_med_photz = np.median(stel_mass_photz,axis=0)
stel_16_photz = np.quantile(stel_mass_photz,0.16,axis=0)
stel_84_photz = np.quantile(stel_mass_photz,0.84,axis=0)

#print(np.shape(stel_med))
#print(np.shape(stel_16))

dust_med_impz = np.median(dust_mass_impz,axis=0)
dust_16_impz = np.quantile(dust_mass_impz,0.16,axis=0)
dust_84_impz = np.quantile(dust_mass_impz,0.84,axis=0)

dust_med_photz = np.median(dust_mass_photz,axis=0)
dust_16_photz = np.quantile(dust_mass_photz,0.16,axis=0)
dust_84_photz = np.quantile(dust_mass_photz,0.84,axis=0)

sfr_med_impz = np.median(sfr_impz,axis=0)
sfr_16_impz = np.quantile(sfr_impz,0.16,axis=0)
sfr_84_impz = np.quantile(sfr_impz,0.84,axis=0)

sfr_med_photz = np.median(sfr_photz,axis=0)
sfr_16_photz = np.quantile(sfr_photz,0.16,axis=0)
sfr_84_photz = np.quantile(sfr_photz,0.84,axis=0)

metal_med_impz = np.median(metal_impz,axis=0)
metal_16_impz = np.quantile(metal_impz,0.16,axis=0)
metal_84_impz = np.quantile(metal_impz,0.84,axis=0)

metal_med_photz = np.median(metal_photz,axis=0)
metal_16_photz = np.quantile(metal_photz,0.16,axis=0)
metal_84_photz = np.quantile(metal_photz,0.84,axis=0)

age_med_impz = np.median(age_impz,axis=0)
age_16_impz = np.quantile(age_impz,0.16,axis=0)
age_84_impz = np.quantile(age_impz,0.84,axis=0)

age_med_photz = np.median(age_photz,axis=0)
age_16_photz = np.quantile(age_photz,0.16,axis=0)
age_84_photz = np.quantile(age_photz,0.84,axis=0)

z_med_imp = np.median(z_imp,axis=0)
z_16_imp = np.quantile(z_imp,0.16,axis=0)
z_84_imp = np.quantile(z_imp,0.84,axis=0)

z_med_photz = np.median(z_photz,axis=0)
z_16_photz = np.quantile(z_photz,0.16,axis=0)
z_84_photz = np.quantile(z_photz,0.84,axis=0)

phot_med = np.median(data_imputed,axis=0)
phot_16 = np.quantile(data_imputed,0.16,axis=0)
phot_84 = np.quantile(data_imputed,0.84,axis=0)
#print(np.shape(phot_med))
#print(np.shape(phot_16))

# end copy


oto = np.linspace(7,12.5,100)
plt.plot(oto,oto,linestyle = '--',color = 'black')
plt.errorbar(gal_stel,stel_full_mean,yerr = stel_full_std,fmt='o',label='NG',alpha=0.1,markersize=3)
plt.xlabel('log true M*')
plt.ylabel('log predicted M*')
plt.xlim(7.2,12.5)
plt.ylim(7.2,12.5)
plt.title(f'Full run test 0.2')
plt.savefig('model_v0_outfigs/model_test0.2_stel_full.png')
plt.errorbar(gal_stel,stel_med_impz,yerr = [stel_med_impz-stel_16_impz,stel_84_impz-stel_med_impz],fmt='o',label='NG imperfect imputed',alpha=0.1,markersize=3)
plt.errorbar(gal_stel,stel_med_photz,yerr = [stel_med_photz-stel_16_photz,stel_84_photz-stel_med_photz],fmt='o',label='NG imperfect photz',alpha=0.1,markersize=3)
plt.legend()
plt.savefig('model_v0_outfigs/model_test0.2_stel.png')
plt.close()




oto = np.linspace(2,9,100)
plt.plot(oto,oto,linestyle = '--',color = 'black')
plt.errorbar(gal_dust,dust_full_mean,yerr = dust_full_std,fmt='o',label='NG',alpha=0.1,markersize=3)
plt.xlabel('log true M_dust')
plt.ylabel('log predicted M_dust')
plt.xlim(2,9)
plt.ylim(2,9)
plt.title(f'Full run test 0.2')
plt.savefig('model_v0_outfigs/model_test0.2_dust_full.png')
plt.errorbar(gal_dust,dust_med_impz,yerr = [dust_med_impz-dust_16_impz,dust_84_impz-dust_med_impz],fmt='o',label='NG imperfect imputed',alpha=0.1,markersize=3)
plt.errorbar(gal_dust,dust_med_photz,yerr = [dust_med_photz-dust_16_photz,dust_84_photz-dust_med_photz],fmt='o',label='NG imperfect photz',alpha=0.1,markersize=3)
plt.legend()
plt.savefig('model_v0_outfigs/model_test0.2_dust.png')
plt.close()


oto = np.linspace(-0.1,2.5,100)
plt.plot(oto,oto,linestyle = '--',color = 'black')
plt.errorbar(gal_sfr,sfr_full_mean,yerr = sfr_full_std,fmt='o',label='NG',alpha=0.1,markersize=3)
plt.xlabel('true log (sfr+1)')
plt.ylabel('predicted log (sfr+1)')
plt.xlim(-0.1,1.2)
plt.ylim(-0.1,1.2)
plt.title(f'Full run test 0.2')
plt.savefig('model_v0_outfigs/model_test0.2_sfr_full.png')
plt.errorbar(gal_sfr,sfr_med_impz,yerr = [sfr_med_impz-sfr_16_impz,sfr_84_impz-sfr_med_impz],fmt='o',label='NG imperfect imputed',alpha=0.1,markersize=3)
plt.errorbar(gal_sfr,sfr_med_photz,yerr = [sfr_med_photz-sfr_16_photz,sfr_84_photz-sfr_med_photz],fmt='o',label='NG imperfect photz',alpha=0.1,markersize=3)
plt.legend()
plt.savefig('model_v0_outfigs/model_test0.2_sfr.png')
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


#oto = np.linspace(0,2.5,100)
#plt.plot(oto,oto,linestyle = '--',color = 'black')
#plt.errorbar(gal_z,z_full_mean,yerr = z_full_std,fmt='o',label='NG',alpha=0.5,markersize=3)
#plt.scatter(gal_z,z_full_mean,label='NG',alpha=0.5,s=3)
hist_bins = np.arange(-2.5,2.5,0.1)
plt.hist(z_full_mean - gal_z,bins = hist_bins,alpha=0.5,label='NG')
plt.xlabel('predicted-true z')
plt.xlim(-2.1,2.1)
plt.title(f'Full run test 0.2')
plt.savefig('model_v0_outfigs/model_test0.2_z_full.png')
plt.hist(z_med_imp-gal_z,bins = hist_bins,alpha=0.5,label='NG imperfect imputed')
plt.hist(z_med_photz-gal_z,bins = hist_bins,alpha=0.5,label='NG imperfect photz')
        #gal_z,z_med,yerr = [z_med-z_16,age_84-z_med],fmt='o',label='NG imperfect',alpha=0.5,markersize=3)
plt.legend()
plt.savefig('model_v0_outfigs/model_test0.2_z.png')
plt.close()


sig_off_stel = (stel_full_mean-gal_stel)/stel_full_std
sig_off_stel_noise_impz = process_error(gal_stel,stel_med_impz,stel_16_impz,stel_84_impz)
sig_off_stel_noise_photz = process_error(gal_stel,stel_med_photz,stel_16_photz,stel_84_photz)

sig_off_dust = (dust_full_mean-gal_dust)/dust_full_std
sig_off_dust_noise_impz = process_error(gal_dust,dust_med_impz,dust_16_impz,dust_84_impz)
sig_off_dust_noise_photz = process_error(gal_dust,dust_med_photz,dust_16_photz,dust_84_photz)


sig_off_sfr = (sfr_full_mean-gal_sfr)/sfr_full_std
sig_off_sfr_noise_impz = process_error(gal_sfr,sfr_med_impz,sfr_16_impz,sfr_84_impz)
sig_off_sfr_noise_photz = process_error(gal_sfr,sfr_med_photz,sfr_16_photz,sfr_84_photz)

#sig_off_metal = np.abs((metal_full_mean-gal_metal)/metal_full_std)
#sig_off_metal_noise = process_error(gal_metal,metal_med,metal_16,metal_84)

#sig_off_age = np.abs((age_full_mean-gal_age)/age_full_std)
#sig_off_age_noise = process_error(gal_age,age_med,age_16,age_84)



sig_off_z = (z_full_mean-gal_z)/z_full_std
sig_off_z_noise_impz = process_error(gal_z,z_med_imp,z_16_imp,z_84_imp)
sig_off_z_noise_photz = process_error(gal_z,z_med_photz,z_16_photz,z_84_photz)

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

# first absolute histogram, then error scaled histogram, then cumulative error histogram
def hists_maker(prop_name,save_name,truth,mean_all,sig_off_all,imp_mean,imp_sig_off,photz_mean,photz_sig_off):
    bins1 = np.arange(-3,3,0.1)
    plt.hist(mean_all-truth,bins = bins1,label = 'full',alpha = 0.3)
    plt.hist(imp_mean-truth,bins = bins1,label = 'imperfect imputed z',alpha=0.3)
    plt.hist(photz_mean-truth,bins = bins1,label = 'imperfect phot z',alpha = 0.3)
    plt.legend()
    plt.xlabel(f'predicted {prop_name} - true {prop_name}')
    plt.yscale('log')
    plt.savefig(f'model_v0_outfigs/acc_test_{save_name}_0.2.png')
    plt.close()

    bins2 = np.arange(-5,5,0.1)
    plt.hist(sig_off_all,bins = bins2,label = 'full',alpha = 0.3)
    plt.hist(imp_sig_off,bins = bins2,label = 'imperfect imputed z',alpha=0.3)
    plt.hist(photz_sig_off,bins = bins2,label = 'imperfect phot z',alpha = 0.3)
    plt.legend()
    plt.xlabel(f'predicted {prop_name} - true {prop_name}/uncertainty')
    plt.yscale('log')
    plt.savefig(f'model_v0_outfigs/error_test1_{save_name}_0.2.png')
    plt.close()

    bins3 = np.arange(0,5,0.1)
    plt.hist(np.abs(sig_off_all),bins = bins3,density = True,cumulative=True,label = 'full',alpha = 0.3)
    plt.hist(np.abs(imp_sig_off),bins = bins3,density = True,cumulative=True,label = 'imperfect imputed z',alpha=0.3)
    plt.hist(np.abs(photz_sig_off),bins = bins3,density = True,cumulative=True,label = 'imperfect phot z',alpha = 0.3)
    plt.legend()
    plt.xlabel(f'cumulative {prop_name} offset plot')
    plt.savefig(f'model_v0_outfigs/error_test2_{save_name}_0.2.png')
    plt.close()


hists_maker('log(M*)','stel',gal_stel,stel_full_mean,sig_off_stel,stel_med_impz,sig_off_stel_noise_impz,stel_med_photz,sig_off_stel_noise_photz)
hists_maker('log(Mdust)','dust',gal_dust,dust_full_mean,sig_off_dust,dust_med_impz,sig_off_dust_noise_impz,dust_med_photz,sig_off_dust_noise_photz)
hists_maker('log(sfr+1)','sfr',gal_sfr,sfr_full_mean,sig_off_sfr,sfr_med_impz,sig_off_sfr_noise_impz,sfr_med_photz,sig_off_sfr_noise_photz)
hists_maker('z','z',gal_z,z_full_mean,sig_off_z,z_med_imp,sig_off_z_noise_impz,z_med_photz,sig_off_z_noise_photz)
