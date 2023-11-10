import numpy as np
import scipy as sp
import sys,os
from glob import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors


from hyperion.model import ModelOutput
import sedpy
from sedpy.observate import load_filters
import yt
import caesar

from astropy.cosmology import Planck13
from astropy import units as u
from astropy import constants
import h5py, re

import filter_helper

H0 = float(sys.argv[1])
snap_str = sys.argv[2]

sed_directory = f'/orange/narayanan/d.zimmerman/simba/m25n512/paper1_results/snap{snap_str}/pd_runs/'
caesar_loc = f"/orange/narayanan/d.zimmerman/simba/m25n512/paper1_results/caesar_cats/caesar_simba_{snap_str}.hdf5"
caes_file = caesar.load(caesar_loc)
z = caes_file.simulation.redshift
print(z)
outfile = '/orange/narayanan/d.zimmerman/simba/ml_data/all_filter_simba_snap'+snap_str+'_m25n512.npz'

save_figs = True
save_figs_count = 10


phot_list = []
sed_list = []

gal_labels = []

good_count = 0

#z_override = 0
z_val = [0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

colormap = mpl.colors.Normalize(vmin=-1,vmax=3)
m = mpl.cm.ScalarMappable(norm=colormap,cmap='inferno_r')#'

for i in range(len(caes_file.galaxies)):
    galaxy_num = i
    print(f'galaxy = {galaxy_num}')
    try:
        comp_sed = ModelOutput(sed_directory+f'/snap{snap_str}.galaxy{galaxy_num}.rtout.sed')
        wav_rest_sed,lum_obs_sed = comp_sed.get_sed(inclination=0,aperture=-1)
    except:
        print('failed galaxy:',galaxy_num)
        continue
    gal_labels.append(galaxy_num)
    good_count += 1
    wav_sed  = np.asarray(wav_rest_sed)[::-1]*u.micron #wav is in micron

    flux = np.asarray(lum_obs_sed)[::-1]*u.erg/u.s

    phot = []
    lam_a = []
    flux_converted = []

    for zv in z_val:
        lam_filt,ph,la,fc,filt_name = filter_helper.photo_calc(wav_sed,flux,z = zv,H0 = H0,txtfile = 'default')
        phot.append(ph)
        lam_a.append(la)
        flux_converted.append(fc)

    if(save_figs and good_count < save_figs_count):
        for j in range(1,len(lam_a)):
            plt.plot(lam_a[j],flux_converted[j],label = f"{z_val[j]}",c = m.to_rgba(z_val[j]))
            plt.scatter(lam_filt,phot[j],alpha=0.5,c = m.to_rgba(z_val[j]))
        plt.xscale('log')
        plt.yscale('log')
        #plt.legend()
        plt.savefig(f'outspec/test_spectrum{i}.png')
        plt.close()

    phot_list.append(phot)
    sed_list.append(flux_converted)

s_masses = np.array([caes_file.galaxies[g].masses['stellar'].to('Msun').value for g in gal_labels])
d_masses = np.array([caes_file.galaxies[g].masses['dust'].to('Msun').value for g in gal_labels])
metal = np.array([caes_file.galaxies[g].metallicities['stellar'].value for g in gal_labels])
sfr = np.array([caes_file.galaxies[g].sfr_100.to('Msun/yr').value for g in gal_labels])
ages = np.array([caes_file.galaxies[g].ages['mass_weighted'].to('Gyr').value for g in gal_labels])
print()
print(good_count)
print(np.array(s_masses))
#print(np.array(d_masses))
#print(np.array(metal))
#print(np.array(sfr))
print(lam_filt)
np.savez(outfile,wav_filt=lam_filt,wav_sed = lam_a,phot_data = np.array(phot_list),sed_data = np.array(sed_list),gal_num=np.array(gal_labels),stellar_mass=s_masses,dust_mass=d_masses,metallicity=metal,sfr=sfr,ages=ages,z=np.array(z_val),filter_names = filt_name)
