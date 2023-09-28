import numpy as np
import sedpy
import scipy
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import math

def photo_calc(lam,lumlam,z = 0,H0 = 70,txtfile = 'default'):
    cosmo = FlatLambdaCDM(Om0=0.3,Tcmb = 2.725,H0 = H0)
    if(z<10**-4):
        dl = 10*u.pc
    else:
        dl = cosmo.luminositydistance(z).to('pc')
    lam_a = lam.to(u.AA)
    lumlam_a = lumlam.to(u.Lsun/u.AA)
    
    filter_names = []
    if(txtfile == 'default'):
        with open('all_filters.txt') as f:
            for line in f:
                filter_names.append(line)
    else:
        with open(txtfile) as f:
            for line in f:
                filter_names.append(line)

    lam_a = lam_a*(1+z)
    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]


    flam = lumlam_a/(4.*math.pi*dl**2.)

    #nu = constants.c.cgs/(wav_sed.to(u.cm))
    #nu = nu.to(u.Hz)

    flux /= wav_sed
    flux = flux.to(u.erg/u.s/u.cm/u.cm/u.micron)
    #flux = flux.value
    #flux = flux.to(u.mJy)

    #print(wav_sed)


    filters_unsorted = load_filters(filternames)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]
    gal_phot = []

    for j in range(len(filters)):
            flux_range = []
            #wav_range = []
            filt_lambda = filters[j].wavelength*u.AA
            filt_lambda = filt_lambda.to(u.micron)
            for k in filt_lambda:
                   flux_range.append(flux[find_nearest(wav_sed,k)].value)
                   #wav_range.append(wave[find_nearest(wav_sed,k)])
            #print(filt_lambda)
            #print(filters[j].transmission)
            #print(flux_range)
            a = np.trapz(filt_lambda.value * filters[j].transmission * flux_range, filt_lambda.value)
            b = np.trapz(filt_lambda.value * filters[j].transmission, filt_lambda.value)
            ratio = a/b #u.erg/u.s/(u.cm**2)/u.micron
            gal_phot.append(ratio)# .to(u.erg/u.s/(u.cm**2)))
    #print(wav_rest_sed[(wav_rest_sed < np.max(filters[0].wavelength*10**-4)) & (wav_rest_sed > np.min(filters[0].wavelength*10**-4))])
