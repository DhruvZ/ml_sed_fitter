import numpy as np
import sedpy
import scipy
from astropy.cosmology import FlatLambdaCDM
from astropy import constants
import astropy.units as u
import math


def find_nearest(array,value):
    idx = (np.abs(np.array(array)-value)).argmin()
    return idx

def photo_calc(lam,lumtlam,z = 0,H0 = 70,txtfile = 'default'):
    cosmo = FlatLambdaCDM(Om0=0.3,Tcmb0 = 2.725,H0 = H0)
    if(z<10**-4):
        dl = (10*u.pc).to(u.cm)

    else:
        dl = cosmo.luminositydistance(z).to(u.cm)
    lam_a = lam.to(u.AA)
    lumtlam = lumtlam.to(u.erg/u.s)
    
    filter_names = []
    if(txtfile == 'default'):
        with open('all_filters.txt') as f:
            for line in f:
                filter_names.append(line.rstrip())
    else:
        with open(txtfile) as f:
            for line in f:
                filter_names.append(line.rstrip())

    flux = lumtlam/(4.*math.pi*(dl)**2.)

    nu = constants.c.cgs/(lam_a.to(u.cm))
    nu = nu.to(u.Hz)

    flux /= nu
    flux = flux.to(u.Jy)

    #print(wav_sed)

    lam_a = lam_a*(1+z)
    #print(filter_names)
    filters_unsorted = sedpy.observate.load_filters(filter_names)
    waves_unsorted = [x.wave_mean for x in filters_unsorted]
    filters = [x for _,x in sorted(zip(waves_unsorted,filters_unsorted))]
    names_sorted = [y for _,y in sorted(zip(waves_unsorted,filter_names))]
    #print(filters)
    #print(names_sorted)
    #raise Exception()
    gal_phot = []

    waves_sorted = [x.wave_mean for x in filters]


    for j in range(len(filters)):
            flux_range = []
            #wav_range = []
            filt_lambda = filters[j].wavelength*u.AA
            for k in filt_lambda:
                   flux_range.append(flux[find_nearest(lam_a.value,k.value)].value)
                   #wav_range.append(wave[find_nearest(wav_sed,k)])
            #print(filt_lambda)
            #print(filters[j].transmission)
            #print(flux_range)
            a = np.trapz(filt_lambda.value * filters[j].transmission * flux_range, filt_lambda.value)
            b = np.trapz(filt_lambda.value * filters[j].transmission, filt_lambda.value)
            ratio = a/b #u.erg/u.s/(u.cm**2)/u.micron
            gal_phot.append(ratio)# .to(u.erg/u.s/(u.cm**2)))
    return waves_sorted,gal_phot,lam_a,flux,names_sorted
