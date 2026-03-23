#!/usr/bin/env python
#SBATCH --time=5:00:00 # walltime
#SBATCH --nodes=1 # number of nodes
#SBATCH --exclusive
#SBATCH -J "bosz_fit" # job name
#SBATCH --output=output.txt

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from PyAstronomy import pyasl
#import pysynphot as S

import re

import pandas as pd

from astropy.time import Time

#from astroquery.gaia import Gaia
from astroquery.utils.tap.core import TapPlus
from dust_extinction.parameter_averages import G23

import os

from tqdm import trange

from astropy import units as u, constants as c

import time

#import pyphot
#from pyphot import unit
#lib = pyphot.get_library()

from astropy.wcs import WCS

#from multiprocessing import Pool
from multiprocess import Pool

import emcee

from astropy.table import Table
from astropy.convolution import convolve, Box1DKernel

from joblib import Parallel, delayed

from scipy.signal import savgol_filter

import warnings
warnings.filterwarnings("ignore")


def median_norm(waves, fluxes, window_angs = 15, top_mean = 20, poly_deg = 6, smooth_wave = 1.0, do_smooth = False):
    
    select = abs(fluxes)>0
    wave = waves[select]
    flux = fluxes[select]
    
    if do_smooth:
        smooth_N = int(smooth_wave/np.mean(np.diff(wave)))
    else:
        smooth_N = 1.0
    kernel = Box1DKernel(smooth_N)
    flux_smoothed = convolve(flux, kernel)
    
    window = int(window_angs/np.mean(np.diff(wave)))
        
    norm_wave = np.zeros(len(wave)-window)
    norm_flux = np.zeros(len(wave)-window)
    
    top_values = []
    top_waves = []
    
    for i in range(int(window/2), int(len(wave)-window/2), int(window/4)):
        
        top_30 = np.partition(flux_smoothed[int(i-window/2):int(i+window/2)], -top_mean)[-top_mean:]
        median_cont_med_spec = np.median(top_30)
        
        top_waves.append(wave[i])
        top_values.append(median_cont_med_spec)
        
    top_waves = np.array(top_waves)
    top_values = np.array(top_values)
        
    '''
    norm_poly = np.polyfit((top_waves-np.mean(top_waves)),
                          (top_values-np.mean(top_values)),
                          deg = poly_deg)
    
    cont_vals = np.polyval(norm_poly, wave-np.mean(top_waves))+np.mean(top_values)
    '''
    
    cont_vals = np.interp(wave, top_waves, top_values)
    
    return wave, flux/cont_vals


def merge_orders(filename, wave_cut = 20, order_uplim = 25, wave_low = 4500.0, wave_up = 9000):
    
    hdul = fits.open(filename)
    
    wave = hdul[0].data[0][0].copy()
    normalized_flux = hdul[0].data[3][0].copy()
    
    resol = np.mean(np.diff(wave))
    num_index = int(wave_cut/resol)

    normalized_flux[:num_index] = 0.0
    normalized_flux[-num_index:] = 0.0
    
    wave, normalized_flux = median_norm(wave, normalized_flux, do_smooth=True)
    
    for i in range(1,order_uplim):
        
        wave_old = wave.copy()
        normalized_flux_old = normalized_flux.copy()
        
        wave_new = hdul[0].data[0][i].copy()
        wave = np.concatenate((wave_old, wave_new), axis = 0)
        
        wave_sort_ind = np.argsort(wave)
        wave = wave[wave_sort_ind]
                
        resol = np.mean(np.diff(wave_new))
        
        num_index = int(wave_cut/resol)
        
        normalized_flux_new = hdul[0].data[3][i].copy()
        normalized_flux_new[:num_index] = 0.0
        normalized_flux_new[-num_index:] = 0.0
        
        wave_new, normalized_flux_new = median_norm(wave_new, normalized_flux_new, do_smooth=True)
        
        flux_interp_old = np.interp(wave, wave_old, normalized_flux_old, left =0, right = 0)
        flux_interp_new = np.interp(wave, wave_new, normalized_flux_new, left =0, right = 0)
        
        normalize_factor = ((flux_interp_old!=0.0) + 0.0) + ((flux_interp_new!=0.0) + 0.0)
        normalize_factor[normalize_factor==0.0] = 1.0
        
        normalized_flux = (flux_interp_old + flux_interp_new)/normalize_factor
                
    select = (wave<wave_up) & (wave>wave_low)
            
    return wave[select], normalized_flux[select]




def find_template_vsini_rv(wave_obj, normalized_flux_obj, temps_in_grid,
                  wave_range = [6000, 6400], buffer = 50, rv_grid = np.arange(-300.0, 310, 10.0),
                 show = False, vsini_grid = np.arange(5, 100, 5.0), nf_grid = np.arange(0.95, 1.05, 500), 
                           model = 'coelho',
                          fix_temp = False, fix_logg = True, logg_in_grid = 3.5,
                          do_print = True, data_snr = 50.0,
                          limit_alpha = False, limit_vmicro = False,
                          savgol_window = 25):
    
    if model=='coelho':
        coelho_model_files = [os.path.join(root, file) for root, _, files in os.walk("/Volumes/T9/coelho_hires_models/models_1750207560/coelho_highres/") for file in files]

        star_model_files = []

        for coelho_model_file in coelho_model_files:
            if ('t0{}'.format(int(temp_in_grid)) in coelho_model_file) and ('._' not in coelho_model_file) and ('.DS' not in coelho_model_file):
                star_model_files.append(coelho_model_file)
                
    if model=='bosz':
        coelho_model_files = [os.path.join(root, file) for root, _, files in os.walk('./bosz_models_manual/') for file in files]

        star_model_files = []

        for temp_in_grid in temps_in_grid:
            for coelho_model_file in coelho_model_files:

                if (('._' in coelho_model_file) or ('.DS' in coelho_model_file) or ('.gz' not in coelho_model_file)):
                    continue

                match = re.search(r'_a([+-]?\d*\.?\d+)', coelho_model_file)
                alp = float(match.group(1))

                match = re.search(r'_v(\d+(?:\.\d+)?)', coelho_model_file)
                vmi = float(match.group(1))

                if fix_temp:
                    temp_cond = ('_t{}'.format(int(temp_in_grid)) in coelho_model_file)
                else:
                    temp_cond = ('_t{}'.format(int(temp_in_grid)) in coelho_model_file) or ('_t{}'.format(int(temp_in_grid-250)) in coelho_model_file) or ('_t{}'.format(int(temp_in_grid+250)) in coelho_model_file)
                if fix_logg:
                    logg_cond = ('_g+{}'.format((logg_in_grid)) in coelho_model_file)
                else:
                    logg_cond = True
                if limit_vmicro:
                    vmic_cond = vmi<=2.0
                else:
                    vmic_cond = True
                if limit_alpha:
                    alpha_cond = alp<=0.25
                else:
                    alpha_cond = True
                if temp_cond and logg_cond and alpha_cond and vmic_cond and ('_c+0.00' in coelho_model_file):
                    star_model_files.append(coelho_model_file)
                    if do_print:
                        print(coelho_model_file)
                
        bosz_wave = pd.read_csv('./bosz_models_manual/r50000/bosz2024_wave_r50000.txt', header = None)[0].values
        select_bosz_wave = (bosz_wave<(wave_range[1]+buffer))&(bosz_wave>(wave_range[0]-buffer))
        coelho_wave_pre_resampled = bosz_wave[select_bosz_wave]

    template_nos = []
    
    diff_rms_grid = np.zeros((len(star_model_files), len(vsini_grid), len(rv_grid), len(nf_grid)))

    select_obj = (wave_obj<wave_range[1])&(wave_obj>wave_range[0])
    wave_obj_select = wave_obj[select_obj]
    normalized_flux_obj_select = normalized_flux_obj[select_obj]

    print(len(star_model_files))

    for star_model_filei in range(len(star_model_files)):

        template_nos.append(star_model_filei)

        star_model_file = star_model_files[star_model_filei]

        if model=='coelho':
            coelho = pd.read_csv(star_model_file,
                                 skiprows = 8, header = None, delim_whitespace = True)

            coelho = coelho[(coelho[0]<wave_range[1]+buffer)&(coelho[0]>wave_range[0]-buffer)]

            coelho_wave_pre_resampled = coelho[0].values
            coelho_flux_pre_resampled = coelho[1].values
            
        if model=='bosz':
            coelho_flux_pre_resampled = pd.read_csv(star_model_file, header = None,
                  delim_whitespace = True)[0].values[select_bosz_wave]

        ###### vsini below #########
        
        wave_res = np.min(np.diff(coelho_wave_pre_resampled))
        coelho_wave = np.arange(coelho_wave_pre_resampled[0], coelho_wave_pre_resampled[-1]+wave_res,
                               wave_res)
                
        coelho_flux = np.interp(coelho_wave, coelho_wave_pre_resampled, coelho_flux_pre_resampled)
        
        for vsinii in range(len(vsini_grid)):
            
            vsini = vsini_grid[vsinii]
                
            coelho_flux_broadened = pyasl.rotBroad(coelho_wave, coelho_flux, 0.4, vsini)
            
            coelho_wave_norm, coelho_flux_norm = median_norm(coelho_wave, coelho_flux_broadened, 
                                                            do_smooth=False)
        
        ############################
        
            for rvi in range(len(rv_grid)):

                rv = rv_grid[rvi]

                coelho_wave_norm_shifted = coelho_wave_norm + rv*coelho_wave_norm/3e5

                coelho_data_interp = np.interp(wave_obj_select, coelho_wave_norm_shifted, coelho_flux_norm)

                select_abs = normalized_flux_obj_select<100.0
                
                for nfi in range(len(nf_grid)):
                    
                    nf = nf_grid[nfi]

                    
                    
                    flux_obj_smooth = savgol_filter(normalized_flux_obj_select, window_length=savgol_window, 
                                        polyorder=2)
        
                    residuals = normalized_flux_obj_select - flux_obj_smooth
                    
                    res = pd.Series(residuals)

                    sigma_x = res.rolling(
                        window=savgol_window,
                        center=True,
                        min_periods=savgol_window//2
                    ).std()
                    sigma2_arr = sigma_x**2
                    

                    #sigma2_arr = normalized_flux_obj_select/100.0

                    diff_rms = np.sum(((coelho_data_interp[select_abs]*nf - normalized_flux_obj_select[select_abs])**2)/(sigma2_arr[select_abs]))

                    diff_rms_grid[star_model_filei][vsinii][rvi][nfi] = diff_rms

    weight_grid = np.exp((np.min(diff_rms_grid) - diff_rms_grid)/2.0)
    weight_sum = np.sum(weight_grid)
                
    min_index = np.unravel_index(np.argmin(diff_rms_grid), diff_rms_grid.shape)
    
    rv = rv_grid[min_index[2]]
    star_model_file_min = star_model_files[min_index[0]]
    vsini = vsini_grid[min_index[1]]
    nf = nf_grid[min_index[3]]
    
    match = re.search(r'_m([+-]?\d*\.?\d+)', star_model_file_min)
    met = float(match.group(1))
    
    match = re.search(r'_t([+-]?\d*\.?\d+)', star_model_file_min)
    tem = float(match.group(1))

    match = re.search(r'_a([+-]?\d*\.?\d+)', star_model_file_min)
    alp = float(match.group(1))

    match = re.search(r'_c([+-]?\d*\.?\d+)', star_model_file_min)
    car = float(match.group(1))

    met_mean = 0.0
    tem_mean = 0.0
    alp_mean = 0.0
    car_mean = 0.0
    vmi_mean = 0.0

    vsini_mean = 0.0
    rv_mean = 0.0

    star_best_chi2s = []
    star_best_tems = []
    star_best_mets = []

    for star_model_filei in range(len(star_model_files)):
        chi2min = 1e10
        for vsinii in range(len(vsini_grid)):
            for rvi in range(len(rv_grid)):
                for nfi in range(len(nf_grid)):
                    star_model_file = star_model_files[star_model_filei]

                    match = re.search(r'_m([+-]?\d*\.?\d+)', star_model_file)
                    met = float(match.group(1))
                    
                    match = re.search(r'_t([+-]?\d*\.?\d+)', star_model_file)
                    tem = float(match.group(1))

                    match = re.search(r'_a([+-]?\d*\.?\d+)', star_model_file)
                    alp = float(match.group(1))

                    match = re.search(r'_c([+-]?\d*\.?\d+)', star_model_file)
                    car = float(match.group(1))

                    match = re.search(r'_v(\d+(?:\.\d+)?)', star_model_file)
                    vmi = float(match.group(1))

                    if diff_rms_grid[star_model_filei][vsinii][rvi][nfi]<chi2min:
                        best_met = met
                        best_tem = tem
                        chi2min = diff_rms_grid[star_model_filei][vsinii][rvi][nfi]
                        best_chi2 = chi2min

                    weight = weight_grid[star_model_filei][vsinii][rvi][nfi]

                    met_mean = met_mean + met * weight/weight_sum
                    tem_mean = tem_mean + tem * weight/weight_sum
                    alp_mean = alp_mean + alp * weight/weight_sum
                    car_mean = car_mean + car * weight/weight_sum
                    vmi_mean = vmi_mean + vmi * weight/weight_sum

                    vsini_mean = vsini_mean + vsini_grid[vsinii] * weight/weight_sum
                    rv_mean = rv_mean + rv_grid[rvi] * weight/weight_sum

        star_best_chi2s.append(best_chi2)
        star_best_tems.append(best_tem)
        star_best_mets.append(best_met)

    star_best_chi2s = np.array(star_best_chi2s)
    star_best_tems = np.array(star_best_tems)
    star_best_mets = np.array(star_best_mets)

    topN_idx = np.argsort(star_best_chi2s)[:10]
    topN_met_mean = np.mean(star_best_mets[topN_idx])
    topN_met_std = np.std(star_best_mets[topN_idx])
    if topN_met_std==0:
        topN_met_std = 1e-10


    
    if do_print:
        print('Model File: ',star_model_file)
        print('RV (km/s): ',rv)
        print('vsini (km/s): ',vsini)
        print('norm fact: ',nf)
        print('metallicity: ',met)
        print('temperature: ',tem)
    
    return diff_rms_grid, star_model_file_min, rv_mean, vsini_mean, met_mean, tem_mean, alp_mean, car_mean, vmi_mean, topN_met_mean, topN_met_std

      

GAIA_TAP_URL = 'https://gea.esac.esa.int/tap-server/tap'
gaia = TapPlus(url=GAIA_TAP_URL)

def search_gaia(desgn):

    query = f"""
    SELECT *
    FROM gaiadr3.gaia_source
    WHERE designation = '{desgn}'
    AND parallax IS NOT NULL
    """

    #Gaia.ROW_LIMIT = -1

    gaia = TapPlus(url=GAIA_TAP_URL)
    job = gaia.launch_job_async(query)
    result = job.get_results()
        
    bprp = result['bp_rp'].value
    Teff = 10**(3.999-0.654*(bprp)+0.709*(bprp**2)-0.316*(bprp**3))
    
    teff_avg = (result['teff_gspphot'].value[0] + Teff[0])/2.0
    
    print('Teff: ',Teff[0])
    print('teff_gspphot: ',result['teff_gspphot'].value[0])
    print('Teff avg: ',teff_avg)
    print('Ra, Dec: ',result['ra'].value[0], result['dec'].value[0])
    print('PMRa, PMDec: ',result['pmra'].value[0], result['pmdec'].value[0])
    print('PM: ',result['pm'].value[0])
    print('Gmag: ',result['phot_g_mean_mag'].value[0])
    print('Distance: ',1000.0/result['parallax'].value[0])
    
    return result, Teff[0], result['teff_gspphot'].value[0], teff_avg


def filename_to_mjd(filename):
    # Regex to capture YYYYMMDD_UTXX:XX:XX.xxx
    match = re.search(r'(\d{8}_UT\d{2}:\d{2}:\d{2}\.\d+)', filename)
    if not match:
        raise ValueError(f"Could not find timestamp in {filename}")
    
    timestamp_str = match.group(1)  # e.g. '20250617_UT00:10:55.496'
    
    # Remove 'UT' and parse into ISO-like format
    date_str, time_str = timestamp_str.split('_UT')
    iso_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]} {time_str}"
    
    # Convert to astropy Time and MJD
    t = Time(iso_str, format='iso', scale='utc')
    return t.mjd

def read_coadd(gaia_id_num, data_files_path = './uv_feros/', wave_range = [5300, 5700],
              show = True):
    
    data_files = os.listdir(data_files_path)
    
    count = 0.0
    waves = []
    fluxes = []
    mjds = []
    for data_file in data_files:
        
        if gaia_id_num in data_file:
            
            mjds.append(filename_to_mjd(data_file))
            
            count+=1.0
            
            wave, flux = merge_orders('./uv_feros/{}'.format(data_file))
                        
            print('./uv_feros/{}'.format(data_file))
            
            waves.append(wave)
            fluxes.append(flux)
            
            if count==1.0:
                wave_ref = wave.copy()
                flux_coadd = flux.copy()
            else:
                flux_interp = np.interp(wave_ref, wave, flux)
                flux_coadd = flux_coadd + flux_interp
                
    print('Number of Coadds: ',count)
    
    if show:
        select = (wave_ref<wave_range[1])&(wave_ref>wave_range[0])
        plt.plot(wave_ref[select], flux_coadd[select]/count)
        plt.xlim(wave_range[0], wave_range[1])
        plt.show()
        
        for (wave, flux) in zip(waves, fluxes):
            select = (wave<wave_range[1])&(wave>wave_range[0])
            plt.plot(wave[select], flux[select])
            plt.xlim(wave_range[0], wave_range[1])
        plt.show()
                
    return wave_ref, flux_coadd/count, waves, fluxes, mjds

def get_galah_broadened(filename, rv, vsini, ld = 0.4):
    galah_spec = fits.open(filename)
    galah_wcs = WCS(galah_spec[0].header)
    galah_waves = []
    for i in range(4096):
        galah_waves.append(galah_wcs.pixel_to_world(i).value)
    galah_waves = np.array(galah_waves)
    galah_flux = galah_spec[0].data
    galah_spec[0].header
    galah_flux_broadened = pyasl.rotBroad(galah_waves, galah_flux, ld, vsini)
    galah_waves = galah_waves * (1 + rv/3e5)
    return galah_waves, galah_flux_broadened


'''
Object Data
'''
wave_obj, flux_obj, wave_all_obj, flux_all_obj, mjds = read_coadd('3499149202247569536', show = False, wave_range = [6320, 6360])
wave_range = np.arange(4500, 6550, 50)
'''
'''


'''
Test with GALAH
'''
#filename = './galah_spectra/galah/dr4/spectra/hermes/com_lowtemp_highsnr/160422/1604220030010861.fits'
#wave_galah, flux_galah = get_galah_broadened(filename, 51, 85)

#wave_obj, flux_obj = median_norm(wave_galah, flux_galah)
#wave_range = np.arange(5650, 5900, 50) #for 2
#wave_range = np.arange(4750, 4950, 50) #for 1
'''
'''

def run_one_window(wri):

    wrl = wave_range[wri]
    wrh = wave_range[wri + 1]

    if wrl == 5850:
        return None  # skip

    diff_rms_grid, star_model_file, rv, vsini, met, tem, alp, car, vmi, topN_met, topN_met_err = find_template_vsini_rv(
        wave_obj, flux_obj, [5000, 4750],
        wave_range=[wrl, wrh],
        model='bosz',
        fix_logg=True, fix_temp=True,
        rv_grid=np.arange(45, 60, 1.0),
        vsini_grid=np.arange(80, 100, 1.0),
        nf_grid=[1.0],
        do_print=False,
        limit_alpha = True,
        limit_vmicro = True
    )

    return rv, vsini, met, tem, alp, car, vmi, topN_met, topN_met_err, star_model_file

#if __name__ == "__main__":
results = Parallel(n_jobs=63)(
    delayed(run_one_window)(i)
    for i in range(len(wave_range) - 1)
)

rv_all, vsini_all, met_all, tem_all, alp_all, car_all, vmi_all, topN_met_all, topN_met_err_all, smf_all = [], [], [], [], [], [], [], [], [], []

for res in results:
    if res is None:
        continue
    rv, vsini, met, tem, alp, car, vmi, topN_met, topN_met_err, smf = res
    rv_all.append(rv)
    vsini_all.append(vsini)
    met_all.append(met)
    tem_all.append(tem)
    alp_all.append(alp)
    car_all.append(car)
    vmi_all.append(vmi)
    topN_met_all.append(topN_met)
    topN_met_err_all.append(topN_met_err)
    smf_all.append(smf)

topN_met_all = np.array(topN_met_all)
topN_met_err_all = np.array(topN_met_err_all)

weighted_topN_met_mean = np.sum(topN_met_all/topN_met_err_all**2)/np.sum(1.0/topN_met_err_all**2)
weighted_topN_met_err = np.sqrt(1.0/np.sum(1.0/topN_met_err_all**2))

print('All Z: ',met_all)
print('All SMF: ',smf_all)
print('Z: ',np.mean(met_all), np.std(met_all))
print('Alpha: ',np.mean(alp_all), np.std(alp_all))
print('Carbon: ',np.mean(car_all), np.std(car_all))
print('Z (quantile): ',np.median(met_all), np.quantile(met_all, 0.15), np.quantile(met_all, 0.85))
print('T: ', np.mean(tem_all), np.std(tem_all))
print('vsini: ', np.mean(vsini_all), np.std(vsini_all))
print('RV: ',np.mean(rv_all), np.std(rv_all))
print('Vmivro: ',np.mean(vmi_all), np.std(vmi_all))
print('Weighted Top N Z mean: ',weighted_topN_met_mean)
print('Weighted Top N Z Error: ',weighted_topN_met_err)
print('Top N Z Mean: ',np.mean(topN_met_all))
print('Top N Z Error: ',np.std(topN_met_all))
