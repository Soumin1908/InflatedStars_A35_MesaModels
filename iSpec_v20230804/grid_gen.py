import os
import sys
import numpy as np
import logging
import multiprocessing
from multiprocessing import Pool

################################################################################
#--- iSpec directory -------------------------------------------------------------
#ispec_dir = os.path.dirname(os.path.realpath(__file__)) + "/"
ispec_dir = './'
sys.path.insert(0, os.path.abspath(ispec_dir))
import ispec


#--- Change LOG level ----------------------------------------------------------
#LOG_LEVEL = "warning"
LOG_LEVEL = "info"
logger = logging.getLogger() # root logger, common for all
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
################################################################################

teff_grid = np.arange(4700, 5450, 50)
mh_grid = np.arange(-0.5, +0.5, 0.05)

def precompute_synthetic_grid(code="spectrum"):
    precomputed_grid_dir = "example_grid_%s/" % (code)

    ## - Read grid ranges from file
    #from astropy.io import ascii
    #ranges_filename = "input/minigrid/initial_estimate_grid_ranges.tsv"
    #ranges = ascii.read(ranges_filename, delimiter="\t")
    ## - or define them directly here (example of only 2 reference points):
    
    nmodels = len(teff_grid)*len(mh_grid)
    
    ranges = np.recarray((nmodels,),  dtype=[('teff', int), ('logg', float), ('MH', float), ('alpha', float), ('vmic', float)])
    
    c = 0
    for teff in teff_grid:
        for mh in mh_grid:
                ranges['teff'][c] = teff
                ranges['logg'][c] = 3.5
                ranges['MH'][c] = mh
                ranges['alpha'][c] = 0.1
                ranges['vmic'][c] = ispec.estimate_vmic(ranges['teff'][c], ranges['logg'][c], ranges['MH'][c])
                c+=1
    
    '''
    ranges['teff'][0] = 4800
    ranges['logg'][0] = 3.5
    ranges['MH'][0] = -0.5
    ranges['alpha'][0] = 0.0
    ranges['vmic'][0] = ispec.estimate_vmic(ranges['teff'][0], ranges['logg'][0], ranges['MH'][0])
    ranges['teff'][1] = 5000
    ranges['logg'][1] = 3.5
    ranges['MH'][1] = +0.2
    ranges['alpha'][1] = 0.0
    ranges['vmic'][1] = ispec.estimate_vmic(ranges['teff'][1], ranges['logg'][1], ranges['MH'][1])
    '''

    # Wavelengths
    #initial_wave = 480.0
    #final_wave = 680.0
    initial_wave = 450.0
    final_wave = 650.0
    step_wave = 0.003
    wavelengths = np.arange(initial_wave, final_wave, step_wave)

    to_resolution = 50000 # Individual files will not be convolved but the grid will be (for fast comparison)
    number_of_processes = 8 # It can be parallelized for computers with multiple processors


    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/"

    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv6_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv6_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(atomic_linelist_file, wave_base=initial_wave, wave_top=final_wave)
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] # Select lines that have some minimal contribution in the sun

    isotopes = ispec.read_isotope_data(isotope_file)

    if "ATLAS" in model:
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    else:
        # MARCS
        solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)
    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)

    ## Custom fixed abundances
    #fixed_abundances = ispec.create_free_abundances_structure(["C", "N", "O"], chemical_elements, solar_abundances)
    #fixed_abundances['Abund'] = [-3.49, -3.71, -3.54] # Abundances in SPECTRUM scale (i.e., x - 12.0 - 0.036) and in the same order ["C", "N", "O"]
    ## No fixed abundances
    fixed_abundances = None


    ispec.precompute_synthetic_grid(precomputed_grid_dir, ranges, wavelengths, to_resolution, \
                                    modeled_layers_pack, atomic_linelist, isotopes, solar_abundances, \
                                    segments=None, number_of_processes=number_of_processes, \
                                    code=code, steps=False)
                                    
                                    
if __name__ == '__main__':
        precompute_synthetic_grid(code="spectrum")
