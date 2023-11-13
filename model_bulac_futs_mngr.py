# -*- coding: utf-8 -*-
"""
Created on Feb 28, 2023
Last updated: Apr. 04, 2023

@author: Climate Lead Group; Luis Victor-Gallardo, Jairo Quirós-Tortós
Suggested citation. TBC
                                
"""

import pandas as pd
from pyDOE import lhs
import multiprocessing as mp
import math
import time
import model_bulac_v202304_futs as bulac
import pickle
import sys
import numpy as np
import os
from datetime import date

###############################################################################
# Will read path to save files. 
cwd = os.getcwd()
path = cwd + "/outputs"
###############################################################################

if __name__ == '__main__':

    # Call the information for experimentation:
    di_nam = 'data_inputs_20230411.xlsx'
      
    start_1 = time.time()
    
    ###############################################################################
    # Open the general sheets:  
    # General sheets:
    df1_general = pd.read_excel(di_nam, sheet_name="2_general")
    
    # Calibraton and sets sheets:
    '''
    This code the energy balances introduced by the user:
    '''
    df2_fuel_eq = pd.read_excel(di_nam, sheet_name="3_FUEQ")
    df2_EB = pd.read_excel(di_nam, sheet_name="4_EB")
    df2_InsCap = pd.read_excel(di_nam, sheet_name="5_InsCap")
    df2_scen_sets = pd.read_excel(di_nam, sheet_name="6_scen_sets")
    df2_sets2pp = pd.read_excel(di_nam, sheet_name="7_set2pp")
    df2_trans_sets = pd.read_excel(di_nam, sheet_name="8_trans_sets")
    df2_trans_sets_eq = pd.read_excel(di_nam, sheet_name="9_trans_sets_eq")
    
    # Scenarios sheets:
    df3_scen = pd.read_excel(di_nam, sheet_name="10_scen")
    df3_scen_dems = pd.read_excel(di_nam, sheet_name="11_scen_dems")
    df3_tpt_data = pd.read_excel(di_nam, sheet_name="12_trans_data")
    
    # Technical sheets:
    df4_cfs = pd.read_excel(di_nam, sheet_name="13_cfs")
    df4_ef = pd.read_excel(di_nam, sheet_name="14_emissions")
    df4_job_fac = pd.read_excel(di_nam, sheet_name="15_job_fac")
    df4_tran_dist_fac = pd.read_excel(di_nam, sheet_name="16_t&d")
    df4_caps_rest = pd.read_excel(di_nam, sheet_name="17_cap_rest")
    
    # Economic sheets:
    df5_ext = pd.read_excel(di_nam, sheet_name="18_ext")
    d5_power_techs = pd.read_excel(di_nam, sheet_name="19_power_cost")
    d5_tpt = pd.read_excel(di_nam, sheet_name="20_trans_cost")
    d5_tax = pd.read_excel(di_nam, sheet_name="21_tax")
    
    # Define base inputs AND open the reference data bases.
    base_inputs = [df1_general, df2_fuel_eq, df2_EB, df2_InsCap, df2_scen_sets,
                df2_sets2pp, df2_trans_sets, df2_trans_sets_eq, df3_scen,
                df3_scen_dems, df3_tpt_data, df4_cfs, df4_ef, df4_job_fac,
                df4_tran_dist_fac, df4_caps_rest, df5_ext, d5_power_techs,
                d5_tpt, d5_tax]
    
    dict_database = pickle.load(open('dict_database.pickle', 'rb'))
    
    ##############################################################################

    df_exp = pd.read_excel(di_nam, sheet_name="99_exp")
    
    params_vary = df_exp['Parameter'].tolist()
       
    P = len(params_vary)  # the number of specific parameters to vary
    nindx = df1_general['Parameter'].tolist().index('num_futs')
    N = df1_general['Value'].tolist()[nindx]

    # Now we are ready to apply the LHS function:
    np.random.seed( 555 )

    this_hypercube = lhs(P, samples = N)
 
    # print('check the lhs')
    # sys.exit()
    
    '''
    The variable manipulation for each P is defined below:
    
    1) GDP
    gdp_dict
    
    2) Elasticity of demand (passenger)
    projtype_ela_pas
    
    3) Elasticity of demand (freight)
    projtype_ela_fre
    
    NOTE: IF THE VEHICLE FLEETS ARE USER-DEFINED, THEN, THE VARIATION OF GDP AND ELASTICITY SHOULD FUNCTION AS A RULE OF 3.
    
    4) Fuel costs
    list_var_cost (SELECT THE TECHS)
    
    5) Electricity and H2 costs
    list_var_cost (SELECT THE TECHS)
    
    6) ZEV capital costs
    list_cap_cost (SELECT THE TECHS)
    
    7) Fuel consumption
    fuel_con_lst (SELECT THE TECHS)
    
    8) Electricity and H2 performance
    fuel_con_lst (SELECT THE TECHS)
    
    9) Mode shift (public transport)
    list_mode_shift
    list_non_motorized
    
    10) Electrification
    list_electrification
    list_hydrogen
    '''

    # Define the elements:
    n_fpc = df1_general['Parameter'].tolist().index('futs_per_cycle')
    max_x_per_iter = df1_general['Value'].tolist()[n_fpc]
    
    y = N / max_x_per_iter
    y_ceil = math.ceil( y )
    
    run_parallel = True
    if run_parallel:
        # Iterate across the iterables:
        for n in range(0, y_ceil):
            print('###')
            n_ini = n*max_x_per_iter
            processes = []
            #
            start2 = time.time()
            #
            if n_ini + max_x_per_iter <= N:
                max_iter = n_ini + max_x_per_iter
            else:
                max_iter = N
            #
            for n2 in range( n_ini , max_iter ):
                fut_id = n2 + 1
                print('Future: ' + str(fut_id))
                p = mp.Process(target=bulac.bulac_engine, args=(base_inputs,
                                                                dict_database,
                                                                fut_id,
                                                                this_hypercube,
                                                                df_exp) )
                processes.append(p)
                p.start()
            #
            for process in processes:
                process.join()
            #
            end2 = time.time()   
            time_elapsed2 = -start2 + end2
            print('Cycle: ' + str(n) + '/' + str(y_ceil) + ' | ' + 
                str(time_elapsed2) + ' seconds' )
    
    # Gather the list of output csvs:
    get_listdir = [i for i in os.listdir(path) if 'model_BULAC_simulation_' in i]
    list_files = []
    for afile in get_listdir : 
        list_files.append(pd.read_csv(path + '/' + afile, index_col=None, header=0))
    df_all = pd.concat(list_files, axis=0, ignore_index=True)
    today = date.today()
    df_all.to_csv ( 
        'model_BULAC_futs_' + str( today ).replace( '-', '_' ) + '.csv',
        index = None, header=True)
    
    # Recording final time of execution:
    end_f = time.time()
    te_f = -start_1 + end_f  # te: time_elapsed
    print(str(te_f) + ' seconds /', str(te_f/60) + ' minutes')
    print('*: This multiple-future BULAC is finished.')
