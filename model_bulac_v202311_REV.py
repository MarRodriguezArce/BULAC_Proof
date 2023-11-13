# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13, 2021
Last updated: Aug. 23, 2023

@author: Climate Lead Group; Luis Victor-Gallardo, Jairo Quirós-Tortós
Suggested citation: UNEP (2022). Is Natural Gas a Good Investment for Latin 
                                America and the Caribbean? From Economic to 
                                Employment and Climate Impacts of the Power
                                Sector. https://wedocs.unep.org/handle/20.500.11822/40923
                                
"""

import pandas as pd
import pickle
import sys
from copy import deepcopy
import math
import numpy as np
import time
import os
import warnings


# Import functions that support this model:
from model_bulac_funcs import intersection_2, interpolation_to_end, \
    fun_reverse_dict_data, fun_extract_new_dict_data, fun_dem_model_projtype, \
    fun_dem_proj, fun_unpack_costs, fun_unpack_taxes, \
    interpolation_non_linear_final, interpolation_to_end_debug

pd.options.mode.chained_assignment = None  # default='warn'

# Globally suppress the FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning)

# > Define booleans to control process:
# if True, it overwrites energy projections with transport model
overwrite_transport_model = False

##############################################################################
# SIMULATION: Implement the equations designed in "model_design" #
# Recording initial time of execution
start_1 = time.time()
# di_nam = 'data_inputs_20231009.xlsx'
di_nam = 'data_inputs_20231106.xlsx'
# di_nam = 'data_inputs_20230907_PPTX.xlsx'


###############################################################################
# 0) open the reference data bases.

dict_database = pickle.load(open('dict_db.pickle', 'rb'))

# print('Check the dict database.')
# sys.exit()

###############################################################################
# Will save all files in one outputs folder. 
# Folder is created if it does not exist
cwd = os.getcwd()
path = cwd + "/outputs"

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

###############################################################################
# Here we relate power plants and production per carrier in the base year:
'''
#> Select the technologies that exist in the base year:
PP_Offshore_Wind: -
PP_Onshore_Wind: base year
PP_PV Utility_Solar: base year
PP_PV DistComm_Solar: -
PP_PV DistResi_Solar: -
PP_CSP_Solar: -
PP_Geothermal: base year
PP_Hydro: base year
PP_Nuclear: base year
PP_Thermal.re_Sugar cane and derivatives: base year
PP_PV Utility+Battery_Solar: -
PP_Thermal_Coal: base year
PP_Thermal_Natural Gas: base year
ST_Utility Scale Battery: -
ST_Commercial Battery: -
ST_Residential Battery: -
ST_Pumped Hydro: -
PP_Other: -
PP_Thermal_Fuel oil: - (we cannot distinguish Diesel/Fuel Oil with our data)
PP_Thermal_Diesel: base year
'''

'''
#> Select the fuels that exist in the base year:
Coal: base year
Oil: base year
Natural gas: base year
Biofuels: base year
Waste: -
Nuclear: base year
Hydro: base year
Geothermal: base year
Solar PV: base year
Solar thermal: -
Wind: base year
Tide: -
Other sources: -
'''

dict_equiv_country = { \
    'Argentina':'Argentina',  # this works for iea dataframe
    'Barbados':'Barbados',
    'Belice':'Belice',
    'Bolivia':'Bolivia',
    'Brasil':'Brasil',
    'Chile':'Chile',
    'Colombia':'Colombia',
    'Costa Rica':'Costa Rica',
    'Cuba':'Cuba',
    'Ecuador':'Ecuador ',
    'El Salvador':'El Salvador',
    'Grenada':'Grenada',
    'Guatemala':'Guatemala',
    'Guyana':'Guyana ',
    'Haiti':'Haiti',
    'Honduras':'Honduras',
    'Jamaica':'Jamaica',
    'Mexico':'Mexico ',
    'Nicaragua':'Nicaragua',
    'Panama':'Panama',
    'Paraguay':'Paraguay',
    'Peru':'Peru',
    'Dominican Republic':'Republica Dominicana',
    'Suriname':'Suriname',
    'Trinidad and Tobago':'Trinidad and Tobago',
    'Uruguay':'Uruguay',
    'Venezuela':'Venuezuela'}

dict_equiv_country_2 = { \
    'Argentina':'Argentina',  # this works for iea dataframe
    'Barbados':'Barbados',
    'Belice':'Belice',
    'Bolivia':'Bolivia',
    'Brasil':'Brasil',
    'Chile':'Chile',
    'Colombia':'Colombia',
    'Costa Rica':'Costa Rica',
    'Cuba':'Cuba',
    'Ecuador':'Ecuador',
    'El Salvador':'El Salvador',
    'Grenada':'Grenada',
    'Guatemala':'Guatemala',
    'Guyana':'Guyana',
    'Haiti':'Haiti',
    'Honduras':'Honduras',
    'Jamaica':'Jamaica',
    'Mexico':'Mexico',
    'Nicaragua':'Nicaragua',
    'Panama':'Panama',
    'Paraguay':'Paraguay',
    'Peru':'Peru',
    'Republica Dominicana':'Republica Dominicana',
    'Suriname':'Suriname',
    'Trinidad and Tobago':'Trinidad and Tobago',
    'Uruguay':'Uruguay',
    'Venezuela':'Venezuela'}

# Find the common countries per region:
unique_reg = []
dict_regs_and_countries_raw = {}
dict_regs_and_countries = {}

k1_count = 0
k1_list = []
for k1 in list(dict_database.keys()):  # across databases
    k1_list.append(k1)
    for k2 in list(dict_database[k1].keys()):  # across regions
        dummy_list = list(dict_database[k1][k2].keys())
        add_dummy = {k1:dummy_list}
        if k1_count == 0:
            dict_regs_and_countries_raw.update({k2:add_dummy})
        else:
            if k2 == '2_CA':
                k2 = '2_Central America'
            else:
                pass
            dict_regs_and_countries_raw[k2].update(add_dummy)
    k1_count += 1

for reg in list(dict_regs_and_countries_raw.keys()):
    if 'Trinidad & Tobago' in dict_regs_and_countries_raw[reg][k1_list[0]]:
        fix_idx = dict_regs_and_countries_raw[reg][k1_list[0]].index('Trinidad & Tobago')
        dict_regs_and_countries_raw[reg][k1_list[0]][fix_idx] = 'Trinidad and Tobago'
    country_list = intersection_2(dict_regs_and_countries_raw[reg][k1_list[0]],
                                  dict_regs_and_countries_raw[reg][k1_list[1]])
    dict_regs_and_countries.update({reg:country_list})

###############################################################################
# *Input parameters are listed below*:
# Capacity: (initial condition by power plant Tech, OLADE, dict_database)
# Production: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Imports: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Exports: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Externality cost: (by Fuel, IMF, data_inputs => costs_externalities)
# CAPEX (capital unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# CAU (max. activity production per unit of capacity, ATB, data_inputs => costs_power_techs)
# Fixed FOM (fixed FOM unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# Grid connection cost (GCC unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# Heat Rate (IAR (only for fossil-based plants), ATB, data_inputs => costs_power_techs)
    # WARNING! may need IAR for renewables
# Net capacity factor: (real activity per max. activity possible, ATB, data_inputs => costs_power_techs)
# Operational life (by power plant Tech, ATB, data_inputs => costs_power_techs)
# Variable FOM (by power plant activity, ATB, data_inputs => costs_power_techs)
# Emission factor: (by Fuel, type == consumption, see README of data_inputs)
# Demand energy intensity (by Demand sector == Tech, data_inputs => scenarios)
# Distribution of end-use consumption (by Fuel, data_inputs => scenarios)
# Distribution of new electrical energy generation (by power plant Tech, data_inputs => scenarios)
# %Imports (by Fuel, data_inputs => scenarios)
# %Exports (by Fuel, data_inputs => scenarios)
# GDP growth (combine with energy intensity, data_inputs => scenarios)
# GDP (grab from historic data, and combine with GDP growth)
# Export price (by Fuel) # WARNING! MISSING CORRECT CALIBRATION!
# Import price (by Fuel) # WARNING! MISSING CORRECT CALIBRATION!

# *Output parameters are listed below*:
# Energy Demand by Fuel and Demand sector (Tech)
# New capacity: endogenous // f(demand, scenarios, net capacity factor)
# CAPEX (by power plant Tech)
# Fixed OPEX (by power plant Tech)
# Var OPEX (by Fuel)
# Imports expenses (by Fuel)
# Exports revenue (by Fuel)
# Emissions (by Fuel)
# Externality Global Warming (by Fuel)
# Externality Local Pollution (by Fuel)

# Review the EB structure:
# Capacity: Cap > region > country > tech (power plant) > year (2019)
# Demand: EB > region > country > Energy consumption > Demand sector > Fuel > year str(2019)
# Transformation: EB > region > country > Total transformation > Demand sector > Fuel > year str(2019)
    # here, negative: use // positive: produce
# Local production: EB > Total supply > Exports/Imports/Production > Fuel > year str(2019)

# Then, simply produce a table with inputs and outputs for each dimension combination, but make sure to have the calculations done first.

###############################################################################
# 3) implementing...
# 3a) open the "data_inputs.xlsx", each one of the sheets into dfs:

'''
Name definitions:
di_nam: data_inputs_name
'''

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
df3_scen_matrix = pd.read_excel(di_nam, sheet_name="11.2_scen_matrix")
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

##############################################################################
# Process the content of the general sheet (1_general):
df1_general.set_index('Parameter', inplace = True)
dict_general_inp = df1_general.to_dict('index')
'''
The columns of the general dictionary are:
['Value', 'Year', 'Attribute', 'Unit', 'Source', 'Description']

The keys (variables) of the dictionary are:
 ['ini_year', 'fin_year', 'country', 'gdp', 'discount_rate', 'discount_year']
'''

# Call years
per_first_yr = dict_general_inp['ini_year']['Value']
per_last_yr = dict_general_inp['fin_year']['Value']
time_vector = [i for i in range(per_first_yr, per_last_yr+1)]

# Call countries
dict_gen_all_param_names = list(dict_general_inp.keys())
dict_gen_country_params = [i for i in dict_gen_all_param_names
                           if 'country' in i]

general_country_list, gen_cntry_list_param_idx = [], []
for cntry_idx in dict_gen_country_params:
    general_country_list.append(dict_general_inp[cntry_idx]['Value'])
    gen_cntry_list_param_idx.append(cntry_idx)

# Get the regions of interest and store them for future use:
regions_list = []
all_regions = dict_regs_and_countries.keys()
country_2_reg = {}
for areg in all_regions:
    all_country_list = dict_regs_and_countries[areg]
    for cntry in all_country_list:
        country_2_reg.update({cntry:areg})

        # Store useful regions for future use:
        if areg not in regions_list and cntry in general_country_list:
            regions_list.append(areg)

# Call GDP and population.
gdp_dict = {}  # per country index
popbase_dict = {}
popfinal_dict = {}
popproj_dict = {}
for cntry in general_country_list:
    cntry_list_idx = general_country_list.index(cntry)
    cntry_idx = gen_cntry_list_param_idx[cntry_list_idx]
    gdp_idx = 'gdp_' + str(cntry_idx.split('_')[-1])
    popbase_idx = 'pop_base_' + str(cntry_idx.split('_')[-1])
    popfinal_idx = 'pop_final_' + str(cntry_idx.split('_')[-1])
    popproj_idx = 'pop_proj_' + str(cntry_idx.split('_')[-1])

    if gdp_idx in list(dict_general_inp.keys()):
        gdp_value = dict_general_inp[gdp_idx]['Value']
        gdp_dict.update({cntry: gdp_value})

        popbase_value = dict_general_inp[popbase_idx]['Value']
        popbase_dict.update({cntry: popbase_value})

        popfinal_value = dict_general_inp[popfinal_idx]['Value']
        popfinal_dict.update({cntry: popfinal_value})

        popproj_value = dict_general_inp[popproj_idx]['Value']
        popproj_dict.update({cntry: popproj_value})
    else:
        print('There is no GDP value defined for: ' + cntry)

'''
Development note: only introduce 1 GDP year for the future. The rest of the
years should be controlled by the GDP growth parameter.

The population is introduced for two years: first and last. The interpolation
is linear. This can change of other data is provided.
'''

# Call information for discounting:
r_rate = dict_general_inp['discount_rate']['Value']
r_year = dict_general_inp['discount_year']['Value']
ini_simu_yr = dict_general_inp['ini_simu_yr']['Value']

##############################################################################
# Process the content of structural sheets:

# This code extracts the sets used for energy balancing:
list_scen_fuels = df2_scen_sets['Fuel'].tolist()
list_scen_fuel_primary_and_secondary = \
    df2_scen_sets['Primary, Secondary or Power'].tolist()
list_scen_fuels_u = list(dict.fromkeys(list_scen_fuels))
list_scen_fuels_u_prim_and_sec = []
list_scen_fuels_cat_u = []
for af in list_scen_fuels_u:
    this_fuel_idx = list_scen_fuels.index(af)
    this_fuel_cat = list_scen_fuel_primary_and_secondary[this_fuel_idx]
    list_scen_fuels_cat_u.append(this_fuel_cat)
    if this_fuel_cat in ['Primary', 'Secondary']:
        list_scen_fuels_u_prim_and_sec.append(af)

# This code extracts sets to connect power plants to energy balance:
dict_equiv_pp_fuel = {}
dict_equiv_pp_fuel_rev = {}
for n in range(len(df2_sets2pp['Technology'])):
    dict_equiv_pp_fuel.update(
        {df2_sets2pp['Technology'][n]:\
         df2_sets2pp['Fuel'][n]})
    dict_equiv_pp_fuel_rev.update(
        {df2_sets2pp['Fuel'][n]:\
         df2_sets2pp['Technology'][n]})

# This code extracts the transport sets and its structure:
list_trn_type = df2_trans_sets['Type'].tolist()
list_trn_fuel = df2_trans_sets['Fuel'].tolist()
list_trn_type_and_fuel = []
for n in range(len(list_trn_type)):
    this_type, this_fuel = list_trn_type[n], list_trn_fuel[n]
    if this_fuel != '-':
        this_type_and_fuel = this_type + '_' + this_fuel
    else:
        this_type_and_fuel = this_type
    list_trn_type_and_fuel.append(this_type_and_fuel)

list_trn_lvl1_u_raw = df2_trans_sets['Demand set level 1'].tolist()
list_trn_lvl2_u_raw = df2_trans_sets['Demand set level 2'].tolist()
list_trn_lvl1_u = \
    [i for i in list(dict.fromkeys(list_trn_lvl1_u_raw)) if '-' != i]
list_trn_lvl2_u = \
    [i for i in list(dict.fromkeys(list_trn_lvl2_u_raw)) if '-' != i]
# The level 2 list only applies to passenger vehicles
dict_trn_nest = {}
for l1 in range(len(list_trn_lvl1_u)):
    this_l1 = list_trn_lvl1_u[l1]
    dict_trn_nest.update({this_l1:{}})
    if this_l1 != 'Passenger':
        this_l2 = 'All'
        mask_trans_t_and_f = \
            (df2_trans_sets['Demand set level 1'] == this_l1) & \
            (df2_trans_sets['Fuel'] == '-')
        df_transport_t_and_f = df2_trans_sets.loc[mask_trans_t_and_f]
        list_trn_types = df_transport_t_and_f['Type'].tolist()
        dict_trn_nest[this_l1].update({this_l2:deepcopy(list_trn_types)})
    else:
        for l2 in range(len(list_trn_lvl2_u)):
            this_l2 = list_trn_lvl2_u[l2]
            mask_trans_t_and_f = \
                (df2_trans_sets['Demand set level 1'] == this_l1) & \
                (df2_trans_sets['Demand set level 2'] == this_l2) & \
                (df2_trans_sets['Fuel'] == '-')
            df_transport_t_and_f = df2_trans_sets.loc[mask_trans_t_and_f]
            list_trn_types = df_transport_t_and_f['Type'].tolist()
            dict_trn_nest[this_l1].update({this_l2:deepcopy(list_trn_types)})

# This code extracts set change equivalence:
pack_fe = {'new2old':{}, 'old2new':{}}
for n in range(len(df2_fuel_eq['OLADE_structure'].tolist())):
    old_struc = df2_fuel_eq['OLADE_structure'].tolist()[n]
    new_struc = df2_fuel_eq['New_structure'].tolist()[n]
    pack_fe['new2old'].update({new_struc: old_struc})
    pack_fe['old2new'].update({old_struc: new_struc})

# we have to open the first data frame:
# 1) list all the unique elements from the "df3_tpt_data" parameters:
tr_list_scenarios = df3_tpt_data['Scenario'].tolist()
tr_list_scenarios_u = list(dict.fromkeys(tr_list_scenarios))

tr_list_app_countries = df3_tpt_data['Application_Countries'].tolist()
tr_list_app_countries_u = list(dict.fromkeys(tr_list_app_countries))


# OVERRIDE COUNTRIES FOR TESTING
# tr_list_app_countries_u = ['Republica Dominicana', 'Costa Rica']
# tr_list_app_countries_u = ['Republica Dominicana']
# tr_list_app_countries_u = ['Costa Rica', 'El Salvador', 'Guatemala']
# tr_list_app_countries_u = ['Bolivia']
# tr_list_app_countries_u = ['Honduras']
# tr_list_app_countries_u = ['Chile', 'Ecuador']
# tr_list_app_countries_u = ['Uruguay', 'Bolivia', 'Honduras', 'Republica Dominicana']
# tr_list_app_countries_u = ['Uruguay']

# tr_list_app_countries_u = ['Costa Rica']
# tr_list_app_countries_u = ['Colombia']
# tr_list_app_countries_u = ['Brasil']

'''
note: note that you now change (decrease) the thermal capacity factors when supply exceeeds demand
and never raise them again. This can be fixing a lot of things for example the flat demands fro Bolivia.
'''

# tr_list_app_countries_u = ['Honduras']
# tr_list_app_countries_u = ['Bolivia']
# tr_list_app_countries_u = ['Barbados']

# tr_list_app_countries_u = ['Colombia']
# tr_list_app_countries_u = ['Ecuador']
# tr_list_app_countries_u = ['Costa Rica']


tr_list_parameters = df3_tpt_data['Parameter'].tolist()
tr_list_parameters_u = list(dict.fromkeys(tr_list_parameters))

tr_list_type_and_fuel = df3_tpt_data['Type & Fuel ID'].tolist()
tr_list_type_and_fuel_u = list(dict.fromkeys(tr_list_type_and_fuel))

tr_list_type = df3_tpt_data['Type'].tolist()
tr_list_type_u = list(dict.fromkeys(tr_list_type))

tr_list_fuel = df3_tpt_data['Fuel'].tolist()
tr_list_fuel_u = list(dict.fromkeys(tr_list_fuel))

tr_list_projection = df3_tpt_data['projection'].tolist()
tr_list_projection_u = list(dict.fromkeys(tr_list_projection))

# We must overwrite the dict-database based on OLADE for a user_defined
# input to avoid compatibility issues.
use_original_pickle = True
if use_original_pickle is True:
    pass
else:
    dict_database_freeze = deepcopy(dict_database)
    # fun_reverse_dict_data(dict_database_freeze, '5_Southern Cone',
    #                       'Uruguay', True,
    #                       list_scen_fuels_u_prim_and_sec, pack_fe)
    print('We must re-write the base data. This can take a while.')
    # We must use the reference EB and InstCap sheets from data_inputs
    # agile_mode = True
    agile_mode = False
    if agile_mode is False:
        dict_ref_EB, dict_ref_InstCap = \
            fun_extract_new_dict_data(df2_EB, df2_InsCap, per_first_yr)
        with open('dict_ref_EB.pickle', 'wb') as handle1:
            pickle.dump(dict_ref_EB, handle1,
                        protocol=pickle.HIGHEST_PROTOCOL)
        handle1.close()
        with open('dict_ref_InstCap.pickle', 'wb') as handle2:
            pickle.dump(dict_ref_InstCap, handle2,
                        protocol=pickle.HIGHEST_PROTOCOL)
        handle2.close()
    else:
        dict_ref_EB = pickle.load(open('dict_ref_EB.pickle', 'rb'))
        dict_ref_InstCap = \
            pickle.load(open('dict_ref_InstCap.pickle', 'rb'))

    # We must replace the dictionaries:
    dict_database['EB'] = deepcopy(dict_ref_EB)
    dict_database['Cap'] = deepcopy(dict_ref_InstCap)

# print('Review the sets and general inputs')
# sys.exit()

#######################################################################

# 3b) create the nesting structure to iterate across:
    # future > scenario > region > country > yeat
    # WARNING! Inlcude only 1 future in this script, i.e., here we only produce future 0 inputs & outputs

# ... extracting the unique list of future...
scenario_list = list(set(df3_scen['Scenario'].tolist()))
scenario_list.remove('ALL')

scenario_list.sort()

# scenario_list = ['NDCPLUS']
# scenario_list = ['NDC']

dict_test_transport_model = {}

# ... we will work with a single dictionary containing all simulations:
# RULE: most values are time_vector-long lists, except externality unit costs (by country):
ext_by_country = {}

count_under_zero = 0

base_year = str(per_first_yr)

print('PROCESS 1 - RUNNING THE SIMULATIONS')
dict_scen = {}  # fill and append to final dict later
idict_net_cap_factor_by_scen_by_country = {}
store_percent_BAU = {}

# OVERRIDE SCENAIRIO LIST FOR DEBUGGING
# scenario_list = ['BAU', 'NDC']
#scenario_list = ['NDCPLUS']

# Here we must create an empty dictionary with the regional consumption
list_fuel_ALL = list(set([v for v in list(set(df3_scen['Fuel'].tolist()))]))
list_fuel_ALL = [i for i in list_fuel_ALL if type(i) is str]
dict_energy_demand_by_fuel_sum = {
    k: [0] * len(time_vector) for k in list_fuel_ALL}

for s in range(len(scenario_list)):
    this_scen = scenario_list[s]
    print('# 1 - ', this_scen)

    dict_test_transport_model.update({this_scen:{}})

    dict_local_reg = {}
    idict_net_cap_factor_by_scen_by_country.update({this_scen:{}})

    # We need to store a dictionary for each country that store production
    # from inputted capacity and capacity factors for Anomaly identification
    dict_store_res_energy_orig = {}

    for r in range(len(regions_list)):
        this_reg = regions_list[r]
        print('   # 2 - ', this_reg)

        country_list = dict_regs_and_countries[this_reg]
        country_list.sort()
        dict_local_country = {}

        # Add a filter to include countries with transport data only:
        country_list = [c for c in country_list if c in tr_list_app_countries_u]

        for c in range(len(country_list)):
            this_country = country_list[c]
            print('      # 3 - ', this_country)

            dict_test_transport_model[this_scen].update({this_country:{}})

            # ...store the capacity factor by country:
            idict_net_cap_factor_by_scen_by_country[this_scen].update({this_country:{}})

            # ...call the GDP of the base year
            this_gdp_base = gdp_dict[this_country]

            # ...call and make population projection
            this_pop_base = popbase_dict[this_country]
            this_pop_final = popfinal_dict[this_country]
            this_pop_proj = popproj_dict[this_country]
            this_pop_vector_known = ['' for y in range(len(time_vector))]
            this_pop_vector_known[0] = this_pop_base
            this_pop_vector_known[-1] = this_pop_final
            if this_pop_proj == 'Linear':
                this_pop_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr, \
                        this_pop_vector_known, 'last', this_scen, 'Population')

            # ...subselect the scenario dataframe you will use
            mask_scen = \
                (df3_scen['Scenario'] == this_scen) | \
                (df3_scen['Scenario'] == 'ALL')
            df_scen = df3_scen.loc[mask_scen] #  _rc is for "region" and "country"
            df_scen.reset_index(drop=True, inplace=True)

            indices_df_scen = df_scen.index.tolist()
            list_application_countries_all = \
                df_scen['Application_Countries'].tolist()
            list_application_countries = \
                list(set(df_scen['Application_Countries'].tolist()))

            for ac in list_application_countries:
                if this_country in ac.split(' ; '):
                    select_app_countries = deepcopy(ac)

            indices_df_scen_select = [i for i in range(len(indices_df_scen))
                                      if (list_application_countries_all[i]
                                          == select_app_countries) or
                                         (list_application_countries_all[i]
                                          == 'ALL') or
                                         (this_country in
                                          list_application_countries_all[i])
                                          ]

            df_scen_rc = df_scen.iloc[indices_df_scen_select]
            df_scen_rc.reset_index(drop=True, inplace=True)

            # 3c) create the demands per fuel per country in a single dictionary (follow section 2 for structure)
            # This depends on the EB dictionary, the GDP, and the scenarios' <Demand energy intensity>
            # From EB, extract the elements by demand and fuel:

            this_country_2 = dict_equiv_country_2[this_country]

            dict_base_energy_demand = \
                dict_database['EB'][this_reg][this_country_2]['Energy consumption']
            list_demand_sector_techs = list(dict_base_energy_demand.keys())
            list_demand_sector_techs.remove('none')

            list_fuel_raw = list(dict_base_energy_demand['none'].keys())
            list_fuel = [e for e in list_fuel_raw if ('Total' not in e and
                                                      'Non-' not in e)]

            # We must now create a dictionary with the parameter, the technology, and the fuel.
            # By default, demand technologies consume the fuel.
            param_related = 'Demand energy intensity'  # this is in "scenarios"
            param_related_2 = 'GDP growth'  # this is in "scenarios"
            param_related_3 = 'Distribution of end-use consumption'  # this is in "scenarios"

            # Select the "param_related"
            mask_param_related = (df_scen_rc['Parameter'] == param_related)
            df_param_related = df_scen_rc.loc[mask_param_related]
            df_param_related.reset_index(drop=True, inplace=True)

            # Select the "param_related_2"
            mask_param_related_2 = (df_scen_rc['Parameter'] == param_related_2)
            df_param_related_2 = df_scen_rc.loc[mask_param_related_2]
            df_param_related_2.reset_index(drop=True, inplace=True)

            # Select the "param_related_3"
            mask_param_related_3 = (df_scen_rc['Parameter'] == param_related_3)
            df_param_related_3 = df_scen_rc.loc[mask_param_related_3]
            df_param_related_3.reset_index(drop=True, inplace=True)

            # ...select an alternative "param_related_3" where scenarios can be managed easily
            mask_scen_3 = \
                (df3_scen_dems['Scenario'] == this_scen) | \
                (df3_scen_dems['Scenario'] == 'ALL')
            df_scen_3 = df3_scen_dems.loc[mask_scen_3]
            df_scen_3.reset_index(drop=True, inplace=True)

            indices_df_scen_spec = df_scen_3.index.tolist()
            list_application_countries_spec = \
                df_scen_3['Application_Countries'].tolist()

            indices_df_scen_select = [i for i in range(len(indices_df_scen_spec))
                                      if (list_application_countries_spec[i]
                                          == select_app_countries) or
                                         (list_application_countries_spec[i]
                                          == 'ALL') or
                                         (this_country in
                                          list_application_countries_spec[i])
                                          ]

            df_scen_3_spec = df_scen_3.iloc[indices_df_scen_select]
            df_scen_3_spec.reset_index(drop=True, inplace=True)  # this should be ready to use

            mask_scen_3_2 = \
                ((df3_scen_matrix['Scenario'] == this_scen) | \
                (df3_scen_matrix['Scenario'] == 'ALL')) & \
                (df3_scen_matrix['Country'] == this_country)
            df_scen_3_2 = df3_scen_matrix.loc[mask_scen_3_2]
            df_scen_3_2.reset_index(drop=True, inplace=True)

            # print('have we read the new sheet?')
            # sys.exit()

            # We may add other fuels that are not present in the original list fuel
            list_fuel_additional = [v for v in list(set(df_param_related_3['Fuel'].tolist())) if v not in list_fuel]
            # print('got here')
            # sys.exit()
            list_fuel_orig = deepcopy(list_fuel)
            list_fuel = list_fuel_orig + list_fuel_additional 

            ###################################################################

            # ...acting for "GDP growth"
            this_gdp_growth_projection = df_param_related_2.iloc[0]['projection']
            this_gdp_growth_value_type = df_param_related_2.iloc[0]['value']
            this_gdp_growth_vals_raw = []
            this_gdp_growth_vals = []
            for y in time_vector:
                this_gdp_growth_vals_raw.append(df_param_related_2.iloc[0][y])
            if (this_gdp_growth_projection == 'flat' and
                    this_gdp_growth_value_type == 'constant'):
                for y in range(len(time_vector)):
                    this_gdp_growth_vals.append(this_gdp_growth_vals_raw[0])
            if this_gdp_growth_projection == 'user_defined':
                this_gdp_growth_vals = deepcopy(this_gdp_growth_vals_raw)

            # ...acting for GDP and GDP per capita:
            this_gdp_vals = []
            this_gdp_per_cap_vals = []
            this_gdp_pc_growth_vals = []
            this_pop_growth_vals = []
            for y in range(len(time_vector)):
                if y == 0:
                    this_gdp_vals.append(this_gdp_base)
                else:
                    this_growth = this_gdp_growth_vals[y]
                    next_year_gdp = this_gdp_vals[-1]*(1+this_growth/100)
                    this_gdp_vals.append(next_year_gdp)

                this_y_gdp_per_capita = \
                    this_gdp_vals[-1]/(this_pop_vector[y]*1e6)
                this_gdp_per_cap_vals.append(this_y_gdp_per_capita)
                if y != 0:
                    # Calculate the growth of the GDP per capita
                    gdp_pc_last = this_gdp_per_cap_vals[y-1]
                    gdp_pc_present = this_gdp_per_cap_vals[y]
                    this_gdp_pc_growth = \
                        100*(gdp_pc_present - gdp_pc_last)/gdp_pc_last
                    this_gdp_pc_growth_vals.append(this_gdp_pc_growth)

                    # Calculate the growth of the population
                    pop_last = this_pop_vector[y-1]
                    pop_present = this_pop_vector[y]
                    this_pop_growth = 100*(pop_present - pop_last)/pop_last
                    this_pop_growth_vals.append(this_pop_growth)
                else:
                    this_gdp_pc_growth_vals.append(0)
                    this_pop_growth_vals.append(0)

            # Create the energy demand dictionary (has a projection):
            dict_energy_demand = {}  # by sector
            dict_energy_intensity = {}  # by sector
            dict_energy_demand_by_fuel = {}  # by fuel
            tech_counter = 0
            for tech in list_demand_sector_techs:

                tech_idx = df_param_related['Demand sector'].tolist().index(tech)
                tech_counter += 1

                # ...acting for "Demand energy intensity" (_dei)
                this_tech_dei_df_param_related = df_param_related.iloc[tech_idx]
                this_tech_dei_projection = this_tech_dei_df_param_related['projection']
                this_tech_dei_value_type = this_tech_dei_df_param_related['value']
                this_tech_dei_known_vals_raw = []
                this_tech_dei_known_vals = []

                ref_energy_consumption = \
                    dict_base_energy_demand[tech]['Total'][base_year]

                y_count = 0
                for y in time_vector:
                    this_tech_dei_known_vals_raw.append(this_tech_dei_df_param_related[y])
                    # Act already by attending "endogenous" calls:
                    if this_tech_dei_df_param_related[y] == 'endogenous':
                        add_value = ref_energy_consumption*1e9/this_gdp_vals[y_count]  # MJ/USD
                    elif math.isnan(float(this_tech_dei_df_param_related[y])) is True and y_count >= 1:
                        add_value = ''
                    elif (float(this_tech_dei_df_param_related[y]) != 0.0 and
                            this_tech_dei_value_type == 'rel_by'):
                        add_value = \
                            this_tech_dei_known_vals[0]*this_tech_dei_df_param_related[y]
                    this_tech_dei_known_vals.append(add_value)
                    y_count += 1

                this_tech_dei_vals = \
                    interpolation_to_end(time_vector, ini_simu_yr, \
                        this_tech_dei_known_vals, 'last', this_scen, '')

                # ...since we have the intensities, we can obtain the demands:
                this_tech_ed_vals = []
                y_count = 0
                for y in time_vector:
                    add_value = \
                        this_tech_dei_vals[y_count]*this_gdp_vals[y_count]/1e9  # PJ
                    this_tech_ed_vals.append(add_value)
                    y_count += 1

                # store the total energy demand:
                dict_energy_demand.update({tech:{'Total':this_tech_ed_vals}})
                dict_energy_intensity.update({tech:{'Total':this_tech_dei_vals}})

                # ...we can also obtain the demands per fuel, which can vary depending on the "apply_type"
                # we will now iterate across fuels to find demands:                

                total_sector_demand_baseyear = 0  # this applies for a *demand technology*
                for loc_fuel in list_fuel:  # sum across al fuels for a denominator
                    if 'Total' not in loc_fuel and 'Non-' not in loc_fuel:
                        if loc_fuel in list_fuel_orig:
                            total_sector_demand_baseyear += \
                                dict_base_energy_demand[tech][loc_fuel][base_year]

                        if tech_counter == 1:  # we must calculate the total energy demand by fuel
                            zero_list = [0 for y in range(len(time_vector))]
                            dict_energy_demand_by_fuel.update({loc_fuel:zero_list})

                # ...these are variables are needed for internal distribution of demands
                check_percent = False
                store_fush = {}  # "fush"  means "fuel shares"
                store_fush_rem = {}

                # Here we need to add the None_reduction to the energy dictionary

                # Store a dictionary with the endogenous shares to use below:
                dict_endo_sh = {}

                for fuel in list_fuel:
                    #if 'Total' not in fuel and 'Other' not in fuel and 'Non-' not in fuel:
                    if 'Total' not in fuel and 'Non-' not in fuel:
                        fuel_idx = df_param_related_3['Fuel'].tolist().index(fuel)

                        # ...acting for "Distribution of end-use consumption" (_deuc)
                        this_fuel_deuc_df_param_related = df_param_related_3.iloc[tech_idx]
                        this_fuel_deuc_projection = this_fuel_deuc_df_param_related['projection']
                        this_fuel_deuc_value_type = this_fuel_deuc_df_param_related['value']
                        this_fuel_deuc_known_vals_raw = []
                        this_fuel_deuc_known_vals = []

                        # ...our goal here: obtain final demand by fuel:
                        this_tech_fuel_ed_vals = []

                        # ...here, seek the EB proportion and keep it constant using "fuel demand" (_fd)
                        # ...we also need to extract the total fuel demand, which is the correct denominator
                        total_fuel_demand_baseyear = 0
                        total_fuel_demand_baseyear_2 = 0
                        for tech_internal in list_demand_sector_techs:
                            if fuel in list_fuel_orig:
                                total_fuel_demand_baseyear += \
                                    dict_base_energy_demand[tech_internal][fuel][base_year]
                        #
                        if fuel in list_fuel_orig:
                            num_fd = dict_base_energy_demand[tech][fuel][base_year]
                        else:
                            num_fd = 0
                        if total_sector_demand_baseyear != 0:
                            den_fd = total_sector_demand_baseyear
                        else:
                            den_fd = 1
                        endo_share = num_fd/den_fd
                        
                        dict_endo_sh.update({fuel:endo_share})

                        if this_fuel_deuc_projection == 'keep_proportions' or this_fuel_deuc_projection == 'electrify_sector_2_max':
                            y_count = 0
                            for y in time_vector:
                                # ...start by summing across all demands:
                                add_value = \
                                    endo_share*this_tech_ed_vals[y_count]
                                this_tech_fuel_ed_vals.append(add_value)

                                # ...be sure to add the fuel demand too:
                                dict_energy_demand_by_fuel[fuel][y_count] += add_value

                                y_count += 1

                        elif this_fuel_deuc_projection == 'redistribute':  # here we need to change the demand relationships by sector, in a smooth manner (i.e., interpolate)
                            mask_3_spec_tech = \
                                ((df_scen_3_spec['Demand sector'] == tech) &
                                 (df_scen_3_spec['Fuel'] == fuel))
                            df_redistribute_data = \
                                df_scen_3_spec.loc[mask_3_spec_tech]

                            try:
                                if df_redistribute_data['value'].iloc[0] == 'percent':
                                    check_percent = True
                            except Exception:
                                check_percent = False

                            this_fush_known_vals_raw = []
                            this_fush_known_vals = []
                            if check_percent is True:  # is compatible with "interpolate"
                                for y in time_vector:
                                    try:
                                        add_value = \
                                            df_redistribute_data[y].iloc[0]
                                    except Exception:
                                        add_value = 0

                                    this_fush_known_vals_raw.append(add_value)
                                    if type(add_value) is int:
                                        if math.isnan(add_value) is False:
                                            this_fush_known_vals.append(add_value/100)
                                        else:
                                            pass
                                    elif str(y) == str(base_year):
                                        this_fush_known_vals.append(endo_share)
                                    elif str(y) <= str(ini_simu_yr):
                                        this_fush_known_vals.append(endo_share)
                                    else:
                                        this_fush_known_vals.append('')

                                if add_value != 'rem':
                                    this_fush_vals = \
                                        interpolation_to_end(time_vector, 
                                                             ini_simu_yr,
                                                             this_fush_known_vals,
                                                             'last',
                                                             this_scen, '')

                                else:  # we need to fill later                               
                                    this_fush_vals = \
                                        [0 for y in range(len(time_vector))]
                                    store_fush_rem.update({fuel:this_fush_vals})
                                store_fush.update({fuel:this_fush_vals})

                                y_count = 0
                                for y in time_vector:
                                    add_value = \
                                        this_fush_vals[y_count]*this_tech_ed_vals[y_count]
                                    this_tech_fuel_ed_vals.append(add_value)
                                    dict_energy_demand_by_fuel[fuel][y_count] += add_value

                                    y_count += 1

                            if check_percent is not True:  # should do the same as in 'keep_proportions'
                                y_count = 0
                                for y in time_vector:
                                    add_value = \
                                        endo_share*this_tech_ed_vals[y_count]
                                    this_tech_fuel_ed_vals.append(add_value)
                                    dict_energy_demand_by_fuel[fuel][y_count] += add_value
                                    y_count += 1

                        dict_energy_demand[tech].update({fuel:this_tech_fuel_ed_vals})

                # ...here we need to run the remainder if necessary:
                if check_percent is True:
                    fuel_rem = list(store_fush_rem.keys())[0]
                    oneminus_rem_list_fush = store_fush_rem[fuel_rem]
                    for fuel in list_fuel:
                        if fuel != fuel_rem:
                            for y in range(len(time_vector)):
                                oneminus_rem_list_fush[y] += store_fush[fuel][y]

                    this_tech_fuel_ed_vals = []
                    for y in range(len(time_vector)):
                        store_fush[fuel_rem][y] = 1-oneminus_rem_list_fush[y]
                        add_value = \
                            store_fush[fuel_rem][y]*this_tech_ed_vals[y]
                        this_tech_fuel_ed_vals.append(add_value)
                        dict_energy_demand_by_fuel[fuel_rem][y] += add_value

                    dict_energy_demand[tech].update({fuel_rem:this_tech_fuel_ed_vals})

            """
            We can modify the *dict_energy_demand_by_fuel* and the 
            *dict_energy_demand* dictionaries to reflect the effect of the 
            measures!
            """
            
            """
            Store the list of columns for "df_scen_3_2":
            
            ['Rule',
             'Category',
             'Measure',
             'Country',
             'Scenario',
             'Share for subcategory',
             'Substituted vector',
             'Substitute vector',
             'Displaced energy per unit [MJ/unit]',
             'New energy requirement per unit [MJ/unit]',
             'Number of units',
             'Units',
             'Target years',
             'New displacement base year',
             'Displaced value before first target year',
             'Displaced value between target years',
             'Displaced value in target years',
             'Displaced value after last target year',
             'New requirement base year',
             'New requirement value before first target year',
             'New requirement value between target years',
             'New requirement value in target years',
             'New requirement value after last target year']
            
            """
            
            # print('stop before modifying the demand for end-use assumptions')
            # sys.exit()

            # 3e) Open the hydrogen efficiency
            param_related_15 = 'Green_Hydrogen_Prod_Eff'
            mask_15 = (df_scen_rc['Parameter'] == param_related_15)
            df_param_related_15 = df_scen_rc.loc[mask_15]
            df_param_related_15.reset_index(drop=True, inplace=True)

            list_h2_effic = list(df_param_related_15[time_vector].iloc[0])

            # Here we can modify the demand based on end-use assumptions:
            if len(df_scen_3_2.index.tolist()) > 0:
                print('Here we are modifying the demands according to "df_scen_3_2".')
                
                # Iterate across each applicable modification
                for m in range(len(df_scen_3_2.index.tolist())):
                    df32_rule = df_scen_3_2["Rule"].iloc[m]
                    df32_sh_cat = df_scen_3_2["Category"].iloc[m]
                    
                    
                    df32_sh_subcat_str = str(df_scen_3_2["Share for subcategory"].iloc[m]).split(' ; ')
                    df32_sh_subcat = [float(x) if isinstance(x, str) else x for x in df32_sh_subcat_str]
                    
                    df32_substituted = df_scen_3_2["Substituted vector"].iloc[m]
                    df32_substitute = df_scen_3_2["Substitute vector"].iloc[m]
                    
                    df32_depu_str = str(df_scen_3_2["Displaced energy per unit [MJ/unit]"].iloc[m]).split(" ; ")
                    df32_depu = [float(x) if isinstance(x, str) else x for x in df32_depu_str]
                    
                    df32_nerpu = df_scen_3_2["New energy requirement per unit [MJ/unit]"].iloc[m]
                    
                    df32_numunits_str = str(df_scen_3_2["Number of units"].iloc[m]).split(" ; ")
                    df32_numunits = [float(x) if isinstance(x, str) else x for x in df32_numunits_str]
                    
                    df32_units = df_scen_3_2["Units"].iloc[m]
                    df32_target_year_str = str(df_scen_3_2["Target years"].iloc[m])
                    
                    df32_ndby = df_scen_3_2["Displacement base year"].iloc[m]
                    df32_dvbfty = df_scen_3_2["Displaced value before first target year"].iloc[m]
                    df32_dvbetty = df_scen_3_2["Displaced value between target years"].iloc[m]
                    df32_dvty = df_scen_3_2["Displaced value in target years"].iloc[m]
                    df32_dvalty = df_scen_3_2["Displaced value after last target year"].iloc[m]
                    
                    df32_nrby = df_scen_3_2["New requirement base year"].iloc[m]
                    df32_nrvbfty = df_scen_3_2["New requirement value before first target year"].iloc[m]
                    df32_nrbetty = df_scen_3_2["New requirement value between target years"].iloc[m]
                    df32_nrty = df_scen_3_2["New requirement value in target years"].iloc[m]
                    df32_nralty = df_scen_3_2["New requirement value after last target year"].iloc[m]

                    add_vector_general =  [0] * len(time_vector)

                    if df32_rule == "Substitute energy per unit":
                        avector_counter = 0
                        df32_dvty_list_str = str(df32_dvty).split(' ; ')
                        df32_dvty_list = [float(x) if isinstance(x, str) else x for x in df32_dvty_list_str]
                        
                        for avector in df32_substituted.split(' ; '):
                            if avector != 'none':
                                base_disp_vector = deepcopy(dict_energy_demand[df32_sh_cat][avector])
                                base_disp_vector_agg = deepcopy(dict_energy_demand_by_fuel[avector])
                            else:
                                base_disp_vector = [0] * len(time_vector)
                                base_disp_vector_agg = [0] * len(time_vector)

                            if df32_substitute != 'none':
                                base_newreq_vector = deepcopy(dict_energy_demand[df32_sh_cat][df32_substitute])
                                base_newreq_vector_agg = deepcopy(dict_energy_demand_by_fuel[df32_substitute])

                            '''
                            print('/n')
                            print(avector_counter, avector)
                            print(base_disp_vector[:11])
                            '''

                            subtract_vector = [0] * len(time_vector)
                            
                            loc_depu = df32_depu[avector_counter]
                                                      
                            '''
                            Multiple target years can be defined in this step:
                            we will work only with one target year
                            '''
                            by_idx = time_vector.index(df32_ndby)
                            ty_list_str = df32_target_year_str.split(';')
                            ty_list = [int(x) if isinstance(x, str) else x for x in ty_list_str]
                            ty_index_list = []
                            ty_counter = 0
                            for ty in ty_list:
                                ty_index = time_vector.index(ty)
                                ty_index_list.append(ty_index)

                                subtract_vector[ty_index] = df32_dvty_list[ty_counter]/1e9
                                if base_disp_vector[ty_index] - df32_dvty_list[ty_counter]/1e9 < 0:
                                    print('The subtraction suggestion is not acceptable! Lower the displacement value.')
                                    sys.exit()
                                    
                                ty_counter += 1

                            # Here we fill the subtract vector with all the required targets
                            subsy_counter, ty_idx_counter = 0, 0
                            for subsy in time_vector:
                                if subsy_counter <= by_idx:
                                    # print('This should happen always 0', subsy)
                                    pass  # keep the list as zero
                                    
                                elif subsy_counter > by_idx and subsy < ty_list[0] and df32_dvbfty == 'interp':
                                    # print('This should happen always 1', subsy)
                                    delta_den = ty_index_list[0] - by_idx
                                    delta_num = subtract_vector[ty_index_list[0]]
                                    delta = delta_num/delta_den
                                    subtract_vector[subsy_counter] = subtract_vector[subsy_counter-1] + delta
                                    
                                elif subsy in ty_list:
                                    # print('This should happen always 2', subsy)
                                    
                                    ty_idx_counter += 1
                                    
                                elif subsy_counter > by_idx and subsy > ty_list[0] and subsy < ty_list[-1] and df32_dvbetty == 'interp':
                                    # print('This should happen only with multiple target years', subsy)
                                    
                                    # print('NOTE: ', ty_idx_counter, subtract_vector[subsy_counter-1], subtract_vector[ty_index_list[ty_idx_counter - 1]], subtract_vector[ty_index_list[ty_idx_counter]])
                                    
                                    tys_idxs_last = ty_index_list[ty_idx_counter - 1]
                                    tys_idxs_this = ty_index_list[ty_idx_counter]
                                    delta_den = tys_idxs_this - tys_idxs_last
                                    delta_num = subtract_vector[ty_index_list[ty_idx_counter]] - subtract_vector[ty_index_list[ty_idx_counter - 1]]
                                    delta = delta_num/delta_den
                                    subtract_vector[subsy_counter] = subtract_vector[subsy_counter-1] + delta
                                    
                                elif subsy_counter > by_idx and subsy > ty_list[-1] and df32_nralty == 'flat': 
                                    # print('This should happen always 3', subsy)
                                    subtract_vector[subsy_counter] = subtract_vector[subsy_counter - 1]

                                else:
                                    # print('No specified interpolation condition found for displacement formulas!')
                                    sys.exit()
                                    
                                subsy_counter += 1

                            avector_counter += 1

                            # After having obtained the subtract vector, we need an add vector using the substitution ratios:
                            if avector != 'none':
                                rep_fac_ene = df32_nerpu/loc_depu
                                add_vector_local = [rep_fac_ene*substval for substval in subtract_vector]
                            else:
                                print('This functionality is not available')
                                sys.exit()

                            # Store how much of the new vector is needed in total
                            add_vector_general = [a + b for a, b in zip(add_vector_general, add_vector_local)]

                            # Change the vector that suffered a subtraction, locally and in aggregate:
                            changed_disp_vector = [a - b for a, b in zip(base_disp_vector, subtract_vector)]
                            change_disp_vector_agg = [a - b for a, b in zip(base_disp_vector_agg, subtract_vector)]
                            # Update the corresponding dictionaries:
                            if df32_substituted != df32_substitute:
                                dict_energy_demand[df32_sh_cat][avector] = deepcopy(changed_disp_vector)
                                dict_energy_demand_by_fuel[avector] = deepcopy(change_disp_vector_agg)
                            else:
                                print('Happens for increase in energy demand.')

                            # Change the vector that needed more energy, just locally:
                            changed_newreq_vector = [a + b for a, b in zip(base_newreq_vector, add_vector_local)]
                            # Update the corresponding dictionary:
                            if df32_substitute != 'none':
                                dict_energy_demand[df32_sh_cat][df32_substitute] = deepcopy(changed_newreq_vector)

                            # print('have gotten here 1')
                            # sys.exit()

                            '''

                            print('/n')
                            print(subtract_vector[:11])
                            print(changed_disp_vector[:11])

                            print('/n')
                            print(df32_substitute, rep_fac_ene, df32_nerpu, loc_depu)
                            print(add_vector_local[:11])
                            print(base_newreq_vector[:11])
                            print(changed_newreq_vector[:11])
                            if df32_substitute != 'none':
                                print(dict_energy_demand[df32_sh_cat][df32_substitute][:11])
                            print('------------------')
                            print('\n')

                            '''

                        # Change the vector that needed more energy, just in aggregate:
                        change_newreq_vector_agg = [a + b for a, b in zip(base_newreq_vector_agg, add_vector_general)]
                        # Update the corresponding dictionary:
                        if df32_substitute != 'none':
                            dict_energy_demand_by_fuel[df32_substitute] = deepcopy(change_newreq_vector_agg)


                    if df32_rule == "Substitute energy share":
                        avector_counter = 0
                        for avector in df32_substituted.split(' ; '):
                            base_disp_vector = deepcopy(dict_energy_demand[df32_sh_cat][avector])
                            base_disp_vector_agg = deepcopy(dict_energy_demand_by_fuel[avector])

                            '''
                            print('/n')
                            print(avector_counter, avector)
                            print(base_disp_vector[:11])
                            '''

                            if df32_substitute != 'none':
                                if df32_substitute == 'Hydrogen':
                                    df32_substitute_orig = 'Hydrogen'
                                    df32_substitute = 'Electricity'
                                else:
                                    df32_substitute_orig = df32_substitute
                                base_newreq_vector = deepcopy(dict_energy_demand[df32_sh_cat][df32_substitute])
                                base_newreq_vector_agg = deepcopy(dict_energy_demand_by_fuel[df32_substitute])
                            else:
                                base_newreq_vector = [0] * len(time_vector)
                                base_newreq_vector_agg = [0] * len(time_vector)

                            subtract_vector = [0] * len(time_vector)
                            
                            sh_subcat = df32_sh_subcat[avector_counter]
                            loc_depu = df32_depu[avector_counter]
                                                      
                            '''
                            Multiple target years can be defined in this step:
                            we will work only with one target year
                            '''
                            by_idx = time_vector.index(df32_ndby)
                            ty_list_str = df32_target_year_str.split(';')
                            ty_list = [int(x) if isinstance(x, str) else x for x in ty_list_str]
                            ty_index_list = []
                            ty_counter = 0
                            for ty in ty_list:
                                ty_index = time_vector.index(ty)
                                ty_index_list.append(ty_index)

                                tval = df32_numunits[ty_counter]/100
                            
                                subtract_vector[ty_index] = base_disp_vector[ty_index]*sh_subcat*tval
                                
                                # print(ty_index, base_disp_vector[ty_index], sh_subcat, tval)

                                ty_counter += 1

                            # Here we fill the subtract vector with all the required targets
                            subsy_counter, ty_idx_counter = 0, 0
                            for subsy in time_vector:
                                if subsy_counter <= by_idx:
                                    # print('This should happen always 0', subsy)
                                    pass  # keep the list as zero
                                    
                                elif subsy_counter > by_idx and subsy < ty_list[0] and df32_dvbfty == 'interp':
                                    # print('This should happen always 1', subsy)
                                    delta_den = ty_index_list[0] - by_idx
                                    delta_num = subtract_vector[ty_index_list[0]]
                                    delta = delta_num/delta_den
                                    subtract_vector[subsy_counter] = subtract_vector[subsy_counter-1] + delta
                                    
                                elif subsy in ty_list and df32_dvty == 'endo':
                                    # print('This should happen always 2', subsy)
                                    
                                    ty_idx_counter += 1
                                    
                                elif subsy_counter > by_idx and subsy > ty_list[0] and subsy < ty_list[-1] and df32_dvbetty == 'interp':
                                    # print('This should happen only with multiple target years', subsy)
                                    
                                    # print('NOTE: ', ty_idx_counter, subtract_vector[subsy_counter-1], subtract_vector[ty_index_list[ty_idx_counter - 1]], subtract_vector[ty_index_list[ty_idx_counter]])
                                    
                                    tys_idxs_last = ty_index_list[ty_idx_counter - 1]
                                    tys_idxs_this = ty_index_list[ty_idx_counter]
                                    delta_den = tys_idxs_this - tys_idxs_last
                                    delta_num = subtract_vector[ty_index_list[ty_idx_counter]] - subtract_vector[ty_index_list[ty_idx_counter - 1]]
                                    delta = delta_num/delta_den
                                    subtract_vector[subsy_counter] = subtract_vector[subsy_counter-1] + delta
                                    
                                elif subsy_counter > by_idx and subsy > ty_list[-1] and df32_nralty == 'flat': 
                                    # print('This should happen always 3', subsy)
                                    subtract_vector[subsy_counter] = base_disp_vector[subsy_counter]*sh_subcat*tval

                                else:
                                    # print('No specified interpolation condition found for displacement formulas!')
                                    sys.exit()
                                    
                                subsy_counter += 1

                            avector_counter += 1

                            # After having obtained the subtract vector, we need an add vector using the substitution ratios:
                            rep_fac_ene = df32_nerpu/loc_depu
                            add_vector_local = [rep_fac_ene*substval for substval in subtract_vector]
                            if df32_substitute_orig == 'Hydrogen':
                                add_vector_local = [v/(list_h2_effic[i]/100) for i, v in enumerate(add_vector_local)]

                            # Store how much of the new vector is needed in total
                            add_vector_general = [a + b for a, b in zip(add_vector_general, add_vector_local)]

                            # Change the vector that suffered a subtraction, locally and in aggregate:
                            changed_disp_vector = [a - b for a, b in zip(base_disp_vector, subtract_vector)]
                            change_disp_vector_agg = [a - b for a, b in zip(base_disp_vector_agg, subtract_vector)]
                            # Update the corresponding dictionaries:
                            dict_energy_demand[df32_sh_cat][avector] = deepcopy(changed_disp_vector)
                            dict_energy_demand_by_fuel[avector] = deepcopy(change_disp_vector_agg)
                            

                            # Change the vector that needed more energy, just locally:
                            changed_newreq_vector = [a + b for a, b in zip(base_newreq_vector, add_vector_local)]
                            # Update the corresponding dictionary:
                            if df32_substitute != 'none':
                                dict_energy_demand[df32_sh_cat][df32_substitute] = deepcopy(changed_newreq_vector)

                            # print('have gotten here 1')
                            # sys.exit()
                            '''

                            print('/n')
                            print(subtract_vector[:11])
                            print(changed_disp_vector[:11])

                            print('/n')
                            print(df32_substitute, rep_fac_ene, df32_nerpu, loc_depu)
                            print(add_vector_local[:11])
                            print(base_newreq_vector[:11])
                            print(changed_newreq_vector[:11])
                            if df32_substitute != 'none':
                                print(dict_energy_demand[df32_sh_cat][df32_substitute][:11])
                            print('------------------')
                            print('\n')

                            '''

                        # Change the vector that needed more energy, just in aggregate:
                        change_newreq_vector_agg = [a + b for a, b in zip(base_newreq_vector_agg, add_vector_general)]
                        # Update the corresponding dictionary:
                        if df32_substitute != 'none':
                            dict_energy_demand_by_fuel[df32_substitute] = deepcopy(change_newreq_vector_agg)

                        # print('have gotten here 2')
                        # sys.exit()
                        
                       
                        
                        

                print('        > The modifications are complete!')
                # sys.exit()


            # Here we can modify the demand from the endogenous calculation if the electrical demand needs an override
            param_related_13 = 'Fixed electricity production'
            mask_13 = (df_scen_rc['Parameter'] == param_related_13)
            df_param_related_13 = df_scen_rc.loc[mask_13]
            df_param_related_13.reset_index(drop=True, inplace=True)
            if df_param_related_13['projection'].iloc[0] != 'ignore':
                dict_energy_demand_ref = deepcopy(dict_energy_demand)
                dict_energy_demand_by_fuel_ref = deepcopy(dict_energy_demand_by_fuel)

                # Grab the electricity demand from the base year:
                dem_elec_tot_ref = deepcopy(dict_energy_demand_by_fuel_ref['Electricity'])
                dem_elec_tot_ref_by = dem_elec_tot_ref[0]

                # This means that we need to override the electrical demand by estimating the adjustment across sectors
                if df_param_related_13['projection'].iloc[0] == 'interpolate':
                    # Iterate across all years and determine:
                    #   Years that need to be filled according to interpolation
                    #   Years with an exact value that need to be added into the list
                    #   Years that need to follow the growth of the previous trajectory

                    known_vals = []
                    known_yrs = []
                    find_yrs = []
                    use_vals, use_vals_ratio = [], []

                    if df_param_related_13['Unit'].iloc[0] == 'GWh':
                        gwh_to_pj = 0.0036
                    else:
                        gwh_to_pj = 1

                    for y in range(len(time_vector)):
                        this_13_val = df_param_related_13[time_vector[y]].iloc[0]
                        if y > 0:
                            last_13_val = df_param_related_13[time_vector[y-1]].iloc[0]
                            change_13_val = (dem_elec_tot_ref[y] - dem_elec_tot_ref[y-1]) / dem_elec_tot_ref[y-1]
                        
                        # print(this_13_val, type(this_13_val), time_vector[y])
                        
                        if y == 0:
                            known_vals.append(dem_elec_tot_ref_by)
                            known_yrs.append(time_vector[y])
                            use_vals.append(dem_elec_tot_ref_by)
                            use_vals_ratio.append(1)
                            # ('happen 1', time_vector[y])
                        elif isinstance(this_13_val, (float, np.floating, int)) and not np.isnan(this_13_val):
                            this_13_val *= gwh_to_pj
                            known_vals.append(this_13_val)
                            known_yrs.append(time_vector[y])
                            use_vals.append(this_13_val)
                            ratio_adjust = this_13_val/dem_elec_tot_ref[y]
                            ratio_adjust_idx = deepcopy(y)
                            use_vals_ratio.append(ratio_adjust)
                            # print('happen 2', time_vector[y])
                        elif type(this_13_val) is not str:
                            if math.isnan(this_13_val):
                                find_yrs.append(time_vector[y])
                                use_vals.append('')
                                use_vals_ratio.append('')
                                # print('happen 3', time_vector[y])
                        elif this_13_val == 'follow_gdp_intensity': # assume that the last known value must be continued
                            add_13_val = (1 + change_13_val)*known_vals[-1]
                            known_vals.append(add_13_val)
                            known_yrs.append(time_vector[y])
                            use_vals.append(add_13_val)
                            use_vals_ratio.append(1)
                            # print('happen 4', time_vector[y])

                    interp_vals_linear = \
                        interpolation_to_end(time_vector, 2021, use_vals,
                                             'last', this_scen, '')
                    interp_mults_linear = \
                        interpolation_to_end(time_vector, 2021, use_vals_ratio,
                                             'last', this_scen, '')
                    interp_vals_spirit = [interp_mults_linear[y]*dem_elec_tot_ref[y] if y < ratio_adjust_idx else interp_vals_linear[y] for y in range(len(time_vector))]
                       
                    # Let's use the spirit to calculate the ratio between new and old demand
                    adj_ele_ratio_list = [ivs / det if det != 0 else 0 for ivs, det in zip(interp_vals_spirit, dem_elec_tot_ref)]

                    # Now we need to update the electricity demand for each sector:
                    for sec in list(dict_energy_demand.keys()):
                        ref_ele_list = deepcopy(dict_energy_demand[sec]['Electricity'])
                        result_list = [ref * adj for ref, adj in zip(ref_ele_list, adj_ele_ratio_list)]
                        dict_energy_demand[sec]['Electricity'] = deepcopy(result_list)

                    ref_ele_list_tot = [ref * adj for ref, adj in zip(dem_elec_tot_ref, adj_ele_ratio_list)]
                    dict_energy_demand_by_fuel['Electricity'] = deepcopy(ref_ele_list_tot)

                    # print('stop here until the electricity demands have been updated - inside')
                    # sys.exit()


            # print('stop here until the electricity demands have been updated')
            # sys.exit()

            # We can continue modifying the demands if a specific type of projection requires to increase the ambition:
            if this_fuel_deuc_projection == 'electrify_sector_2_max':
                for sec in list(dict_energy_demand.keys()):
                    ref_afuel_list_sum = [0] * len(time_vector)
                    # print('\n')
                    # print(sec)
                    #print(list(dict_energy_demand[sec].keys()))
                    
                    for afuel in list(dict_energy_demand[sec].keys()):
                        if 'Total' not in afuel and 'Non-' not in afuel:
                            ref_afuel_list = deepcopy(dict_energy_demand[sec][afuel])
                            ref_afuel_list_sum = [a + b for a, b in zip(ref_afuel_list_sum, ref_afuel_list)]
                    
                    if sum(ref_afuel_list_sum) > 0:
                        # Act upon electrification, i.e., the share:
                        elec_sh = [100*a / b for a, b in zip(dict_energy_demand[sec]['Electricity'], ref_afuel_list_sum)]
                        elec_sh_orig = deepcopy(elec_sh)
                        if sum(elec_sh) == 0:
                            elec_sh = [0.01] * len(time_vector)
                            dict_energy_demand[sec]['Electricity'] = \
                                [a * b/100 for a, b in zip(elec_sh, ref_afuel_list_sum)]
                        for y in range(len(time_vector)):
                            if float(elec_sh[y]) == 0.0:
                                elec_sh[y] = 0.001  # add this as a buffer for errors
                                # print(y, 'happens?')
                        
                        # The rest total:
                        rest_sh = [100 - a for a in elec_sh]
                        
                        if elec_sh[-1] < this_fuel_deuc_value_type:
                            # print('Are things OK or Wrong?')
                            mult_empty_norm = ['' for y in range(len(time_vector))]
                            

                            
                            mult_empty_norm[0], mult_empty_norm[1], mult_empty_norm[2] = elec_sh[0], elec_sh[1], elec_sh[2]
                            mult_empty_norm[3], mult_empty_norm[4], mult_empty_norm[5] = elec_sh[3], elec_sh[4], elec_sh[5]
                            mult_empty_norm[6], mult_empty_norm[7], mult_empty_norm[8] = elec_sh[6], elec_sh[7], elec_sh[8]
                            mult_empty_norm[9] = elec_sh[9]
                            mult_empty_norm[-1] = this_fuel_deuc_value_type
                            mult_interp_norm = \
                                interpolation_to_end(time_vector, ini_simu_yr, \
                                mult_empty_norm, 'last', this_scen, '')
    
                            mult_interp_norm_comp = \
                                [a - b for a, b in zip([100]*len(time_vector), mult_interp_norm)]
    
                            adj_ratio_rest = \
                                [a / b for a, b in zip(mult_interp_norm_comp, rest_sh)]
    
                            # We must find a multiplier for the last year, and interpolate starting in 2023
                            adj_ratio_ele = \
                                [a / b for a, b in zip(mult_interp_norm, elec_sh)]
    
   
                            '''
                            last_val_mult = 75/elec_sh[-1]
                            mult_empty = ['' for y in range(len(time_vector))]
                            mult_empty[0], mult_empty[1], mult_empty[2] = 1, 1, 1
                            mult_empty[-1] = last_val_mult
                            mult_interp = \
                                interpolation_to_end(time_vector, 2023, \
                                mult_empty, 'last', this_scen, '')
                            '''
    
                            orig_list = dict_energy_demand[sec]['Electricity']
                            dict_energy_demand[sec]['Electricity'] = \
                                [a * b for a, b in zip(adj_ratio_ele, orig_list)]

                            # if this_country_2 == 'Colombia' and sec == 'Transport':
                            # if this_country_2 == 'Costa Rica' and sec == 'Transport':
                            #    print('rev')
                            #    sys.exit()
    
                            # The rest of the share:
                            for afuel in list(dict_energy_demand[sec].keys()):
                                if 'Total' not in afuel and 'Non-' not in afuel and afuel != 'Electricity':
                                    afuel_sh = [100*a / b for a, b in zip(dict_energy_demand[sec][afuel], ref_afuel_list_sum)]  # may be unnecessary
                                    orig_list = dict_energy_demand[sec][afuel]
                                    dict_energy_demand[sec][afuel] = \
                                        [a * b for a, b in zip(adj_ratio_rest, orig_list)]
                                    
                            # Estimate the new total
                            new_total_sum_list = [0] * len(time_vector)
                            for afuel in list(dict_energy_demand[sec].keys()):
                                if 'Total' not in afuel and 'Non-' not in afuel:
                                    new_total_sum_list = \
                                        [a + b for a, b in zip(dict_energy_demand[sec][afuel], new_total_sum_list)]
                                    
                            # Calculate the new difference between totals after adjustment:
                            diff_list = [100*(a-b)/a for a, b in zip(ref_afuel_list_sum, new_total_sum_list)]
                            if sum(diff_list)/len(diff_list) > 1:
                                print('Algorithm does not work.')
                                print(diff_list)
                                sys.exit()
                            
                            # print('review this')
                            # sys.exit()

                # Now update the demands per fuel:
                for afuel in list(dict_energy_demand_by_fuel.keys()):
                    afuel_sum_list = [0] * len(time_vector)
                    for sec in list(dict_energy_demand.keys()):
                        afuel_sum_list = \
                            [a + b for a, b in zip(afuel_sum_list, dict_energy_demand[sec][afuel])]
                        dict_energy_demand_by_fuel[afuel] = deepcopy(afuel_sum_list)
            
            # print('We got until here')
            #sys.exit()

            # Now, let's store the energy demand projections in the country dictionary:
            # parameters: 'Energy demand by sector', 'Energy intensity by sector'
            dict_local_country.update({this_country:{'Energy demand by sector': dict_energy_demand}})
            dict_local_country[this_country].update({'Energy intensity by sector': dict_energy_intensity})
            dict_local_country[this_country].update({'Energy demand by fuel': dict_energy_demand_by_fuel})

            """
            INSTRUCTIONS:
            1) Perform the transport demand calculations
            2) Check the demand component: rewrite the transport energy demand projection
            3) Store the demand and print the energy demand difference
            """
            #print('Rewrite the demand component here')
            types_all = []  # this is defined here to avoid an undefined variable

            # Define the dictionary that stores emissions here to keep
            # emissions from transport technologies.
            # 'emission factors':
            mask_emission_factors = (df4_ef['Type'] == 'Standard')
            this_df4_ef = df4_ef.loc[mask_emission_factors]
            this_df4_ef.reset_index(drop=True, inplace=True)
            this_df4_ef_fuels = \
                this_df4_ef['Fuel'].tolist()
    
            emissions_fuels_list = []
            emissions_fuels_dict = {}
            for f in range(len(this_df4_ef_fuels)):
                this_fuel = this_df4_ef_fuels[f]
                this_proj = this_df4_ef.iloc[f]['Projection']
                base_emission_val = this_df4_ef.iloc[f][time_vector[0]]
                if this_proj == 'flat':
                    list_emission_year = [base_emission_val for y in range(len(time_vector))]
                emissions_fuels_list.append(this_fuel)
                emissions_fuels_dict.update({this_fuel:list_emission_year})

            emissions_demand = {}  # crucial output
            if overwrite_transport_model:
                # We must edit the "dict_energy_demand" content with the transport modeling
                dict_energy_demand_trn = deepcopy(dict_energy_demand['Transport'])
                transport_fuel_sets = \
                    df2_trans_sets_eq['Transport Fuel'].tolist()
                transport_scenario_sets = \
                    df2_trans_sets_eq['Energy Fuel'].tolist()
                dict_eq_transport_fuels = {}
                for te in range(len(transport_fuel_sets)):
                    t_fuel_set = transport_fuel_sets[te]
                    t_scen_set = transport_scenario_sets[te]
                    dict_eq_transport_fuels.update({t_fuel_set:t_scen_set})
    
                # NOTE: we must edit the projection from "dict_energy_demand" first;
                # once that is complete, we must edit the "dict_energy_demand_by_fuel" accordingly.
    
                # Now we must follow standard transport equations and estimations:
                # TM 1) Estimate the demand projection for the country
                """
                Demand_Passenger = sum_i (km_passenger_i * fleet_passenger_i * load_factor_i)
                """
                # Store load factor and kilometers:
                dict_lf, dict_km = {}, {}
                #
                # 1.a) estimate the demand in the base year
    
                set_pass_trn = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl1_u_raw)
                    if v == 'Passenger']
    
                set_pass_trn_dem_by = {}
                set_pass_trn_fleet_by = {}
                set_pass_trn_dem_sh = {}
    
                for n in range(len(set_pass_trn)):
                    set_pass_trn_dem_by.update({set_pass_trn[n]:0})
                    set_pass_trn_fleet_by.update({set_pass_trn[n]:0})
                    set_pass_trn_dem_sh.update({set_pass_trn[n]:0})
    
                # Select the scenario:
                mask_select_trn_scen = \
                    (df3_tpt_data['Scenario'] == this_scen) | (df3_tpt_data['Scenario'] == 'ALL')
                df_trn_data = df3_tpt_data.loc[mask_select_trn_scen]
                df_trn_data.reset_index(drop=True, inplace=True)
    
                sum_pass_trn_dem_by = 0  # Gpkm
                for spt in set_pass_trn:
                    mask_fby_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Residual fleet')
                    fby_spt = df_trn_data.loc[mask_fby_spt][per_first_yr].sum()
    
                    mask_km_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Distance')
                    km_spt = df_trn_data.loc[mask_km_spt][per_first_yr].sum()
    
                    mask_lf_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Load Factor')
                    lf_spt = df_trn_data.loc[mask_lf_spt][per_first_yr].sum()
    
                    dict_lf.update({spt: deepcopy(lf_spt)})
                    dict_km.update({spt: deepcopy(km_spt)})
    
                    set_pass_trn_fleet_by[spt] = fby_spt
                    set_pass_trn_dem_by[spt] = fby_spt*km_spt*lf_spt/1e9
                    sum_pass_trn_dem_by += fby_spt*km_spt*lf_spt/1e9
    
                for spt in set_pass_trn:
                    set_pass_trn_dem_sh[spt] = 100*set_pass_trn_dem_by[spt]/sum_pass_trn_dem_by
    
                set_fre_trn = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl1_u_raw)
                    if v == 'Freight']
    
                set_fre_trn_dem_by = {}
                set_fre_trn_fleet_by = {}
                set_fre_trn_dem_sh = {}
    
                for n in range(len(set_fre_trn)):
                    set_fre_trn_dem_by.update({set_fre_trn[n]:0})
                    set_fre_trn_fleet_by.update({set_fre_trn[n]:0})
                    set_fre_trn_dem_sh.update({set_fre_trn[n]:0})
    
                sum_fre_trn_dem_by = 0  # Gtkm
                for spt in set_fre_trn:
                    mask_fby_spt = (df_trn_data['Type'] == spt) & \
                                    (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                    (df_trn_data['Parameter'] == 'Residual fleet')
                    fby_spt = df_trn_data.loc[mask_fby_spt][per_first_yr].sum()
    
                    mask_km_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Distance')
                    km_spt = df_trn_data.loc[mask_km_spt][per_first_yr].sum()
    
                    mask_lf_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Load Factor')
                    lf_spt = df_trn_data.loc[mask_lf_spt][per_first_yr].sum()
    
                    dict_lf.update({spt: deepcopy(lf_spt)})
                    dict_km.update({spt: deepcopy(km_spt)})
    
                    set_fre_trn_fleet_by[spt] = fby_spt
                    set_fre_trn_dem_by[spt] = fby_spt*km_spt*lf_spt/1e9
                    sum_fre_trn_dem_by += fby_spt*km_spt*lf_spt/1e9
    
                for spt in set_fre_trn:
                    set_fre_trn_dem_sh[spt] = 100*set_fre_trn_dem_by[spt]/sum_fre_trn_dem_by
    
                # 1.b) estimate the demand growth
                # for this we need to extract a couple of variables from
                # the "df_trn_data":
                # Elasticities:
                projtype_ela_pas, mask_ela_pas = \
                    fun_dem_model_projtype('Passenger', this_country, 'Elasticity',
                                        'projection', df_trn_data)
                list_ela_pas = \
                    fun_dem_proj(time_vector, projtype_ela_pas,
                                mask_ela_pas, df_trn_data)
    
                projtype_ela_fre, mask_ela_fre = \
                    fun_dem_model_projtype('Freight', this_country, 'Elasticity',
                                        'projection', df_trn_data)
                list_ela_fre = \
                    fun_dem_proj(time_vector, projtype_ela_fre,
                                mask_ela_fre, df_trn_data)
    
                projtype_ela_oth, mask_ela_oth = \
                    fun_dem_model_projtype('Other', this_country, 'Elasticity',
                                        'projection', df_trn_data)
                list_ela_oth = \
                    fun_dem_proj(time_vector, projtype_ela_oth,
                                mask_ela_oth, df_trn_data)
    
                # Demands:
                projtype_dem_pas, mask_dem_pas = \
                    fun_dem_model_projtype('Passenger', this_country, 'Demand',
                                        'projection', df_trn_data)
                if 'endogenous' not in projtype_dem_pas:
                    pass_trn_dem = \
                        fun_dem_proj(time_vector, projtype_dem_pas,
                                    mask_dem_pas, df_trn_data)
    
                projtype_dem_fre, mask_dem_fre = \
                    fun_dem_model_projtype('Freight', this_country, 'Demand',
                                        'projection', df_trn_data)
                if 'endogenous' not in projtype_dem_fre:
                    fre_trn_dem = \
                        fun_dem_proj(time_vector, projtype_dem_fre,
                                    mask_dem_fre, df_trn_data)
    
                projtype_dem_oth, mask_dem_oth = \
                    fun_dem_model_projtype('Other', this_country, 'Demand',
                                        'projection', df_trn_data)
                '''
                Note: "Other" [transport demands] is a category currently
                unused
                '''
    
                if 'endogenous' in projtype_dem_pas:
                    pass_trn_dem = [0 for y in range(len(time_vector))]
                if 'endogenous' in projtype_dem_fre:
                    fre_trn_dem = [0 for y in range(len(time_vector))]
                # We must project transport demand here.
                for y in range(len(time_vector)):
                    if y == 0:
                        pass_trn_dem[y] = sum_pass_trn_dem_by
                        fre_trn_dem[y] = sum_fre_trn_dem_by
                    else:
                        gdp_gr = this_gdp_growth_vals[y]/100
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        pop_gr = this_pop_growth_vals[y]/100
    
                        if projtype_dem_pas == 'endogenous_gdp':
                            trn_gr_pas = 1 + (gdp_gr*list_ela_pas[y])
                        
                        if projtype_dem_fre == 'endogenous_gdp':
                            trn_gr_fre = 1 + (gdp_gr*list_ela_fre[y])
    
                        if projtype_dem_pas == 'endogenous_gdp_pc':
                            trn_gr_pas = 1 + (gdp_pc_gr*list_ela_pas[y])
                        
                        if projtype_dem_fre == 'endogenous_gdp_pc':
                            trn_gr_fre = 1 + (gdp_pc_gr*list_ela_fre[y])
    
                        if projtype_dem_pas == 'endogenous_pop':
                            trn_gr_pas = 1 + (pop_gr*list_ela_pas[y])
                        
                        if projtype_dem_fre == 'endogenous_pop':
                            trn_gr_fre = 1 + (pop_gr*list_ela_fre[y])
    
                        if 'endogenous' in projtype_dem_pas:
                            pass_trn_dem[y] = trn_gr_pas*pass_trn_dem[y-1]
                        if 'endogenous' in projtype_dem_fre:
                            fre_trn_dem[y] = trn_gr_fre*fre_trn_dem[y-1]
    
                # 1.c) apply the mode shift and non-motorized parameters:
                set_pass_trn_priv = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl2_u_raw)
                    if v == 'Private']
                set_pass_trn_pub = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl2_u_raw)
                    if v == 'Public']
    
                pass_trn_dem_sh_private = 0  # this should be an integer
                for n in range(len(set_pass_trn_priv)):
                    pass_trn_dem_sh_private += \
                        set_pass_trn_dem_sh[set_pass_trn_priv[n]]
    
                pass_trn_dem_sh_public = 0
                for n in range(len(set_pass_trn_pub)):
                    pass_trn_dem_sh_public += \
                        set_pass_trn_dem_sh[set_pass_trn_pub[n]]
    
                pass_trn_dem_sh_private_k = {}
                for n in range(len(set_pass_trn_priv)):
                    this_sh_k = set_pass_trn_dem_sh[set_pass_trn_priv[n]]
                    this_sh_k_adj = \
                        100*this_sh_k/pass_trn_dem_sh_private
                    add_sh_k = {set_pass_trn_priv[n]:this_sh_k_adj}
                    pass_trn_dem_sh_private_k.update(deepcopy(add_sh_k))
    
                pass_trn_dem_sh_public_k = {}
                for n in range(len(set_pass_trn_pub)):
                    this_sh_k = set_pass_trn_dem_sh[set_pass_trn_pub[n]]
                    this_sh_k_adj = \
                        100*this_sh_k/pass_trn_dem_sh_public
                    add_sh_k = {set_pass_trn_pub[n]:this_sh_k_adj}
                    pass_trn_dem_sh_public_k.update(deepcopy(add_sh_k))
    
                # ...the goal is to have a list of participation per type:
                gpkm_pri_k = {}
                for n in range(len(set_pass_trn_priv)):
                    gpkm_pri_k.update({set_pass_trn_priv[n]:[]})
                gpkm_pub_k = {}
                for n in range(len(set_pass_trn_pub)):
                    gpkm_pub_k.update({set_pass_trn_pub[n]:[]})
                gpkm_nonmot = []
    
                list_mode_shift = []
                mask_mode_shift = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Mode shift')
                list_non_motorized = []
                mask_non_motorized = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                    (df_trn_data['Parameter'] == 'Non-motorized transport')
                for y in range(len(time_vector)):
                    list_mode_shift.append(df_trn_data.loc[mask_mode_shift][time_vector[y]].iloc[0])
                    list_non_motorized.append(df_trn_data.loc[mask_non_motorized][time_vector[y]].iloc[0])
    
                    # Private types:
                    this_gpkm_priv = \
                        pass_trn_dem[y]*(pass_trn_dem_sh_private-list_mode_shift[-1]-list_non_motorized[-1])/100
                    for n in range(len(set_pass_trn_priv)):
                        this_sh_k_adj = \
                            pass_trn_dem_sh_private_k[set_pass_trn_priv[n]]
                        this_gpkm_k = \
                            this_gpkm_priv*this_sh_k_adj/100
                        gpkm_pri_k[set_pass_trn_priv[n]].append(this_gpkm_k)
                        
                    """
                    NOTE: the share of autos and motos relative to private stays constant (units: pkm)
                    """
                    # Public types:
                    this_gpkm_pub = pass_trn_dem[y]*(pass_trn_dem_sh_public+list_mode_shift[-1])/100
                    for n in range(len(set_pass_trn_pub)):
                        this_sh_k_adj = \
                            pass_trn_dem_sh_public_k[set_pass_trn_pub[n]]
                        this_gpkm_k = \
                            this_gpkm_pub*this_sh_k_adj/100
                        gpkm_pub_k[set_pass_trn_pub[n]].append(this_gpkm_k)
    
                    # Non-mot types:
                    this_gpkm_nonmot = pass_trn_dem[y]*(list_non_motorized[-1])
                    gpkm_nonmot.append(this_gpkm_nonmot)
    
                # 1.d) apply the logistics parameters:
                gtkm_freight_k = {}
                for n in range(len(set_fre_trn)):
                    gtkm_freight_k.update({set_fre_trn[n]: []})
    
                list_logistics = []
                mask_logistics = \
                    (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                    (df_trn_data['Parameter'] == 'Logistics')
                for y in range(len(time_vector)):
                    list_logistics.append(df_trn_data.loc[mask_logistics][time_vector[y]].iloc[0])
    
                    for n in range(len(set_fre_trn)):                    
                        this_fre_sh_k = set_fre_trn_dem_sh[set_fre_trn[n]]
                        this_fre_k = fre_trn_dem[y]*this_fre_sh_k/100
                        gtkm_freight_k[set_fre_trn[n]].append(this_fre_k)
    
                # TM 2) Estimate the required energy for transport
                """
                Paso 1: obtener el % de flota por fuel de cada carrocería
                """
                # A dictionary with the residual fleet will come in handy:
                dict_resi_cap_trn = {}
    
                # Continue distributing the fleet:
                types_pass = set_pass_trn
                types_fre = set_fre_trn
                fuels = transport_fuel_sets
                fuels_nonelectric = [i for i in transport_fuel_sets if
                                    i not in ['ELECTRICIDAD', 'HIDROGENO']]
                set_pass_trn_fleet_by_sh = {}
                for t in types_pass:
                    dict_resi_cap_trn.update({t:{}})
                    total_type = set_pass_trn_fleet_by[t]
                    set_pass_trn_fleet_by_sh.update({t:{}})
                    for f in fuels:
                        dict_resi_cap_trn[t].update({f:[]})
                        mask_tf = (df_trn_data['Type'] == t) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Residual fleet') & \
                                (df_trn_data['Fuel'] == f)
                        fby_tf = df_trn_data.loc[mask_tf][per_first_yr].iloc[0]
                        set_pass_trn_fleet_by_sh[t].update({f:100*fby_tf/total_type})
                        for y in range(len(time_vector)):
                            a_fleet = \
                                df_trn_data.loc[mask_tf][time_vector[y]].iloc[0]
                            dict_resi_cap_trn[t][f].append(a_fleet)
    
                set_fre_trn_fleet_by_sh = {}
                fuels_fre = []
                for t in types_fre:
                    dict_resi_cap_trn.update({t:{}})
                    total_type = set_fre_trn_fleet_by[t]
                    set_fre_trn_fleet_by_sh.update({t:{}})
                    for f in fuels:
                        dict_resi_cap_trn[t].update({f:[]})
                        mask_tf = (df_trn_data['Type'] == t) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Residual fleet') & \
                                (df_trn_data['Fuel'] == f)
                        try:
                            fby_tf = \
                                df_trn_data.loc[mask_tf][per_first_yr].iloc[0]
                            fuels_fre.append(f)
                        except Exception:
                            fby_tf = 0
    
                        if total_type > 0:
                            set_fre_trn_fleet_by_sh[t].update({f:100*fby_tf/total_type})
                        else:
                            set_fre_trn_fleet_by_sh[t].update({f:0})
    
                        for y in range(len(time_vector)):
                            a_fleet = \
                                df_trn_data.loc[mask_tf][time_vector[y]].iloc[0]
                            dict_resi_cap_trn[t][f].append(a_fleet)
    
                """
                Paso 2: Proyectar la participación de cada fuel en cada carrocería usando el parámetro "Electrification"
                """
                dict_fuel_economy = {}
                dict_shares_fleet = {}
                types_all = types_pass + types_fre
    
                for t in types_all:
                    # ...calculating non-electric fuel distribution
                    if t in types_pass:
                        set_trn_fleet_by_sh = set_pass_trn_fleet_by_sh
                    else:
                        set_trn_fleet_by_sh = set_fre_trn_fleet_by_sh
                    sh_non_electric = 0
                    for fne in fuels_nonelectric:
                        sh_non_electric += \
                            set_trn_fleet_by_sh[t][fne]
    
                    sh_non_electric_k = {}
                    for fne in fuels_nonelectric:
                        this_sh_k_f = set_trn_fleet_by_sh[t][fne]
                        if sh_non_electric > 0:
                            this_sh_non_ele_k = \
                                100*this_sh_k_f/sh_non_electric
                        else:
                            this_sh_non_ele_k = 0
                        sh_non_electric_k.update({fne:this_sh_non_ele_k})
    
                    # ...opening electrification percentages:
                    list_electrification = []
                    list_hydrogen = []
                    list_non_electric = []
                    mask_ele = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                            (df_trn_data['Parameter'] == 'Electrification') & \
                            (df_trn_data['Type'] == t)
                    mask_h2 = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                            (df_trn_data['Parameter'] == 'Hydrogen Penetration') & \
                            (df_trn_data['Type'] == t)
    
                    # ...opening fuel economies
                    list_fe_k = {}
                    list_fe_k_masks = {}
                    list_nonele_fleet_k = {}  # open non-electrical fleet
                    for af in fuels:
                        list_fe_k.update({af: []})
                        if af in fuels_nonelectric:
                            list_nonele_fleet_k.update({af: []})
                        mask_fe_af = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                    (df_trn_data['Parameter'] == 'Fuel economy') & \
                                    (df_trn_data['Type'] == t) & \
                                    (df_trn_data['Fuel'] == af)
                        list_fe_k_masks.update({af: deepcopy(mask_fe_af)})
                
                    # ...iterating across years:
                    for y in range(len(time_vector)):
                        this_yea = time_vector[y]
    
                        this_ele = df_trn_data.loc[mask_ele][this_yea].iloc[0]
                        if math.isnan(this_ele):
                            this_ele = 0
                        list_electrification.append(this_ele)
    
                        this_h2 = df_trn_data.loc[mask_h2][this_yea].iloc[0]
                        if math.isnan(this_h2):
                            this_h2 = 0
                        list_hydrogen.append(this_h2)
                        list_non_electric.append(100-this_ele-this_h2)
    
                        for af in fuels_nonelectric:
                            this_sh_ne_k = sh_non_electric_k[af]
                            this_fleet_ne_k = \
                                this_sh_ne_k*list_non_electric[-1]/100
                            if math.isnan(this_fleet_ne_k):
                                this_fleet_ne_k = 0
                            list_nonele_fleet_k[af].append(this_fleet_ne_k)
    
                        for af in fuels:
                            this_mask_fe = list_fe_k_masks[af]
                            this_fe_k = \
                                df_trn_data.loc[this_mask_fe
                                                    ][this_yea].iloc[0]
                            if math.isnan(this_fe_k):
                                this_fe_k = 0
                            list_fe_k[af].append(this_fe_k)
    
                    '''
                    # Disturb 10: make the list electrification and hydrogen different
                    list_electrification_raw = deepcopy(list_electrification)
                    list_electrification = \
                        interpolation_non_linear_final(
                            time_vector, list_electrification_raw, 0.5, 2025)
                    print('review')
                    sys.exit()
                    '''
    
                    # ...store the data for this "type":
                    dict_fuel_economy.update({t:{}})
                    dict_shares_fleet.update({t:{}})
                    for af in fuels:
                        if af in fuels_nonelectric:
                            this_fleet_sh_k = \
                                deepcopy(list_nonele_fleet_k[af])
                        elif af == 'ELECTRICIDAD':
                            this_fleet_sh_k = \
                                deepcopy(list_electrification)
                        elif af == 'HIDROGENO':
                            this_fleet_sh_k = \
                                deepcopy(list_hydrogen)
                        else:
                            print('Undefined set (1). Please check.')
                            sys.exit()
    
                        dict_shares_fleet[t].update({af:this_fleet_sh_k})
                        add_fe_k = deepcopy(list_fe_k[af])
                        dict_fuel_economy[t].update({af:add_fe_k})
    
                """
                Paso 3: calcular la energía requerida para el sector transporte
                """
                dict_trn_pj = {}
                for af in fuels:
                    dict_trn_pj.update({af: []})
    
                dict_gpkm_gtkm = {}
                for t in types_all:
                    if t in list(gpkm_pri_k.keys()):
                        this_gpkm_add = gpkm_pri_k[t]
                    if t in list(gpkm_pub_k.keys()):
                        this_gpkm_add = gpkm_pub_k[t]
                    if t in list(gtkm_freight_k.keys()):
                        this_gpkm_add = gtkm_freight_k[t]
                    dict_gpkm_gtkm.update({t:this_gpkm_add})
    
                dict_trn_pj_2 = {}
                for af in fuels:
                    dict_trn_pj_2.update({af: {}})
                    for t in types_all:
                        dict_trn_pj_2[af].update({t: [0]*len(time_vector)})
    
                # To calculate the fleet vectors, we need the gpkm per
                # vehicle type, which can be obtained in the loop below.
                dict_gpkm_gtkm_k, dict_fleet_k = {}, {}
                dict_fuel_con = {}
                zero_list = [0 for y in range(len(time_vector))]
                for t in types_all:
                    dict_gpkm_gtkm_k.update({t:{}})
                    dict_fleet_k.update({t:{}})
                    dict_fuel_con.update({t:{}})
                    for f in fuels:
                        dict_gpkm_gtkm_k[t].update({f:deepcopy(zero_list)})
                        dict_fleet_k[t].update({f:deepcopy(zero_list)})
                        dict_fuel_con[t].update({f:deepcopy(zero_list)})
    
                # For all fuels and techs, find if there is a valid projected fleet
                # to overwrite the demand modeling.
                dict_proj_fleet = {}
                for this_f in fuels:
                    dict_proj_fleet.update({this_f:{}})
                    for t in types_all:
                        dict_proj_fleet[this_f].update({t:{}})
                        mask_fy_tf = \
                            (df_trn_data['Scenario'] == this_scen) & \
                            (df_trn_data['Fuel'] == this_f) & \
                            (df_trn_data['Type'] == t) & \
                            (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                            (df_trn_data['Parameter'] == 'Projected fleet')
                        fy_tf_list = []
                    
                        for y in range(len(time_vector)):
                            fy_tf = \
                                df_trn_data.loc[mask_fy_tf][time_vector[y]].iloc[0]
                            fy_tf_list.append(fy_tf)
    
                        # Decide what to do with the inputted data:
                        fy_tf_proj = \
                            df_trn_data.loc[mask_fy_tf]['projection'].iloc[0]
                        if fy_tf_proj == 'ignore':
                            dict_proj_fleet[this_f][t].update({'indicate':'ignore'})
                        elif fy_tf_proj == 'user_defined':
                            dict_proj_fleet[this_f][t].update({'indicate':'apply'})
                            dict_proj_fleet[this_f][t].update({'vals':fy_tf_list})
                        else:
                            print('There is an undefined projection type specified for the Projected fleet (12_trans_data).')
                            print("The code will stop at line 1417 for review.")
                            sys.exit()
    
                # For all fuels, find the energy consumption:
                    
                dict_diffs_f_rf = {}
                
                # emissions_demand = {}  # crucial output
                
                for this_f in fuels:
                    
                    dict_diffs_f_rf.update({this_f:{}})
                    
                    this_list = []
                    
                    emis_transport_dict = {}
                    
                    for y in range(len(time_vector)):
                        this_fuel_con = 0
                        for t in types_all:
                            if y == 0:
                                emis_transport_dict.update({t:[]})
                                if t not in list(emissions_demand.keys()):
                                    emissions_demand.update({t:{}})
    
                                    # print('happens?')
                                    # sys.exit()
    
                            this_gpkm_gtkm = dict_gpkm_gtkm[t][y]
                            this_sh_fl = dict_shares_fleet[t][this_f][y]/100
                            this_fe = dict_fuel_economy[t][this_f][y]
    
                            # Extract the fuel consumption:
                            add_fuel_con = \
                                this_gpkm_gtkm*this_sh_fl*this_fe/dict_lf[t]
    
                            dict_fuel_con[t][this_f][y] = deepcopy(add_fuel_con)
                            '''
                            Units analysis: Gpkm*MJ/km = PJ
                            '''
                            this_fuel_con += deepcopy(add_fuel_con)
                            
                            dict_trn_pj_2[this_f][t][y] = deepcopy(add_fuel_con)
                            
                            # Calculate the distribution of gpkm or gtkm:
                            this_gpkm_gtkm_k = this_gpkm_gtkm*this_sh_fl
                            dict_gpkm_gtkm_k[t][this_f][y] = \
                                this_gpkm_gtkm_k
                            # Calculate the fleet:
                            this_fleet_k = 1e9*\
                                (this_gpkm_gtkm_k/dict_lf[t])/dict_km[t]
                            dict_fleet_k[t][this_f][y] = this_fleet_k
    
                            if dict_km[t] == 0:
                                print('review division by zero')
                                sys.exit()
    
                            resi_fleet = dict_resi_cap_trn[t][this_f][0]
                            if y == 0 and resi_fleet != 0:
                                dict_diffs_f_rf[this_f].update({t:this_fleet_k/resi_fleet})
    
                            # This code is added to overwrite the fleet projections:
                            if dict_proj_fleet[this_f][t]['indicate'] == 'apply':
                                proj_fleet_list = dict_proj_fleet[this_f][t]['vals']
                                proj_fleet_y = proj_fleet_list[y]
    
                                # Overwriting fleet:
                                dict_fleet_k[t][this_f][y] = deepcopy(proj_fleet_y)
    
                                # Overwriting gpkm_gtkm (gotta sum):
                                delta_km = (proj_fleet_y-this_fleet_k)*dict_km[t]
                                delta_gpkm_gtkm = delta_km*dict_lf[t]/1e9
                                dict_gpkm_gtkm[t][y] += deepcopy(delta_gpkm_gtkm)
    
                                # Overwriting fuel (gotta sum):
                                delta_fuel_con = delta_km*this_fe/1e9  # PJ
                                dict_fuel_con[t][this_f][y] += delta_fuel_con
                                this_fuel_con += delta_fuel_con
    
                                dict_trn_pj_2[this_f][t][y] += deepcopy(delta_fuel_con)
    
                            # Estimate emissions:
                            fuel_energy_model = dict_eq_transport_fuels[this_f]
                            if fuel_energy_model in list(emissions_fuels_dict.keys()):
                                emis_fact = emissions_fuels_dict[fuel_energy_model][y]
                            else:
                                emis_fact = 0
                            emis_transport = dict_fuel_con[t][this_f][y]*emis_fact
    
                            emis_transport_dict[t].append(emis_transport)
    
                        this_list.append(this_fuel_con)
    
                    #
                    for t in types_all:
                        emissions_demand[t].update({
                            this_f:deepcopy(emis_transport_dict[t])})
    
                    dict_trn_pj[this_f] = deepcopy(this_list)
    
                # print('Check emissions for transport - 0')
                # sys.exit()
    
                # if this_scen != 'BAU':
                #    print('review transport demand projections up until here')
                #    sys.exit()
    
                # *********************************************************
                # We can calculate the required new fleets to satisfy the demand:
                dict_new_fleet_k, dict_accum_new_fleet_k = {}, {}
    
                # We will take advantage to estimate the costs related to
                # fleet and energy; we can check the cost and tax params:
                cost_params = list(dict.fromkeys(d5_tpt['Parameter'].tolist()))
                # ['CapitalCost', 'FixedCost', 'VariableCost', 'OpLife']
                cost_units = list(dict.fromkeys(d5_tpt['Unit'].tolist()))
                # 
                tax_params = list(dict.fromkeys(d5_tax['Parameter'].tolist()))
                # ['Imports', 'IMESI_Venta', 'IVA_Venta', 'Patente',
                # 'IMESI_Combust', 'IVA_Gasoil', 'IVA_Elec', 'Impuesto_Carbono',
                # 'Otros_Gasoil']
    
                # Define the cost outputs:
                dict_capex_out = {}
                dict_fopex_out = {}
                dict_vopex_out = {}
    
                # Define the tax outputs:
                dict_tax_out = {}
                for atax in tax_params:
                    dict_tax_out.update({atax:{}})
                    for t in types_all:
                        dict_tax_out[atax].update({t:{}})
                        for f in fuels:
                            dict_tax_out[atax][t].update({f:{}})
    
                # Let's now start the loop:
                times_neg_new_fleet, times_neg_new_fleet_sto = 0, []
                for t in types_all:
                    dict_new_fleet_k.update({t:{}})
                    dict_capex_out.update({t:{}})
                    dict_fopex_out.update({t:{}})
                    dict_vopex_out.update({t:{}})
                    for f in fuels:
                        # Unpack the costs:
                        list_cap_cost, unit_cap_cost = \
                            fun_unpack_costs('CapitalCost', t, f,
                                            d5_tpt,
                                            time_vector)
                        list_fix_cost, unit_fix_cost = \
                            fun_unpack_costs('FixedCost', t, f,
                                            d5_tpt,
                                            time_vector)
                        list_var_cost, unit_var_cost = \
                            fun_unpack_costs('VariableCost', t, f,
                                            d5_tpt,
                                            time_vector)
                        list_op_life, unit_op_life = \
                            fun_unpack_costs('OpLife', t, f,
                                            d5_tpt,
                                            time_vector)
                        apply_costs = \
                            {'CapitalCost':deepcopy(list_cap_cost),
                            'VariableCost':deepcopy(list_var_cost)}
    
                        # Now we are ready to estimate the "new fleet":
                        tot_fleet_lst = dict_fleet_k[t][f]
                        res_fleet_lst = dict_resi_cap_trn[t][f]
                        fuel_con_lst = dict_fuel_con[t][f]
    
                        # We need to calculate new_fleet and accum_fleet:
                        accum_fleet_lst = [0 for y in range(len(time_vector))]
                        new_fleet_lst = [0 for y in range(len(time_vector))]
                        for y in range(len(time_vector)):
                            if y == 0:
                                this_new_fleet = tot_fleet_lst[y] - \
                                    res_fleet_lst[y]
                            else:
                                this_new_fleet = tot_fleet_lst[y] - \
                                    res_fleet_lst[y] - accum_fleet_lst[y]
                            # We can store the fleet below:
                            if this_new_fleet >= 0:
                                new_fleet_lst[y] = this_new_fleet
                                # The "this_new_fleet" existis during the vehicles lifetime
                                for y2 in range(y, y+int(list_op_life[y])):
                                    if y2 < len(time_vector):
                                        accum_fleet_lst[y2] += this_new_fleet
                            else:
                                times_neg_new_fleet += 1
                                times_neg_new_fleet_sto.append(this_new_fleet)
                                # if this_new_fleet < -10000:
                                #    print('This is surprising')
                                #    sys.exit()
    
                        '''
                        Remember to apply conversions depending on unit
                        USD/veh
                        USD/pkm
                        USD/1000 km
                        USD/liter
                        USD/kWh
                        USD/kg
                        Ref: # http://w.astro.berkeley.edu/~wright/fuel_energy.html
                        '''
                        if unit_var_cost == 'USD/liter' and 'DIESEL' in f:
                            conv_cons = 38.6*(1e-9)  # from liter to PJ
                        if unit_var_cost == 'USD/liter' and 'GASOLINA' in f:
                            conv_cons = 34.2*(1e-9)  # from liter to PJ
                        if unit_var_cost == 'USD/kWh':
                            conv_cons = 3.6e-9  # from kWh to PJ
                        if unit_var_cost == 'USD/kg':
                            conv_cons = (3.6*33.33)*1e-9  # from kg to PJ
    
                        # Proceed with cost calculations:
                        usd_capex_lst = []
                        usd_fopex_lst = []
                        usd_vopex_lst = []
                        for y in range(len(time_vector)):
                            add_usd_capex = \
                                new_fleet_lst[y]*list_cap_cost[y]
                            add_usd_fopex = \
                                tot_fleet_lst[y]*list_fix_cost[y]*dict_km[t]/1000
                            add_usd_vopex = \
                                fuel_con_lst[y]*list_var_cost[y]/conv_cons
                            usd_capex_lst.append(add_usd_capex)
                            usd_fopex_lst.append(add_usd_fopex)
                            usd_vopex_lst.append(add_usd_vopex)
    
                        # print('review this')
                        # sys.exit()
    
                        # Bring cost store variable:
                        dict_new_fleet_k[t].update({f:new_fleet_lst})
                        dict_capex_out[t].update({f:usd_capex_lst})
                        dict_fopex_out[t].update({f:usd_fopex_lst})
                        dict_vopex_out[t].update({f:usd_vopex_lst})
    
                        # Unpack taxes:
                        apply_taxes = {'list_rate':{},
                                    'list_unit':{}, 'ref_param':{}}
                        for atax in tax_params:
                            list_atax, ref_atax, mult_depr = \
                                fun_unpack_taxes(atax, t, f,
                                                d5_tax,
                                                time_vector)
                            list_atax_unit = []  # tax ready for activity data
    
                            '''
                            "mult_depr" is a constant to multiply the depreciation
                            factor; if it varies across years, the implementation
                            must change.
                            '''
    
                            for y in range(len(time_vector)):
                                try:
                                    if ref_atax == 'CapitalCost*':
                                        ref_atax_call = 'CapitalCost'
                                    else:
                                        ref_atax_call = ref_atax
                                    apply_costs_atax = apply_costs[ref_atax_call][y]
                                except Exception:
                                    apply_costs_atax = 0
                                add_atax_unit = \
                                    apply_costs_atax*list_atax[y]/100
                                list_atax_unit.append(add_atax_unit*mult_depr)
    
                            apply_taxes['list_rate'].update({atax:deepcopy(list_atax)})
                            apply_taxes['list_unit'].update({atax:deepcopy(list_atax_unit)})
                            apply_taxes['ref_param'].update({atax:deepcopy(ref_atax)})
    
                            add_atax_val_lst = []
                            for y in range(len(time_vector)):
                                if ref_atax == 'CapitalCost':
                                    add_atax_val = list_atax_unit[y]*new_fleet_lst[y]
                                elif ref_atax == 'CapitalCost*':
                                    add_atax_val = list_atax_unit[y]*tot_fleet_lst[y]
                                else:  # variable cost
                                    add_atax_val = list_atax_unit[y]*fuel_con_lst[y]/conv_cons
                                add_atax_val_lst.append(add_atax_val)
    
                            dict_tax_out[atax][t][f] = \
                                deepcopy(add_atax_val_lst)
    
                # TM 3) Update the transport vector
                dict_eq_trn_fuels_rev = {}
                for this_f in fuels:
                    this_equivalence = dict_eq_transport_fuels[this_f]
                    if 'HIBRIDO' not in this_f:
                        dict_energy_demand_trn[this_equivalence] = \
                            deepcopy(dict_trn_pj[this_f])
                        dict_eq_trn_fuels_rev.update({this_equivalence:this_f})
                    else:
                        for y in range(len(time_vector)):
                            dict_energy_demand_trn[this_equivalence][y] += \
                                deepcopy(dict_trn_pj[this_f][y])
    
                dict_test_transport_model[this_scen][this_country].update({'original':deepcopy(dict_energy_demand),
                                                                        'adjusted':deepcopy(dict_energy_demand_trn)})
                # We must code a test that reviews the error of transport
                # and simple energy modeling:
                fuel_sets_all = list(dict_energy_demand['Transport'].keys())
                fuel_sets_trn = []
                diff_list_all = [0 for y in range(len(time_vector))]
                diff_dict_all = {}
                sum_list_all = [0 for y in range(len(time_vector))]
                error_list_all = [0 for y in range(len(time_vector))]
                for fs in fuel_sets_all:
                    diff_list_fs = []
                    for y in range(len(time_vector)):
                        this_orig_dem = \
                            dict_energy_demand['Transport'][fs][y]
                        this_orig_adj = \
                            dict_energy_demand_trn[fs][y]
                        this_diff = this_orig_dem-this_orig_adj
                        diff_list_fs.append(this_diff)
                        if round(this_diff, 2) != 0:
                            fuel_sets_trn.append(fs)
                            diff_list_all[y] += this_diff
                            sum_list_all[y] += this_orig_adj
                    diff_dict_all.update({fs: diff_list_fs})
    
                error_list_all = \
                    [100*diff_list_all[n]/v for n, v in enumerate(sum_list_all)]
    
                dict_ed_trn_ref = dict_energy_demand['Transport']
                dict_ed_trn_trn = dict_energy_demand_trn

                # Here we must adjust the dictionaries for compatibility:
                dict_energy_demand['Transport'] = \
                    deepcopy(dict_energy_demand_trn)

                # Add fuel consumption per transport tech
                for t in types_all:
                    dict_energy_demand.update({t:{}})
                    for af in fuels:
                        add_dict_trn_pj_2_list = \
                            dict_trn_pj_2[af][t]
                        dict_energy_demand[t].update({af:add_dict_trn_pj_2_list})

            # if this_scen != 'BAU':
            #    print(this_scen, 'compare the estimated energies')
            #    sys.exit()

            ###########################################################

            # ... here we already have a country's demands, now...
            # 3f) open the *externality* and *emission factors*
            # 'externality':
            mask_cost_ext = ((df5_ext['Country'] == this_country) &
                             (df5_ext['Use_row'] == 'Yes'))
            this_df_cost_externalities = df5_ext.loc[mask_cost_ext]
            this_df_cost_externalities.reset_index(drop=True, inplace=True)
            this_df_cost_externalities_fuels = \
                this_df_cost_externalities['Fuel'].tolist()

            externality_fuels_list = []
            externality_fuels_dict = {}

            for f in range(len(this_df_cost_externalities_fuels)):
                this_fuel = this_df_cost_externalities_fuels[f]
                factor_global_warming = this_df_cost_externalities.iloc[f]['Global warming']
                factor_local_pollution = this_df_cost_externalities.iloc[f]['Local pollution']
                unit_multiplier = this_df_cost_externalities.iloc[f]['Unit multiplier']

                externality_fuels_list.append(this_fuel)
                this_externality_dict = {'Global warming':factor_global_warming,
                                         'Local pollution':factor_local_pollution}
                externality_fuels_dict.update({this_fuel: this_externality_dict})

            # ...this is a good space to store externality and emission data of demand (by fuel) and fuels:
            externalities_globalwarming_demand = {}  # crucial output
            externalities_localpollution_demand = {}  # crucial output

            demand_tech_list = [
                i for i in list(dict_energy_demand.keys())
                if i not in types_all]
            for tech in demand_tech_list:
                demand_fuel_list = list(dict_energy_demand[tech].keys())
                emissions_demand.update({tech:{}})
                externalities_globalwarming_demand.update({tech:{}})
                externalities_localpollution_demand.update({tech:{}})
                for fuel in demand_fuel_list:
                    if fuel in emissions_fuels_list:  # store emissions
                        list_emissions_demand = []
                        for y in range(len(time_vector)):
                            add_value = \
                                dict_energy_demand[tech][fuel][y]*emissions_fuels_dict[fuel][y]
                            list_emissions_demand.append(add_value)
                        emissions_demand[tech].update({fuel:list_emissions_demand})

                    if fuel in externality_fuels_list:  # store externalities
                        list_globalwarming_demand = []
                        list_localpollution_demand = []
                        for y in range(len(time_vector)):
                            try:
                                add_value_globalwarming = \
                                    dict_energy_demand[tech][fuel][y]*externality_fuels_dict[fuel]['Global warming']
                            except Exception:
                                add_value_globalwarming = 0
                            list_globalwarming_demand.append(add_value_globalwarming)

                            try:
                                add_value_localpollution = \
                                    dict_energy_demand[tech][fuel][y]*externality_fuels_dict[fuel]['Local pollution']
                            except Exception:
                                add_value_localpollution = 0
                            list_localpollution_demand.append(add_value_localpollution)

                        externalities_globalwarming_demand[tech].update({fuel:list_globalwarming_demand})
                        externalities_localpollution_demand[tech].update({fuel:list_localpollution_demand})

            ext_by_country.update({this_country:deepcopy(externality_fuels_dict)})

            dict_local_country[this_country].update({'Global warming externalities by demand': externalities_globalwarming_demand})
            dict_local_country[this_country].update({'Local pollution externalities by demand': externalities_localpollution_demand})
            dict_local_country[this_country].update({'Emissions by demand': emissions_demand})

            # print('Review emissions calculation')
            # sys.exit()

            # Select the existing capacity from the base cap dictionary:
            """
            if this_reg == '2_Central America':
                this_reg_alt = '2_CA'
            else:
                this_reg_alt = this_reg
            """
            dict_base_caps = \
                dict_database['Cap'][this_reg][this_country]  # by pp type

            # Select the "param_related_4"
            param_related_4 = 'Distribution of new electrical energy generation'
            mask_param_related_4 = (df_scen_rc['Parameter'] == param_related_4)
            df_param_related_4 = df_scen_rc.loc[mask_param_related_4]
            df_param_related_4.reset_index(drop=True, inplace=True)

            list_electric_sets = \
                df_param_related_4['Tech'].tolist()

            mask_filt_techs = \
                ((d5_power_techs['Projection'] != 'none') &
                 (d5_power_techs['Parameter'] == 'Net capacity factor'))
            list_electric_sets_2 = \
                d5_power_techs.loc[mask_filt_techs]['Tech'].tolist()

            list_electric_sets_3 = \
                list(set(list(dict_base_caps.keys())) &
                     set(list_electric_sets_2))

            # > Call other auxiliary variables:

            # Select the "param_related_5"
            param_related_5 = '% Imports for consumption'
            mask_param_related_5 = (df_scen_rc['Parameter'] == param_related_5)
            df_param_related_5 = df_scen_rc.loc[mask_param_related_5]
            df_param_related_5.reset_index(drop=True, inplace=True)

            # Select the "param_related_6"
            param_related_6 = '% Exports for production'
            mask_param_related_6 = (df_scen_rc['Parameter'] == param_related_6)
            df_param_related_6 = df_scen_rc.loc[mask_param_related_6]
            df_param_related_6.reset_index(drop=True, inplace=True)

            # Select the "param_related_7"
            param_related_7 = 'Fuel prices'
            mask_param_related_7 = (df_scen_rc['Parameter'] == param_related_7)
            df_param_related_7 = df_scen_rc.loc[mask_param_related_7]
            df_param_related_7.reset_index(drop=True, inplace=True)

            # ...proceed with the interpolation of fuel prices:
            fuel_list_local = df_param_related_7['Fuel'].tolist()

            for this_fuel in fuel_list_local:
                fuel_idx_7 = df_param_related_7['Fuel'].tolist().index(this_fuel)
                this_fuel_price_projection = df_param_related_7.loc[fuel_idx_7, 'projection']
                this_fuel_price_value_type = df_param_related_7.loc[fuel_idx_7, 'value']

                if (this_fuel_price_projection == 'flat' and this_fuel_price_value_type == 'constant'):
                    for y in range(len(time_vector)):
                        df_param_related_7.loc[fuel_idx_7, time_vector[y]] = \
                            float(df_param_related_7.loc[fuel_idx_7, time_vector[0]])
                        

                if (this_fuel_price_projection == 'Interpolate'):
                    for y in range(len(time_vector)):
                        df_param_related_7[time_vector[y]] = pd.to_numeric(df_param_related_7[time_vector[y]], errors='coerce')

                    # Convert to numeric and interpolate
                    numeric_data = pd.to_numeric(df_param_related_7.loc[fuel_idx_7, time_vector], errors='coerce')
                    interpolated_data = numeric_data.interpolate(limit_direction='both')

                    # Assign interpolated values back to DataFrame
                    with warnings.catch_warnings():
                        df_param_related_7.loc[fuel_idx_7, time_vector] = deepcopy(interpolated_data)

                elif this_fuel_price_projection == 'Percent growth of incomplete years':
                    growth_param = df_param_related_7.loc[ fuel_idx_7, 'value' ]
                    for y in range(len(time_vector)):
                        value_field = df_param_related_7.loc[ fuel_idx_7, time_vector[y] ]
                        if math.isnan(value_field) == True:
                            df_param_related_7.loc[ fuel_idx_7, time_vector[y] ] = \
                                round(df_param_related_7.loc[ fuel_idx_7, time_vector[y-1] ]*(1 + growth_param/100), 4)

            # ('check interpolations end')
            # sys.exit()

            # Select the "param_related_8"
            param_related_8 = 'Planned new capacity'
            mask_param_related_8 = (df_scen_rc['Parameter'] == param_related_8)
            df_param_related_8 = df_scen_rc.loc[mask_param_related_8]
            df_param_related_8.reset_index(drop=True, inplace=True)

            # Select the "param_related_9"
            param_related_9 = 'Phase-out capacity'
            mask_param_related_9 = (df_scen_rc['Parameter'] == param_related_9)
            df_param_related_9 = df_scen_rc.loc[mask_param_related_9]
            df_param_related_9.reset_index(drop=True, inplace=True)

            # Select the "param_related_10"
            param_related_10 = 'Capacity factor change'
            mask_param_related_10 = (df_scen_rc['Parameter'] == param_related_10)
            df_param_related_10 = df_scen_rc.loc[mask_param_related_10]
            df_param_related_10.reset_index(drop=True, inplace=True)

            # Select the existing transformation inputs information (receive negative values only):
            dict_base_transformation = \
                dict_database['EB'][this_reg][this_country_2]['Total transformation']['Power plants']  # by fuel (input)
            dict_base_transformation_2 = \
                dict_database['EB'][this_reg][this_country_2]['Total transformation']['Self-producers']  # by fuel (input)

            base_transformation_fuels = \
                list(dict_base_transformation.keys()) + \
                list(dict_base_transformation_2.keys())
            base_transformation_fuels = list(set(base_transformation_fuels))
            base_transformation_fuels.sort()
            base_electric_fuels_use = []
            base_electric_fuel_use = {}
            base_electric_production = {}
            base_electric_production_pps = {}  # power plants
            base_electric_production_sps = {}  # self-producers

            # ...search here if the fuels have negative values, which indicates whether we have a reasonable match
            for btf in base_transformation_fuels:
                try:
                    btf_value_1 = dict_base_transformation[btf][base_year]
                except Exception:
                    btf_value_1 = 0
                try:
                    btf_value_2 = dict_base_transformation_2[btf][base_year]
                except Exception:
                    btf_value_2 = 0

                btf_value = btf_value_1 + btf_value_2  # // ignore self-producers
                if btf_value < 0:
                    base_electric_fuels_use.append(btf)
                    base_electric_fuel_use.update({btf:-1*btf_value})
                if btf_value > 0:
                    base_electric_production.update({btf:btf_value})
                    base_electric_production_pps.update({btf:btf_value_1/btf_value})
                    base_electric_production_sps.update({btf:btf_value_2/btf_value})

            # ...extract losses and self-consumption
            electricity_losses = \
                dict_database['EB'][this_reg][this_country_2]['Losses']['none']['Electricity'][base_year]
            electricity_self_consumption = \
                dict_database['EB'][this_reg][this_country_2]['Self-consumption']['none']['Electricity'][base_year]
            electricity_imports = \
                dict_database['EB'][this_reg][this_country_2]['Total supply']['Imports']['Electricity'][base_year]
            electricity_exports = \
                dict_database['EB'][this_reg][this_country_2]['Total supply']['Exports']['Electricity'][base_year]
            electricity_adjust = \
                dict_database['EB'][this_reg][this_country_2]['Adjustment']['none']['Electricity'][base_year]

            # ...create imports and exports list:
            electricity_losses_list = []
            electricity_self_consumption_list = []
            electricity_imports_list = []
            electricity_exports_list = []

            losses_share = \
                electricity_losses/dict_energy_demand_by_fuel['Electricity'][0]
            self_consumption_share = \
                electricity_self_consumption/dict_energy_demand_by_fuel['Electricity'][0]
            imports_share = \
                electricity_imports/dict_energy_demand_by_fuel['Electricity'][0]
            exports_share = \
                electricity_exports/dict_energy_demand_by_fuel['Electricity'][0]  # this will be negative

            # ...here we must manipulate the limit to the losses!
            # Select the "param_related_11"
            param_related_11 = 'Max losses'
            mask_param_related_11 = (df_scen_rc['Parameter'] == param_related_11)
            df_param_related_11 = df_scen_rc.loc[mask_param_related_11]
            df_param_related_11.reset_index(drop=True, inplace=True)

            maxloss_projection = df_param_related_11.iloc[0]['projection']
            maxloss_baseyear_str = df_param_related_11.loc[0, time_vector[0]]
            loss_vector = []

            if maxloss_projection == 'flat' and maxloss_baseyear_str == 'endogenous':
                for y in range(len(time_vector)):
                    loss_vector.append(losses_share)

            if maxloss_projection == 'interpolate' and maxloss_baseyear_str == 'endogenous':
                this_known_loss_vals = []
                for y in range(len(time_vector)):
                    if y == 0:
                        this_known_loss_vals.append(losses_share)
                    elif type(df_param_related_11.loc[0, time_vector[y]]) is int:
                        suggested_maxloss = df_param_related_11.loc[0, time_vector[y]]/100
                        if suggested_maxloss < losses_share:
                            this_known_loss_vals.append(suggested_maxloss)
                        else:
                            this_known_loss_vals.append(losses_share)
                    else:
                        this_known_loss_vals.append('')

                loss_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr,
                                         this_known_loss_vals, 'ini',
                                         this_scen, '')

            # ... now apply the appropriate loss vector:
            for y in range(len(time_vector)):
                electricity_losses_list.append(loss_vector[y]*dict_energy_demand_by_fuel['Electricity'][y])
                electricity_self_consumption_list.append(self_consumption_share*dict_energy_demand_by_fuel['Electricity'][y])
                electricity_imports_list.append(imports_share*dict_energy_demand_by_fuel['Electricity'][y])
                electricity_exports_list.append(exports_share*dict_energy_demand_by_fuel['Electricity'][y])

            # 3g) Here we must call some inputs to make the model adjust to the desired electrical demands
            param_related_12 = 'RE TAG'
            mask_12 = (df_scen_rc['Parameter'] == param_related_12)
            df_param_related_12 = df_scen_rc.loc[mask_12]
            df_param_related_12.reset_index(drop=True, inplace=True)

            reno_targets_exist = False
            if len(df_param_related_12.index.tolist()) > 0:
                reno_targets_exist = True
                reno_target_list = df_param_related_12[time_vector].iloc[0].tolist()
            
            # print('did i get here')
            # sys.exit()

            param_related_14 = 'Base year electricity production'
            mask_14 = (df_scen_rc['Parameter'] == param_related_14)
            df_param_related_14 = df_scen_rc.loc[mask_14]
            df_param_related_14.reset_index(drop=True, inplace=True)

            # print('review if this finished - first part')
            # sys.exit()

            # 3h) obtain the required *new electrical capacity by pp*, *electricity production by pp*, *fuel consumption by pp*
            #     to supply the electricity demand:
            # NOTE: we have to subtract the elements
            special_list_1 = ['Chile']
            special_list_2= ['Colombia']
            if this_country in special_list_1:
                demand_method = "Just power plants"
            elif this_country in special_list_2:
                demand_method = "With self-consumption"
            else:
                demand_method = "Any"
            electrical_demand_to_supply = [0 for y in range(len(time_vector))]
            for y in range(len(time_vector)):
                if demand_method == "With self-consumption":
                    electrical_demand_to_supply[y] = \
                        dict_energy_demand_by_fuel['Electricity'][y] + \
                        electricity_losses_list[y] + \
                        electricity_self_consumption_list[y] - \
                        electricity_imports_list[y] - \
                        electricity_exports_list[y]
                if demand_method == "Just power plants" or demand_method == 'Any':
                    electrical_demand_to_supply[y] = \
                        dict_energy_demand_by_fuel['Electricity'][y] + \
                        electricity_losses_list[y] + \
                        0 - \
                        electricity_imports_list[y] - \
                        electricity_exports_list[y]

            # ...here, 'Total' is the net energy loss in transformation
            base_electric_prod = base_electric_production['Electricity']
            if demand_method == "With self-consumption":
                pass
            if demand_method == "Just power plants":
                base_electric_prod *= base_electric_production_pps['Electricity']
                base_electric_prod += electricity_self_consumption_list[0]
            if demand_method == "Just power plants without self-consumption":
                base_electric_prod *= base_electric_production_pps['Electricity']            
            base_electric_use_fuels = \
                deepcopy(base_electric_fuels_use)
            #base_electric_use_fuels.remove('Total')  # not needed anymore
            #base_electric_use_fuels.remove('Total primary sources')

            # ...we can extract the fuels we use in our technological options:
            used_fuel_list = []
            for tech in list_electric_sets:
                used_fuel = tech.split('_')[-1]
                if used_fuel not in used_fuel_list:
                    used_fuel_list.append(used_fuel)

            # ...here we need to iterate across the list of technologies and find the base distribution of electricity production:
            res_energy_shares = {}
            res_energy_sum_1 = 0
            res_energy_sum_2 = 0
            res_energy_sum_3 = 0
            store_percent = {}
            store_percent_rem = {}
            store_use_cap = {}
            store_res_energy = {}
            store_res_energy_all = [0 for y in range(len(time_vector))]
            
            # Add a list to change the unplanned new capaity assigned
            store_unplanned_energy_all = [0 for y in range(len(time_vector))]
            
            # Blank capacity factors:
            cf_by_tech = {}
            forced_newcap_energy_by_tech = {}
            forced_newcap_by_tech = {}
            forced_newcap_energy_all = [0 for y in range(len(time_vector))]

            accum_forced_newcap_by_tech = {}

            # ...this is the first previous loop:
            # 1st power sector loop
            for tech in list_electric_sets_3:

                store_use_cap.update({tech:[0 for y in range(len(time_vector))]})
                store_res_energy.update({tech:[0 for y in range(len(time_vector))]})
                forced_newcap_energy_by_tech.update({tech:[0 for y in range(len(time_vector))]})
                forced_newcap_by_tech.update({tech:[0 for y in range(len(time_vector))]})

                accum_forced_newcap_by_tech.update({tech:[0 for y in range(len(time_vector))]})

                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                # Extract planned and phase out capacities if they exist:
                try:
                    tech_idx_8 = df_param_related_8['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_8 = ''
                try:
                    tech_idx_9 = df_param_related_9['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_9 = ''
                tech_counter += 1

                # ...here extract the technical characteristics of the power techs
                mask_this_tech = \
                    (d5_power_techs['Tech'] == tech)
                this_tech_df_cost_power_techs = \
                    deepcopy(d5_power_techs.loc[mask_this_tech])

                # CAU
                mask_cau = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAU')
                df_mask_cau = \
                    deepcopy(this_tech_df_cost_power_techs.loc[mask_cau])
                df_mask_cau_unitmult = df_mask_cau.iloc[0]['Unit multiplier']
                df_mask_cau_proj = df_mask_cau.iloc[0]['Projection']
                list_tech_cau = []
                if df_mask_cau_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cau.iloc[0][y]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)
                if df_mask_cau_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cau.iloc[0][time_vector[0]]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)

                # Net capacity factor
                mask_cf = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Net capacity factor')
                df_mask_cf = \
                    deepcopy(this_tech_df_cost_power_techs.loc[mask_cf])
                df_mask_cf_unitmult = df_mask_cf.iloc[0]['Unit multiplier']
                df_mask_cf_proj = df_mask_cf.iloc[0]['Projection']
                list_tech_cf = []
                if df_mask_cf_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cf.iloc[0][y]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)
                if df_mask_cf_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cf.iloc[0][time_vector[0]]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)

                # ...here calculate the capacity factor if existent:
                this_tech_base_cap = dict_base_caps[tech][base_year]

                # Here we call the capacity factors from an external analysis:
                if df_mask_cf_proj in ['normalize_by_endogenous', 'flat_endogenous']:
                    ###########################################################
                    #2021
                    mask_cf_tech_2021 = \
                        ((df4_cfs['Capacity Set'] == tech) & \
                         (df4_cfs['Year'] == 2021) & \
                         (df4_cfs['Country'] == this_country))
                    this_df4_cfs_2021 = \
                        deepcopy(df4_cfs.loc[mask_cf_tech_2021])
                    this_df4_cfs_2021.reset_index(inplace=True, drop=True)

                    ###########################################################
                    #2021 avg
                    if len(this_df4_cfs_2021.index.tolist()) != 0:
                        this_cf_yrl_2021 = this_df4_cfs_2021['Capacity factor'].iloc[0]
                        if this_cf_yrl_2021 < 1:
                            this_cf_yrl_2021_add = this_cf_yrl_2021
                            this_cf_cnt_2021 = 1
                        else:
                            this_cf_yrl_2021_add, this_cf_cnt_2021 = 0, 0
                    else:
                        this_cf_yrl_2021_add, this_cf_cnt_2021 = 0, 0

                    this_cf_list = []
                    for y in time_vector:
                        mask_cf_tech = \
                            ((df4_cfs['Capacity Set'] == tech) & \
                             (df4_cfs['Year'] == time_vector[0]) & \
                             (df4_cfs['Country'] == this_country)
                            )

                        this_df4_cfs = \
                            deepcopy(df4_cfs.loc[mask_cf_tech])
                        this_df4_cfs.reset_index(inplace=True, drop=True)

                        if len(this_df4_cfs.index.tolist()) != 0 or len(this_df4_cfs_2021.index.tolist()) != 0:  # proceed

                            if len(this_df4_cfs.index.tolist()) != 0:
                                this_cf_yrl = this_df4_cfs['Capacity factor'].iloc[0]
                            else:
                                this_cf_yrl = 99
                                print('This capacity factor condition is not possible for this version. Review inputs and structure.')
                                sys.exit()

                            # Select the appropiate historic capacity factors:
                            this_cf = this_cf_yrl

                        else:
                            this_cf = 0  # this means we must use a default CF // to be selected

                        this_cf_list.append(this_cf)
                        #
                    #
                #
                # Here we define the capacity factors endogenously:
                if df_mask_cf_proj == 'normalize_by_endogenous':
                    y_counter_i = 0
                    for y in time_vector:
                        this_cf = this_cf_list[y_counter_i]
                        if this_cf != 0:  # this applies only if energy production is existent
                            mult_value = \
                                df_mask_cf.iloc[0][y]/df_mask_cf.iloc[0][time_vector[0]]*df_mask_cf_unitmult
                            add_value = mult_value*this_cf
                        else:  # this applies when the technology is non-existent
                            add_value = \
                                df_mask_cf.iloc[0][y]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)
                        y_counter_i += 1

                if df_mask_cf_proj == 'flat_endogenous':
                    y_counter_i = 0
                    for y in time_vector:
                        this_cf = this_cf_list[y_counter_i]
                        if this_cf != 0:  # this applies only if energy production is existent
                            add_value = \
                                this_cf*df_mask_cf_unitmult
                        else:
                            add_value = \
                                df_mask_cf.iloc[0][time_vector[0]]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)
                        y_counter_i += 1

                # ...calculate the base capacity/energy relationships for base year:
                # This is equivalent to parameter 9:
                this_tech_phase_out_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_9 != '':
                    for y in range(len(time_vector)):
                        this_tech_phase_out_cap[y] = \
                            df_param_related_9.loc[tech_idx_9, time_vector[y]]/1000  # in GW (MW to GW)

                res_energy_base_1 = this_tech_base_cap*8760*list_tech_cf[0]/1000  # MW to GW // CF
                res_energy_sum_1 += res_energy_base_1  # just for the base year

                cf_by_tech.update({tech:deepcopy(list_tech_cf)})

                # ...here store the potential "res_energy" and "use_cap":
                residual_cap_intermediate = \
                    [0 for y in range(len(time_vector))]

                accum_forced_cap_vector = [0 for y in range(len(time_vector))]
                forced_cap_vector = [0 for y in range(len(time_vector))]
                for y in range(len(time_vector)):

                    # ...here we need to store the energy production from forced new capacity
                    if tech_idx_8 != '':
                        forced_cap = \
                            df_param_related_8.loc[tech_idx_8, time_vector[y]]/1000  # in GW (MW to GW)
                    else:
                        forced_cap = 0

                    forced_cap_vector[y] += forced_cap

                    if y == 0:
                        residual_cap_intermediate[y] = this_tech_base_cap/1000
                    else:
                        residual_cap_intermediate[y] = \
                            residual_cap_intermediate[y-1] - this_tech_phase_out_cap[y]

                    # ...here we must add the accumulated planned new capacity:
                    for y_past in range(y+1):
                        accum_forced_cap_vector[y] += forced_cap_vector[y_past]
                    accum_forced_cap = accum_forced_cap_vector[y]

                    use_cap = residual_cap_intermediate[y]  # cap with calibrated values
                    res_energy = \
                        (use_cap)*list_tech_cau[y]*list_tech_cf[y]  # energy with calibrated values

                    store_use_cap[tech][y] += deepcopy(use_cap)
                    store_res_energy[tech][y] += deepcopy(res_energy)
                    store_res_energy_all[y] += deepcopy(res_energy)

                    forced_newcap_energy_by_tech[tech][y] += \
                        deepcopy(accum_forced_cap*list_tech_cau[y]*list_tech_cf[y])
                    forced_newcap_by_tech[tech][y] += \
                        deepcopy(forced_cap)
                    forced_newcap_energy_all[y] += \
                        deepcopy(accum_forced_cap*list_tech_cau[y]*list_tech_cf[y])

                    accum_forced_newcap_by_tech[tech][y] += \
                        deepcopy(accum_forced_cap)

            # Store the energy of the base year:
            store_res_energy_orig = deepcopy(store_res_energy)
            dict_store_res_energy_orig.update({this_country_2: deepcopy(store_res_energy_orig)})

            # ...this is the second previous loop:
            # 2nd power sector loop
            for tech in list_electric_sets_3:
                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                # ...let's bring to the front the base shares:
                this_tech_base_cap = dict_base_caps[tech][base_year]
                list_tech_cf_loc = cf_by_tech[tech]

                res_energy_base_1 = this_tech_base_cap*8760*list_tech_cf_loc[0]/1000  # MW to GW
                energy_dist_1 = res_energy_base_1/res_energy_sum_1

                # ...here we must take advantage of the loop to define the shares we will use:
                check_percent = False
                if list(set(df_param_related_4['value']))[0] == 'percent':
                    check_percent = True
                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_value_type = this_tech_dneeg_df_param_related['value']
                this_tech_dneeg_known_vals_raw = []
                this_tech_dneeg_known_vals = []
                this_tech_dneeg_known_vals_count = 0
                if check_percent is True:  # is compatible with "interpolate"
                    for y in time_vector:
                        add_value = \
                            this_tech_dneeg_df_param_related[y]

                        this_tech_dneeg_known_vals_raw.append(add_value)
                        # if str(y) == str(base_year):
                            # this_tech_dneeg_known_vals.append(energy_dist_1)  # this had been zero before
                        # elif type(add_value) is int or type(add_value) is float:
                        if type(add_value) is int or isinstance(add_value, (float, np.floating, int)):
                            if math.isnan(add_value) is False:
                                this_tech_dneeg_known_vals.append(add_value/100)
                                this_tech_dneeg_known_vals_count += 1
                            elif np.isnan(add_value):
                                this_tech_dneeg_known_vals.append('')
                            else:
                                pass
                        else:
                            this_tech_dneeg_known_vals.append('')

                    if add_value != 'rem':
                        this_tech_dneeg_vals = \
                            interpolation_to_end(time_vector, ini_simu_yr,
                                                 this_tech_dneeg_known_vals,
                                                 'last', this_scen, 'power')

                        if this_tech_dneeg_known_vals_count == len(this_tech_dneeg_vals):
                            this_tech_dneeg_vals = deepcopy(this_tech_dneeg_known_vals)

                    # if this_scen == 'NDCPLUS' and tech == 'PP_PV Utility_Solar':
                    #    print('review dneeg')
                    #    sys.exit()

                    else:  # we need to fill later
                        this_tech_dneeg_vals = \
                            [0 for y in range(len(time_vector))]
                        store_percent_rem.update({tech:this_tech_dneeg_vals})
                    store_percent.update({tech:this_tech_dneeg_vals})


                    # if tech == 'PP_PV Utility_Solar':
                    #    print('review this')
                    #    sys.exit()


            # ...here we need to run the remainder if necessary:
            if check_percent is True:
                tech_rem = list(store_percent_rem.keys())[0]
                oneminus_rem_list = store_percent_rem[tech_rem]
                for tech in list_electric_sets_3:
                    if tech != tech_rem:
                        for y in range(len(time_vector)):
                            oneminus_rem_list[y] += store_percent[tech][y]

                for y in range(len(time_vector)):
                    store_percent[tech_rem][y] = 1-oneminus_rem_list[y]

                # if this_scen == 'NDCPLUS':
                #    print('review this please')
                #    sys.exit()

            # ...we should store the BAU's "store percent" approach:
            if 'BAU' in this_scen:
                store_percent_BAU.update({this_country:deepcopy(store_percent)})

            # ...below, we need to apply an adjustment factor to match supply and demand:
            adjustment_factor = base_electric_prod/store_res_energy_all[0]
            for y in range(len(time_vector)):
                store_res_energy_all[y] *= adjustment_factor
                for tech in list_electric_sets_3:
                    store_res_energy[tech][y] *= adjustment_factor
                    cf_by_tech[tech][y] *= adjustment_factor

            # ...here we need to iterate across the list of technologies:
            fuel_use_electricity = {}  # crucial outputs
            externalities_globalwarming_electricity = {}  # crucial output
            externalities_localpollution_electricity = {}  # crucial output
            emissions_electricity = {}  # crucial output
            total_capacity = {}  # crucial output
            residual_capacity = {}  # crucial output
            new_capacity = {}  # crucial output

            # ...capacity disaggregation:
            cap_new_unplanned = {}
            cap_new_planned = {}
            cap_phase_out = {}

            # ...debugging dictionaries:
            ele_prod_share = {}
            ele_endogenous = {}
            cap_accum = {}

            total_production = {}  # crucial output
            new_production = {}
            capex = {}  # crucial output
            fopex = {}  # crucial output
            vopex = {}  # crucial output
            gcc = {}  # crucial output

            # Create dictionaries to store data from printing
            idict_u_capex = {}
            idict_u_fopex = {}
            idict_u_vopex = {}
            idict_u_gcc = {}
            idict_cau = {}
            idict_net_cap_factor = {}
            idict_hr = {}
            idict_oplife = {}

            # ...create a variable to represent lock-in decisions
            accum_cap_energy_vector = [0 for y in range(len(time_vector))]

            # 3rd power sector loop spotting conditions of surpassing capacity potential
            # ...we copy some of the elements of the 4th power sector loop
            # ...the unique sets with restriction:
            restriction_sets = list(set(df4_caps_rest['Set'].tolist()))

            # ...define the "adjustment_fraction" to recalculate the production shares
            adjustment_fraction = {}
            
            unit_gen_cost_dict = {}
            
            for tech in list_electric_sets_3:
                adjustment_fraction.update({tech:1})

                # ...extract the indices of the technology to define capacity:
                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_apply_type = this_tech_dneeg_df_param_related['apply_type']
                this_tech_dneeg_projection = this_tech_dneeg_df_param_related['projection']
                this_tech_dneeg_value_type = this_tech_dneeg_df_param_related['value']

                try:
                    tech_idx_8 = df_param_related_8['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_8 = ''
                try:
                    tech_idx_9 = df_param_related_9['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_9 = ''
                tech_counter += 1

                # This is equivalent to parameter 8:
                this_tech_forced_new_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_8 != '':
                    for y in range(len(time_vector)):
                        this_tech_forced_new_cap[y] = \
                            df_param_related_8.loc[tech_idx_8, time_vector[y]]/1000

                # This is equivalent to parameter 9:
                this_tech_phase_out_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_9 != '':
                    for y in range(len(time_vector)):
                        this_tech_phase_out_cap[y] = \
                            df_param_related_9.loc[tech_idx_9, time_vector[y]]/1000

                # ...extract the capacity restriction (by 2050)
                mask_restriction = ((df4_caps_rest['Set'] == tech) &\
                                    (df4_caps_rest['Country'] == this_country))
                restriction_value_df = df4_caps_rest.loc[mask_restriction]
                restriction_value_df.reset_index(drop=True, inplace=True)
                if len(restriction_value_df.index.tolist()) != 0:
                    restriction_value = restriction_value_df['Restriction (MW)'].iloc[0]/1000
                else:
                    restriction_value = 999999999

                # ...extract the "Net capacity factor"
                list_tech_cf = cf_by_tech[tech]

                # ...here extract the technical characteristics of the power techs
                mask_this_tech = \
                    (d5_power_techs['Tech'] == tech)
                this_tech_df_cost_power_techs = \
                    d5_power_techs.loc[mask_this_tech]

                # ...extract the CAU
                mask_cau = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAU')
                df_mask_cau = \
                    deepcopy(this_tech_df_cost_power_techs.loc[mask_cau])
                df_mask_cau_unitmult = df_mask_cau.iloc[0]['Unit multiplier']
                df_mask_cau_proj = df_mask_cau.iloc[0]['Projection']
                list_tech_cau = []
                if df_mask_cau_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cau.iloc[0][y]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)
                if df_mask_cau_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cau.iloc[0][time_vector[0]]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)

                # Variable FOM
                mask_vfom = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Variable FOM')
                df_mask_vfom = \
                    this_tech_df_cost_power_techs.loc[mask_vfom]
                df_mask_vfom_unitmult = df_mask_vfom.iloc[0]['Unit multiplier']
                df_mask_vfom_proj = df_mask_vfom.iloc[0]['Projection']
                list_tech_vfom = []
                if df_mask_vfom_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_vfom.iloc[0][y]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)
                if df_mask_vfom_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_vfom.iloc[0][time_vector[0]]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)

                # Heat Rate
                mask_hr = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Heat Rate')
                df_mask_hr = \
                    this_tech_df_cost_power_techs.loc[mask_hr]
                if len(df_mask_hr.index.tolist()) != 0:
                    df_mask_hr_unitmult = df_mask_hr.iloc[0]['Unit multiplier']
                    df_mask_hr_proj = df_mask_hr.iloc[0]['Projection']
                    list_tech_hr = []
                    if df_mask_hr_proj == 'user_defined':
                        for y in time_vector:
                            add_value = \
                                df_mask_hr.iloc[0][y]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                    if df_mask_hr_proj == 'flat':
                        for y in range(len(time_vector)):
                            add_value = \
                                df_mask_hr.iloc[0][time_vector[0]]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                else:
                    list_tech_hr = [0 for y in range(len(time_vector))]

                '''               
                # NOTE: these are fuels with costs
                > Diesel, Fuel Oil, Coal, Natural Gas
                # If the costs had not been introduced when defining the technology, we can define them here
                '''
                if tech == 'PP_Thermal_Fuel oil':
                    df_cost_fueloil = df_param_related_7.loc[(df_param_related_7['Fuel'] == 'Fuel Oil')][time_vector]
                    list_cost_fueloil = df_cost_fueloil.iloc[0].tolist()
                    list_tech_vfom_orig = deepcopy(list_tech_vfom)
                    if sum(list_tech_vfom) == 0:
                        list_tech_vfom = deepcopy(list_cost_fueloil)
                    if sum(list_tech_hr) == 0:
                        print('Simulation of capacity factor variation will not make sense because of Fuel oil')
                        sys.exit()
                if tech == 'PP_Thermal_Diesel':
                    df_cost_diesel = df_param_related_7.loc[(df_param_related_7['Fuel'] == 'Diesel')][time_vector]
                    list_cost_diesel = df_cost_diesel.iloc[0].tolist()
                    if sum(list_tech_vfom) == 0:
                        list_tech_vfom = deepcopy(list_cost_diesel)
                    if sum(list_tech_hr) == 0:
                        print('Simulation of capacity factor variation will not make sense because of Diesel')
                        sys.exit()
                if tech == 'PP_Thermal_Natural Gas':
                    df_cost_ngas = df_param_related_7.loc[(df_param_related_7['Fuel'] == 'Natural Gas')][time_vector]
                    list_cost_ngas = df_cost_ngas.iloc[0].tolist()
                    if sum(list_tech_vfom) == 0:
                        list_tech_vfom = deepcopy(list_cost_ngas)
                    if sum(list_tech_hr) == 0:
                        print('Simulation of capacity factor variation will not make sense because of Natural gas')
                        sys.exit()
                if tech == 'PP_Thermal_Coal':
                    df_cost_coal = df_param_related_7.loc[(df_param_related_7['Fuel'] == 'Coal')][time_vector]
                    list_cost_coal = df_cost_coal.iloc[0].tolist()
                    if sum(list_tech_vfom) == 0:
                        list_tech_vfom = deepcopy(list_cost_coal)
                    if sum(list_tech_hr) == 0:
                        print('Simulation of capacity factor variation will not make sense because of Coal')
                        sys.exit()

                # Store the unit generation cost:               
                unit_gen_cost = [a * b for a, b in zip(list_tech_hr, list_tech_vfom)]  # most likely in $/PJ
                unit_gen_cost_dict.update({tech: deepcopy(unit_gen_cost)})

                # ...define some intermediate variables; some are redefined later
                this_tech_accum_new_cap_unplanned = [0 for y in range(len(time_vector))]
                accum_cap_energy_vector = [0 for y in range(len(time_vector))]

                this_tech_accum_new_cap = [0 for y in range(len(time_vector))]
                this_tech_new_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_cap_unplanned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_new_cap_planned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_residual_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_prod = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_prod = [0 for y in range(len(time_vector))]

                # ...apply the capacity estimation algorithm:
                # if this_scen == 'NDCPLUS':
                #    print('review if anything here is wrong here')
                #    sys.exit()
                    
                if (this_tech_dneeg_projection == 'interpolate' and
                        this_tech_dneeg_value_type == 'percent' and
                        this_tech_dneeg_apply_type == 'all') or (
                        this_tech_dneeg_projection == 'keep_proportions') or (
                        this_tech_dneeg_projection == 'user_defined'):

                    for y in range(len(time_vector)):
                        # calculate the energy that the accumulated unplanned capacity supplies (this is actually unused)
                        if y != 0:
                            this_tech_accum_cap_energy = \
                                list_tech_cau[y]*list_tech_cf[y]*this_tech_accum_new_cap_unplanned[y-1]
                            accum_cap_energy_vector[y] += \
                                this_tech_accum_cap_energy

                        # ...estimate the energy requirement
                        new_req_energy = \
                            electrical_demand_to_supply[y] \
                            - forced_newcap_energy_all[y] \
                            - store_res_energy_all[y]

                        if new_req_energy < 0:
                            count_under_zero += 1
                            new_req_energy = 0

                        use_cap = store_use_cap[tech][y]  # cap with calibrated values
                        res_energy = store_res_energy[tech][y]  # energy that calibrated value produces (not considering new capacity)

                        planned_energy = forced_newcap_energy_by_tech[tech][y]  # energy from planned plants

                        if this_tech_dneeg_projection == 'keep_proportions':  # do not mix things up
                            if y == 0:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[0]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            else:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[y]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            energy_dist = res_energy_base/res_energy_sum  # distribution of energy for "keep_proportions"
                            new_energy_assign = new_req_energy*energy_dist
                        else:
                            new_energy_assign = new_req_energy*store_percent[tech][y]

                        this_tech_total_prod[y] = deepcopy(res_energy + planned_energy + new_energy_assign)
                        if y != 0:
                            this_tech_new_prod[y] = \
                                this_tech_total_prod[y] \
                                - this_tech_total_prod[y-1]
                            if this_tech_new_prod[y] < 0:
                                this_tech_new_prod[y] = 0

                        # Remembering how much should be subtracted
                        if y == 0:
                            subtract_new_cap = 0
                        else:
                            subtract_new_cap = this_tech_accum_new_cap_unplanned[y-1]

                        # Estimating unplanned capacity
                        if list_tech_cau[y]*list_tech_cf[y] != 0: #try:
                            new_cap_unplanned = \
                                new_energy_assign/(list_tech_cau[y]*list_tech_cf[y]) - \
                                subtract_new_cap
                        else:
                            print('division by zero', 'interpolate', 1)
                            sys.exit()

                        # This is a filter to avoid inconsistencies:
                        if new_cap_unplanned < 0:
                            new_cap_unplanned = 0

                        new_cap = new_cap_unplanned + forced_newcap_by_tech[tech][y]

                        # Update the residual capacity
                        if y == 0:
                            residual_cap = use_cap
                            this_tech_total_cap[y] = use_cap
                            this_tech_residual_cap[y] = use_cap
                        else:
                            residual_cap = this_tech_residual_cap[y-1] - this_tech_phase_out_cap[y]
                            this_tech_total_cap[y] += residual_cap + this_tech_accum_new_cap[y-1]
                            this_tech_residual_cap[y] = residual_cap

                        # Adjust accumulated new capacities
                        if y == 0:
                            this_tech_accum_new_cap[y] = new_cap
                            this_tech_accum_new_cap_unplanned[y] = new_cap_unplanned
                        else:
                            this_tech_accum_new_cap[y] = \
                                new_cap + this_tech_accum_new_cap[y-1]
                            this_tech_accum_new_cap_unplanned[y] = \
                                new_cap_unplanned + this_tech_accum_new_cap_unplanned[y-1]

                        this_tech_new_cap[y] += new_cap
                        this_tech_total_cap[y] += new_cap

                        this_tech_new_cap_unplanned[y] = deepcopy(new_cap_unplanned)
                        this_tech_new_cap_planned[y] = deepcopy(forced_newcap_by_tech[tech][y])

                    # ...below we must assess if we need to recalculate the shares:
                    if this_tech_total_cap[-1] < restriction_value:
                        pass  # there is no need to do anything else
                    elif this_tech_total_cap[-1] <= 0:
                        pass  # there is no need to specify anything
                    else:  # we must re-estimate the shares
                        this_adjustment_fraction = \
                            (restriction_value-sum(this_tech_new_cap_planned)) / (this_tech_total_cap[-1] - this_tech_total_cap[0])
                        adjustment_fraction[tech] = this_adjustment_fraction

            # With the adjustment factor complete, we proceed to update:
            sum_adjustment = [0 for y in range(len(time_vector))]
            cum_adjusted = [0 for y in range(len(time_vector))]
            old_share = {}
            new_share = {}
            for tech in list_electric_sets_3:
                old_share.update({tech: [0 for y in range(len(time_vector))]})
                new_share.update({tech: [0 for y in range(len(time_vector))]})
                for y in range(len(time_vector)):
                    if adjustment_fraction[tech] < 1:
                        new_share[tech][y] = adjustment_fraction[tech]*store_percent[tech][y]
                        cum_adjusted[y] += adjustment_fraction[tech]*store_percent[tech][y]
                    else:
                        sum_adjustment[y] += store_percent[tech][y]
                    old_share[tech][y] = store_percent[tech][y]
            for tech in list_electric_sets_3:
                for y in range(len(time_vector)):
                    if adjustment_fraction[tech] < 1:
                        pass  # there is nothing to do here; previous loop was done
                    else:
                        new_share[tech][y] = store_percent[tech][y]*((1-cum_adjusted[y])/(sum_adjustment[y]))

            old_share_sum = [0 for y in range(len(time_vector))]
            new_share_sum = [0 for y in range(len(time_vector))]
            for tech in list_electric_sets_3:
                for y in range(len(time_vector)):
                    old_share_sum[y] += old_share[tech][y]
                    new_share_sum[y] += new_share[tech][y]

            store_percent_freeze = deepcopy(store_percent)
            for tech in list_electric_sets_3:
                for y in range(len(time_vector)):
                    if time_vector[y] <= ini_simu_yr:  # this is to control the first few years
                        pass
                    elif old_share[tech][y] == 0:
                        pass
                    else:
                        store_percent[tech][y] *= \
                            new_share[tech][y]/old_share[tech][y]

            # Now let us sort across technologies to have the sorted cost
            # Create the dictionaries per year with sorted techs
            yearly_sorted_tech_costs = {}
            for year in range(len(next(iter(unit_gen_cost_dict.values())))):  # Using the length of one of the tech lists to determine the number of years
                sorted_techs = sorted(unit_gen_cost_dict.keys(), key=lambda tech: unit_gen_cost_dict[tech][year], reverse=True)
                yearly_sorted_tech_costs[year] = sorted_techs

            # 4th power sector loop
            list_electric_sets_3.sort()
            thermal_filter_out = [
                'PP_Thermal_Diesel',
                'PP_Thermal_Fuel oil',
                'PP_Thermal_Coal',
                'PP_Thermal_Crude',
                'PP_Thermal_Natural Gas']
            list_electric_sets_3_shuffle = [item for item in list_electric_sets_3 if item not in thermal_filter_out]
            list_electric_sets_3_shuffle += thermal_filter_out

            # Remove the specified item
            item_to_move = 'PP_PV Utility+Battery_Solar'
            list_electric_sets_3_shuffle.remove(item_to_move)

            # Append it to the end
            list_electric_sets_3_shuffle.append(item_to_move)
            
            thermal_reductions_store = {}
            thermal_reductions_order = {}
            
            tech_counter = 0
            for tech in list_electric_sets_3_shuffle:

                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                # Extract planned and phase out capacities if they exist:
                try:
                    tech_idx_8 = df_param_related_8['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_8 = ''
                try:
                    tech_idx_9 = df_param_related_9['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_9 = ''
                tech_counter += 1

                # ...here extract the technical characteristics of the power techs
                mask_this_tech = \
                    (d5_power_techs['Tech'] == tech)
                this_tech_df_cost_power_techs = \
                    d5_power_techs.loc[mask_this_tech]

                # ...we can extract one parameter at a time:
                # CAPEX
                mask_capex = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAPEX')
                df_mask_capex = \
                    this_tech_df_cost_power_techs.loc[mask_capex]
                df_mask_capex_unitmult = df_mask_capex.iloc[0]['Unit multiplier']
                df_mask_capex_proj = df_mask_capex.iloc[0]['Projection']
                list_tech_capex = []
                if df_mask_capex_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_capex.iloc[0][y]*df_mask_capex_unitmult
                        list_tech_capex.append(add_value)
                if df_mask_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_capex.iloc[0][time_vector[0]]*df_mask_capex_unitmult
                        list_tech_capex.append(add_value)

                # CAU
                mask_cau = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAU')
                df_mask_cau = \
                    this_tech_df_cost_power_techs.loc[mask_cau]
                df_mask_cau_unitmult = df_mask_cau.iloc[0]['Unit multiplier']
                df_mask_cau_proj = df_mask_cau.iloc[0]['Projection']
                list_tech_cau = []
                if df_mask_cau_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cau.iloc[0][y]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)
                if df_mask_cau_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cau.iloc[0][time_vector[0]]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)

                # Fixed FOM
                mask_ffom = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Fixed FOM')
                df_mask_ffom = \
                    this_tech_df_cost_power_techs.loc[mask_ffom]
                df_mask_ffom_unitmult = df_mask_ffom.iloc[0]['Unit multiplier']
                df_mask_ffom_proj = df_mask_ffom.iloc[0]['Projection']
                list_tech_ffom = []
                if df_mask_ffom_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_ffom.iloc[0][y]*df_mask_ffom_unitmult
                        list_tech_ffom.append(add_value)
                if df_mask_ffom_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_ffom.iloc[0][time_vector[0]]*df_mask_ffom_unitmult
                        list_tech_ffom.append(add_value)

                # Grid connection cost
                mask_gcc = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Grid connection cost')
                df_mask_gcc = \
                    this_tech_df_cost_power_techs.loc[mask_gcc]
                df_mask_gcc_unitmult = df_mask_gcc.iloc[0]['Unit multiplier']
                df_mask_gcc_proj = df_mask_gcc.iloc[0]['Projection']
                list_tech_gcc = []
                if df_mask_gcc_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_gcc.iloc[0][y]*df_mask_gcc_unitmult
                        list_tech_gcc.append(add_value)
                if df_mask_gcc_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_gcc.iloc[0][time_vector[0]]*df_mask_gcc_unitmult
                        list_tech_gcc.append(add_value)

                # Net capacity factor
                list_tech_cf = cf_by_tech[tech]

                # Operational life
                mask_ol = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Operational life')
                df_mask_ol = \
                    this_tech_df_cost_power_techs.loc[mask_ol]
                df_mask_ol_unitmult = df_mask_ol.iloc[0]['Unit multiplier']
                df_mask_ol_proj = df_mask_ol.iloc[0]['Projection']
                list_tech_ol = []
                if df_mask_ol_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_ol.iloc[0][y]*df_mask_ol_unitmult
                        list_tech_ol.append(add_value)
                if df_mask_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_ol.iloc[0][time_vector[0]]*df_mask_ol_unitmult
                        list_tech_ol.append(add_value)

                # Variable FOM
                mask_vfom = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Variable FOM')
                df_mask_vfom = \
                    this_tech_df_cost_power_techs.loc[mask_vfom]
                df_mask_vfom_unitmult = df_mask_vfom.iloc[0]['Unit multiplier']
                df_mask_vfom_proj = df_mask_vfom.iloc[0]['Projection']
                list_tech_vfom = []
                if df_mask_vfom_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_vfom.iloc[0][y]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)
                if df_mask_vfom_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_vfom.iloc[0][time_vector[0]]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)

                # Heat Rate
                mask_hr = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Heat Rate')
                df_mask_hr = \
                    this_tech_df_cost_power_techs.loc[mask_hr]
                if len(df_mask_hr.index.tolist()) != 0:
                    df_mask_hr_unitmult = df_mask_hr.iloc[0]['Unit multiplier']
                    df_mask_hr_proj = df_mask_hr.iloc[0]['Projection']
                    list_tech_hr = []
                    if df_mask_hr_proj == 'user_defined':
                        for y in time_vector:
                            add_value = \
                                df_mask_hr.iloc[0][y]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                    if df_mask_hr_proj == 'flat':
                        for y in range(len(time_vector)):
                            add_value = \
                                df_mask_hr.iloc[0][time_vector[0]]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                else:
                    list_tech_hr = [0 for y in range(len(time_vector))]

                # ...we next need to incorporate the heat rate data into the variable opex //
                # hence, link the commodity to the technology fuel consumption

                # ...storing the power plant information for printing
                idict_u_capex.update({tech:deepcopy(list_tech_capex)})
                idict_u_fopex.update({tech:deepcopy(list_tech_ffom)})
                idict_u_vopex.update({tech:deepcopy(list_tech_vfom)})
                idict_u_gcc.update({tech:deepcopy(list_tech_gcc)})
                idict_cau.update({tech:deepcopy(list_tech_cau)})
                # idict_net_cap_factor.update({tech:deepcopy(list_tech_cf)})
                idict_hr.update({tech:deepcopy(list_tech_hr)})
                idict_oplife.update({tech:deepcopy(list_tech_ol)})

                # idict_net_cap_factor_by_scen_by_country[this_scen][this_country] = deepcopy(idict_net_cap_factor)

                # ...acting for "Distribution of new electrical energy generation" (_dneeg)
                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_apply_type = this_tech_dneeg_df_param_related['apply_type']
                this_tech_dneeg_projection = this_tech_dneeg_df_param_related['projection']
                this_tech_dneeg_value_type = this_tech_dneeg_df_param_related['value']
                this_tech_dneeg_known_vals_raw = []
                this_tech_dneeg_known_vals = []
                use_fuel = this_tech_dneeg_df_param_related['Fuel']

                this_tech_base_cap = dict_base_caps[tech][base_year]
                this_tech_accum_new_cap = [0 for y in range(len(time_vector))]
                this_tech_accum_new_cap_unplanned = [0 for y in range(len(time_vector))]
                this_tech_new_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)

                this_tech_new_cap_unplanned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_new_cap_planned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_energy_dist = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_total_endo = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_residual_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_prod = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_prod = [0 for y in range(len(time_vector))]

                reno_ene_delta_add = [0] * len(time_vector)

                # This is equivalent to parameter 8:
                this_tech_forced_new_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_8 != '':
                    for y in range(len(time_vector)):
                        this_tech_forced_new_cap[y] = \
                            df_param_related_8.loc[tech_idx_8, time_vector[y]]/1000

                # This is equivalent to parameter 9:
                this_tech_phase_out_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_9 != '':
                    for y in range(len(time_vector)):
                        this_tech_phase_out_cap[y] = \
                            df_param_related_9.loc[tech_idx_9, time_vector[y]]/1000

                if (this_tech_dneeg_projection == 'interpolate' and
                        this_tech_dneeg_value_type == 'percent' and
                        this_tech_dneeg_apply_type == 'all') or (
                        this_tech_dneeg_projection == 'keep_proportions') or (
                        this_tech_dneeg_projection == 'user_defined'):

                    # REVIEW THIS
                    new_req_energy_list = []
                    new_req_energy_list_2 = []
                    new_ene_assign_list = []


                    for y in range(len(time_vector)):
                        # calculate the energy that the accumulated unplanned capacity supplies (this is actually unused)
                        if y != 0:
                            this_tech_accum_cap_energy = \
                                list_tech_cau[y]*list_tech_cf[y]*this_tech_accum_new_cap_unplanned[y-1]
                            accum_cap_energy_vector[y] += \
                                this_tech_accum_cap_energy
                    
                        # ...estimate the energy requirement
                        new_req_energy = \
                            electrical_demand_to_supply[y] \
                            - forced_newcap_energy_all[y] \
                            - store_res_energy_all[y]
                        #     - store_unplanned_energy_all[y]

                        new_req_energy_list.append(new_req_energy)
                    
                        # It is convenient to call the capacity here
                        if y > 0:
                            ref_current_cap = this_tech_total_cap[y-1]
                        else:
                            ref_current_cap = store_use_cap[tech][y]
                    
                        # Here we must add the capacity factor adjustment for thermal, such that the renewable target is met and no unnecessary capacity is further planned
                        # Also, considerations about renewable targets must be established
                        '''
                        There are basically 2 options to increase renewability:
                        - reduce thermal capacity factors
                        - increase renewable generation
                        
                        Furthermore, if more production than needed occurs, the thermal capacity factors can be reduced
                        '''
                    
                        # Establish technical capacity factor minimums to cope with reductions:
                        # Take the elements from the power plants:
                        max_cf_dict = {
                            'PP_Thermal_Diesel': 0.0001,
                            'PP_Thermal_Fuel oil': 0.000001,
                            'PP_Thermal_Coal': 0.0001,
                            'PP_Thermal_Crude': 0.0001,
                            'PP_Thermal_Natural Gas': 0.0001}
                    
                        # The sorted thermal technologies will come in handy
                        # Worst-to-best variable cost technologies:
                        wtb_tech_list = yearly_sorted_tech_costs[y][:5]
                        # We need to reshuffle this list so we remove nuclear and introduce Crude;
                        # this may have to change in the future when we update the model with costs.
                        wtb_tech_list = ['PP_Thermal_Diesel',
                            'PP_Thermal_Fuel oil', 'PP_Thermal_Coal',
                            'PP_Thermal_Crude', 'PP_Thermal_Natural Gas']
                        
                        if tech_counter == 1:
                            thermal_reductions_order.update({y:1})

                        # if this_country_2 == 'Costa Rica' and y == 0:
                        #     print('find out the original capacity factor *')
                        #     print(y,tech,cf_by_tech['PP_Thermal_Natural Gas'])
                        #     sys.exit()


                        if new_req_energy < 0:
                            count_under_zero += 1
                            if tech_counter == 1:
                                thermal_reductions = abs(deepcopy(new_req_energy))
                                thermal_reductions_store.update({y:deepcopy(thermal_reductions)})
                            else:
                                try:
                                    thermal_reductions = deepcopy(thermal_reductions_store[y])
                                except Exception:
                                    thermal_reductions = 0
                            new_req_energy = 0
                            
                            if this_scen == 'BAU':
                                thermal_reductions = 0
                                
                            if tech in wtb_tech_list:
                                max_cf_for_reductions = max_cf_dict[tech]
                                list_th_tech_cf = deepcopy(cf_by_tech[tech])
                                if list_th_tech_cf[y] < max_cf_for_reductions:
                                    thermal_reductions_order[y] += 1


                            # This indicates that thermal generation has an opportunity for reduction, i.e., reduce until technical minimums are met
                            enter_reduction_conditionals = False
                            if tech == wtb_tech_list[0] and thermal_reductions_order[y] == 1:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[1] and thermal_reductions_order[y] == 2:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[2] and thermal_reductions_order[y] == 3:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[3] and thermal_reductions_order[y] == 4:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[4] and thermal_reductions_order[y] == 5:
                                enter_reduction_conditionals = True
                            if enter_reduction_conditionals and list_tech_cf[y] > max_cf_for_reductions and tech in wtb_tech_list and thermal_reductions > 0:
                                
                                
                                # print('THIS HAPPENED? 1', time_vector[y], thermal_reductions)
                                
                                curr_energy_tech = list_tech_cau[y]*list_tech_cf[y]*ref_current_cap
                                min_energy_tech = list_tech_cau[y]*max_cf_for_reductions*ref_current_cap
                                max_red_tech = curr_energy_tech - min_energy_tech
                                if max_red_tech >= thermal_reductions:
                                    new_cf = (curr_energy_tech - thermal_reductions)/(list_tech_cau[y]*ref_current_cap)
                                    thermal_reductions = 0
                                else:
                                    new_cf = max_cf_for_reductions
                                    thermal_reductions -= max_red_tech
                                    thermal_reductions_order[y] += 1
                                # for y_aux_cf in range(y, len(time_vector)):
                                # for y_aux_cf in range(y, y+1):
                                for y_aux_cf in range(y, len(time_vector)):
                                    list_tech_cf[y_aux_cf] = deepcopy(new_cf)
                                    store_res_energy[tech][y_aux_cf] *= new_cf/cf_by_tech[tech][y_aux_cf]
                                    forced_newcap_energy_by_tech[tech][y_aux_cf] *= new_cf/cf_by_tech[tech][y_aux_cf]

                                cf_by_tech[tech] = deepcopy(list_tech_cf)
                    
                            if y in list(thermal_reductions_store.keys()):
                                thermal_reductions_store[y] = deepcopy(thermal_reductions)
                    
                        else:
                            thermal_reductions_store.update({y:0})
                    
                        new_req_energy_list_2.append(new_req_energy)
                    
                        '''
                        NOTE:
                        This means that the excess energy must be covered.
                        We design for an excess considering the additional planned capacity out of the excess.
                        '''
                    
                        # NOTE: "store_res_energy_all" has the production of the residual capacity
                    
                        # Instead of distribution to keep proportion, we need to proceed with the interpolation
                        # of the shares that this source will produce:
                        use_cap = store_use_cap[tech][y]  # cap with calibrated values
                        res_energy = store_res_energy[tech][y]  # energy that calibrated value produces (not considering new capacity)
                        res_energy_change = 0
                        if y > 0:
                            res_energy_change = store_res_energy[tech][y] - store_res_energy[tech][y-1]
                    
                        planned_energy = forced_newcap_energy_by_tech[tech][y]  # energy from planned plants
                    
                        if this_tech_dneeg_projection == 'keep_proportions':  # do not mix things up
                            if y == 0:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[0]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            else:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[y]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            energy_dist = res_energy_base/res_energy_sum  # distribution of energy for "keep_proportions"
                            new_energy_assign = new_req_energy*energy_dist
                        else:
                            new_energy_assign = new_req_energy*store_percent[tech][y]
                    
                        new_ene_assign_list.append(new_energy_assign)
                        
                        if tech_counter < len(list_electric_sets_3_shuffle):
                            this_tech_total_prod[y] = deepcopy(res_energy + planned_energy + new_energy_assign)
                        
                        if this_scen == 'BAU':
                            cf_ngas_max = 0.8
                        else:
                            cf_ngas_max = 0.8  # more backup operation
                        # stop_check = False
                        # stop_check = True
                        # if this_country_2 == 'Costa Rica' and y == 2 and stop_check is False:
                        #     print('what happened  here?')
                        #     sys.exit()
                        
                        # if this_country_2 == 'Costa Rica' and y == 0:
                        #     print('find out the original capacity factor')
                        #     print(y,tech,cf_by_tech['PP_Thermal_Natural Gas'])
                        #     sys.exit()
                        # if this_country_2 == 'Costa Rica':
                        #     print(y,tech,cf_by_tech['PP_Thermal_Natural Gas'])
                        
                        if tech == 'PP_Thermal_Natural Gas' and new_energy_assign > 0 and cf_by_tech[tech][y] < cf_ngas_max and y > 0:
                            cf_original = deepcopy(cf_by_tech[tech][y])
                            cf_new_changed = deepcopy(cf_by_tech[tech][y])
                            if (res_energy + planned_energy) > 0:
                                cf_new_changed *= (res_energy + planned_energy + new_energy_assign)/(res_energy + planned_energy)
                            else:
                                cf_new_changed = deepcopy(cf_ngas_max)
                            cf_new_changed_original = deepcopy(cf_new_changed)
                            if cf_new_changed > cf_ngas_max:
                                cf_new_changed = deepcopy(cf_ngas_max)
                            # for y_aux_cf in range(y, len(time_vector)):
                            # for y_aux_cf in range(y, y+1):
                            for y_aux_cf in range(y, len(time_vector)):
                                list_tech_cf[y_aux_cf] = deepcopy(cf_new_changed)
                                store_res_energy[tech][y_aux_cf] *= cf_new_changed/cf_by_tech[tech][y_aux_cf]
                                forced_newcap_energy_by_tech[tech][y_aux_cf] *= cf_new_changed/cf_by_tech[tech][y_aux_cf]
                            cf_by_tech[tech] = deepcopy(list_tech_cf)

                            # if this_country_2 == 'Costa Rica':
                            #    print('check natural gas')
                            #    sys.exit()

                        # if tech == 'PP_Hydro':
                        #   print(time_vector[y], res_energy, planned_energy, new_energy_assign, new_req_energy)
                        
                        # Here the new energy assign of renewables can increase to meet the renewable targets:
                        if tech_counter == len(list_electric_sets_3_shuffle) and reno_targets_exist:
                            
                            # print('count y until here')
                            
                            # sys.exit()
                            
                            list_electric_sets_3_shuffle_rest = list_electric_sets_3_shuffle[:-1]
                            # Here we must find out about the renewable generation
                            reno_gen = [0] * len(time_vector)
                            all_gen_reno_target = [0] * len(time_vector)
                            for suptech in list_electric_sets_3_shuffle_rest:
                                if 'Solar' in suptech or 'Geo' in suptech or 'Sugar' in suptech or 'Wind' in suptech or 'Hydro' in suptech:
                                    reno_gen = [a + b for a, b in zip(reno_gen, total_production[suptech])]
                                all_gen_reno_target = [a + b for a, b in zip(all_gen_reno_target, total_production[suptech])]

                            reno_gen = [a + b for a, b in zip(reno_gen, this_tech_total_prod)]
                            
                            reno_est_demand_to_supply = [100*a / b for a, b in zip(reno_gen, electrical_demand_to_supply)]
                            reno_est = [100*a / b for a, b in zip(reno_gen, all_gen_reno_target)]

                            # We can compare the percentage of renewables
                            if isinstance(reno_target_list[y], (float, np.floating, int)):
                                if reno_est[y] < reno_target_list[y] and not np.isnan(reno_target_list[y]):
                                    print('We need to increase renewability! Basically replace thermal generation with renewable generation for THIS tech.')
                                    # First let's calculate the energy swap required, which is similar to "thermal_reductions":
                                    reno_ene_delta_demand_based = ((reno_target_list[y] - reno_est[y])/100) * electrical_demand_to_supply[y]
                                    
                                    # The 'reno_ene_delta_demand_based' assumes a perfect match between generation and demand, but there are some mismatches.
                                    # To override the mismatches, let's estimate the difference based on total production                                  
                                    reno_ene_delta = ((reno_target_list[y] - reno_est[y])/100) * all_gen_reno_target[y]
                                    
                                    # print('THIS HAPPENED? 2', time_vector[y], thermal_reductions)
                                    
                                    # Then, let us update the important energy variables that control capacity expansion:
                                    # for y_aux_cf in range(y, len(time_vector)):
                                    # for y_aux_cf in range(y, y+1):
                                    for y_aux_cf in range(y, len(time_vector)):
                                        this_tech_total_prod[y_aux_cf] += deepcopy(reno_ene_delta)
                                        reno_ene_delta_add[y_aux_cf] += deepcopy(reno_ene_delta)
                                    new_energy_assign += deepcopy(reno_ene_delta)
                                    new_ene_assign_list[-1] = deepcopy(new_energy_assign)
                                    
                                    thermal_reductions_2 = deepcopy(reno_ene_delta)
                                    
                                    for th_tech in wtb_tech_list:
                                        max_cf_for_reductions = max_cf_dict[th_tech]
                                        list_th_tech_cf = deepcopy(cf_by_tech[th_tech])
                                        if list_th_tech_cf[y] <= max_cf_for_reductions:
                                            thermal_reductions_order[y] += 1
                                        enter_reduction_conditionals = False
                                        
                                        print(th_tech, thermal_reductions_order[y], list_th_tech_cf[y], max_cf_for_reductions)
                                        
                                        if th_tech == wtb_tech_list[0] and thermal_reductions_order[y] == 1:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[1] and thermal_reductions_order[y] == 2:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[2] and thermal_reductions_order[y] == 3:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[3] and thermal_reductions_order[y] == 4:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[4] and thermal_reductions_order[y] == 5:
                                            enter_reduction_conditionals = True
                                        if enter_reduction_conditionals and list_th_tech_cf[y] > max_cf_for_reductions and thermal_reductions_2 >= 0:
                                            # print('got in')
                                            curr_energy_tech = list_tech_cau[y]*list_th_tech_cf[y]*total_capacity[th_tech][y]
                                            min_energy_tech = list_tech_cau[y]*max_cf_for_reductions*total_capacity[th_tech][y]
                                            max_red_tech = curr_energy_tech - min_energy_tech
                                            if max_red_tech >= thermal_reductions_2:
                                                new_cf = (curr_energy_tech - thermal_reductions_2)/(list_tech_cau[y]*total_capacity[th_tech][y])
                                                thermal_reductions_2 = 0
                                                enter_reduction_conditionals = False
                                            else:
                                                new_cf = max_cf_for_reductions
                                                thermal_reductions_2 -= max_red_tech
                                                thermal_reductions_order[y] += 1
                                            # for y_aux_cf in range(y, len(time_vector)):
                                            # for y_aux_cf in range(y, y+1):
                                            for y_aux_cf in range(y, len(time_vector)):
                                                list_th_tech_cf[y_aux_cf] = deepcopy(new_cf)
                                                total_production[th_tech][y_aux_cf] *= new_cf/cf_by_tech[th_tech][y_aux_cf]
                                            cf_by_tech[th_tech] = deepcopy(list_th_tech_cf)
                    
                                            # # if 'PP_Nuclear' == th_tech:
                                            # if 'PP_Thermal_Natural Gas' == th_tech:
                                            #     print('check what is happening with the thermal production')
                                            #     sys.exit()
                    
                                    # print('REVIEW IMPACT')
                                    # sys.exit()

                                    # print_reno_test = True
                                    print_reno_test = False
                                    if print_reno_test:
                                        print('Writing a test verifying the the renewability has been reached according to the RE Target parameter')
                                        # Here we must write a test to check the renewability of the system
                                        reno_gen_verify = [0] * len(time_vector)
                                        all_gen_verify = [0] * len(time_vector)
                                        for suptech in list_electric_sets_3_shuffle_rest:
                                            if 'Solar' in suptech or 'Geo' in suptech or 'Sugar' in suptech or 'Wind' in suptech or 'Hydro' in suptech:
                                                print(suptech)
                                                reno_gen_verify = [a + b for a, b in zip(reno_gen_verify, total_production[suptech])]
                                            all_gen_verify = [a + b for a, b in zip(all_gen_verify, total_production[suptech])]
                                        reno_gen_verify = [a + b for a, b in zip(reno_gen_verify, this_tech_total_prod)]
                                        all_gen_verify = [a + b for a, b in zip(all_gen_verify, this_tech_total_prod)]

                                        ratio_all_gen_verify = [a / b for a, b in zip(reno_gen_verify, all_gen_verify)]
                                        ratio_electrical_demand_to_supply = [a / b for a, b in zip(reno_gen_verify, electrical_demand_to_supply)]

                                        index_2030 = time_vector.index(2030)
                                        ratio_all_gen_verify_2030 = ratio_all_gen_verify[index_2030]
                                        ratio_electrical_demand_to_supply_2030 = ratio_electrical_demand_to_supply[index_2030]

                                        # Take advantage of this area to calculate the difference between the electrical demand to supply and teh generation, in case there is an error
                                        diff_tot_sup_dem = [a - b for a, b in zip(all_gen_verify, electrical_demand_to_supply)]
                                        diff_tot_sup_err = [100*(a - b)/a for a, b in zip(all_gen_verify, electrical_demand_to_supply)]

                                        for suptech2 in list_electric_sets_3_shuffle_rest:
                                            print('>', suptech2, total_production[suptech2][y])

                                        print('Review elements that can be wrong')
                                        sys.exit()

                        if tech_counter == len(list_electric_sets_3_shuffle):
                            this_tech_total_prod[y] = deepcopy(res_energy + planned_energy + new_energy_assign)

                        # if this_scen == 'NDCPLUS' and y == len(time_vector)-1 and tech_counter == len(list_electric_sets_3_shuffle):
                        #    print('check demand balance')
                        #    sys.exit()

                        if y != 0 and tech not in thermal_filter_out:  # This is a debugging section
                            this_tech_new_prod[y] = \
                                this_tech_total_prod[y] \
                                - this_tech_total_prod[y-1]
                            # Some tolerance can be added to allow "negative CAPEX" in a small proportion:
                            tol_min_neg_capex_pj_ele = 0.1
                            if abs(this_tech_new_prod[y]) < tol_min_neg_capex_pj_ele and this_tech_new_prod[y] < 0 and time_vector[y] <= 2023:
                                print('An expected negative in generaton CAPEX occured!')
                                print(this_scen, tech, this_country)
                            elif this_tech_new_prod[y] < -tol_min_neg_capex_pj_ele and time_vector[y] > 2023 and res_energy_change < 0:
                                pass  # this is normal
                            elif this_tech_new_prod[y] < -tol_min_neg_capex_pj_ele and time_vector[y] > 2023 and res_energy_change >= 0:
                                # This means that more capacity probably exists than necessary, so for the punctual year, we adjust the capacity factor
                                ref_list_tech_cf = deepcopy(list_tech_cf[y])
                                if 'Solar' not in tech and 'Wind' not in tech:
                                    list_tech_cf[y] = this_tech_total_prod[y] / (this_tech_total_cap[y-1]*list_tech_cau[y-1])
                                    cf_by_tech[tech][y] = deepcopy(list_tech_cf[y])
                                mult_factor_cf_reno = list_tech_cf[y]/ref_list_tech_cf
                                # print(mult_factor_cf_reno)
                                # print('Does this inconsistency happen?', this_scen, tech, this_country)
                                # print(store_percent[tech])
                                # sys.exit()
                                # this_tech_new_prod[y] = 0
                    
                        # Remembering how much should be subtracted
                        if y == 0:
                            subtract_new_cap = 0
                        else:
                            subtract_new_cap = this_tech_accum_new_cap_unplanned[y-1]
                    
                        # Estimating unplanned capacity
                        if list_tech_cau[y]*list_tech_cf[y] != 0: #try:
                            new_cap_unplanned = \
                                new_energy_assign/(list_tech_cau[y]*list_tech_cf[y]) - \
                                subtract_new_cap
                        else:
                            print('division by zero', 'interpolate', 2)
                            sys.exit()

                        # if tech == 'PP_Hydro' and time_vector[y] == 2024:
                        #    print('Check this')
                        #    sys.exit()

                        # This is a filter to avoid inconsistencies:
                        if new_cap_unplanned < 0:
                            new_cap_unplanned = 0
                    
                        new_cap = new_cap_unplanned + forced_newcap_by_tech[tech][y]
                    
                        # Update the residual capacity
                        if y == 0:
                            residual_cap = use_cap
                            this_tech_total_cap[y] = use_cap
                            this_tech_residual_cap[y] = use_cap
                        else:
                            residual_cap = this_tech_residual_cap[y-1] - this_tech_phase_out_cap[y]
                            this_tech_total_cap[y] += residual_cap + this_tech_accum_new_cap[y-1]
                            this_tech_residual_cap[y] = residual_cap
                    
                        # Adjust accumulated new capacities
                        if y == 0:
                            this_tech_accum_new_cap[y] = new_cap
                            this_tech_accum_new_cap_unplanned[y] = new_cap_unplanned
                        else:
                            this_tech_accum_new_cap[y] = \
                                new_cap + this_tech_accum_new_cap[y-1]
                            this_tech_accum_new_cap_unplanned[y] = \
                                new_cap_unplanned + this_tech_accum_new_cap_unplanned[y-1]
                    
                        this_tech_new_cap[y] += new_cap
                        this_tech_total_cap[y] += new_cap
                    
                        this_tech_new_cap_unplanned[y] = deepcopy(new_cap_unplanned)
                        this_tech_new_cap_planned[y] = deepcopy(forced_newcap_by_tech[tech][y])
                    
                        # ...these are further debugging energy variables
                        this_tech_energy_dist[y] = deepcopy(store_percent[tech][y])
                    
                        # ...these are further debugging capacity/energy variables
                        for aux_y in range(y, len(time_vector)):
                            this_tech_total_endo[aux_y] += deepcopy(new_energy_assign)
                            store_unplanned_energy_all[aux_y] += deepcopy(new_energy_assign)


                    """
                    print(tech)
                    print(new_req_energy_list)
                    print(new_req_energy_list_2)
                    print(new_ene_assign_list)
                    print('\n')
                    # sys.exit()
                    """

                # ...we must now see the additional energy requirements of primary or secondary carriers because of total capacity
                if sum(list_tech_hr) != 0:  # means there is fuel a requirement:
                    list_use_fuel = []
                    for y in range(len(time_vector)):
                        add_value = \
                            this_tech_total_cap[y]*list_tech_cau[y]*list_tech_cf[y]*list_tech_hr[y]
                        list_use_fuel.append(add_value)
                    fuel_use_electricity.update({tech:{use_fuel:list_use_fuel}})
                else:
                    fuel_use_electricity.update({tech:'none'})

                # ...here we store the correspoding physical variables
                total_capacity.update({tech:this_tech_total_cap})
                residual_capacity.update({tech:this_tech_residual_cap})
                new_capacity.update({tech:this_tech_new_cap})

                cap_new_unplanned.update({tech:this_tech_new_cap_unplanned})
                cap_new_planned.update({tech:this_tech_new_cap_planned})
                cap_phase_out.update({tech:this_tech_phase_out_cap})

                total_production.update({tech:this_tech_total_prod})
                new_production.update({tech:this_tech_new_prod})

                # ...here we compute debugging variables:
                ele_prod_share.update({tech:this_tech_energy_dist})
                ele_endogenous.update({tech:this_tech_total_endo})
                cap_accum.update({tech:this_tech_accum_new_cap})

                # ...here we compute the costs by multiplying capacities times unit costs:
                this_tech_capex = [0 for y in range(len(time_vector))]
                this_tech_fopex = [0 for y in range(len(time_vector))]
                this_tech_vopex = [0 for y in range(len(time_vector))]
                this_tech_gcc = [0 for y in range(len(time_vector))]
                for y in range(len(time_vector)):
                    this_tech_capex[y] = this_tech_new_cap[y]*list_tech_capex[y]
                    this_tech_fopex[y] = this_tech_total_cap[y]*list_tech_ffom[y]
                    this_tech_vopex[y] = \
                        this_tech_total_prod[y]*list_tech_vfom[y]
                    this_tech_gcc[y] = this_tech_new_cap[y]*list_tech_gcc[y]

                # if tech == 'PP_Thermal_Crude':
                #     print('please stop here to check what is going on')
                #     sys.exit()

                capex.update({tech:this_tech_capex})
                fopex.update({tech:this_tech_fopex})
                vopex.update({tech:this_tech_vopex})
                gcc.update({tech:this_tech_gcc})

                # ...here we compute the externalities and emissions by multiplying fuel use times unit values:
                if sum(list_tech_hr) != 0:  # means there is fuel a requirement:
                    list_emissions = []
                    list_externalities_globalwarming = []
                    list_externalities_localpollution = []
                    for y in range(len(time_vector)):
                        if use_fuel in emissions_fuels_list:  # ...store emissions here
                            add_value_emissions = \
                                list_use_fuel[y]*emissions_fuels_dict[use_fuel][y]
                            list_emissions.append(add_value_emissions)

                            # ...besides, we must add variable costs from fuel consumption
                            fuel_idx_7 = df_param_related_7['Fuel'].tolist().index(use_fuel)
                            this_tech_vopex[y] += \
                                deepcopy(list_use_fuel[y]*df_param_related_7.loc[fuel_idx_7, time_vector[y]])

                            idict_u_vopex[tech][y] += \
                                deepcopy(df_param_related_7.loc[fuel_idx_7, time_vector[y]])

                        if use_fuel in externality_fuels_list:  # ...store externalities
                            add_value_globalwarming = \
                                list_use_fuel[y]*externality_fuels_dict[use_fuel]['Global warming']
                            add_value_localpollution = \
                                list_use_fuel[y]*externality_fuels_dict[use_fuel]['Local pollution']
                            list_externalities_globalwarming.append(add_value_globalwarming)
                            list_externalities_localpollution.append(add_value_localpollution)

                    emissions_electricity.update({tech:{use_fuel:list_emissions}})
                    externalities_globalwarming_electricity.update({tech:{use_fuel:list_externalities_globalwarming}})
                    externalities_localpollution_electricity.update({tech:{use_fuel:list_externalities_localpollution}})


            # Here we need to add the costs of fuel from non-electricity sectors
            # We need to use "dict_energy_demand" to calculate fuel activity.
            dict_energy_demand_cost = deepcopy(dict_energy_demand)
            dict_energy_demand_cost_disc = deepcopy(dict_energy_demand)
            dict_energy_demand_fuels = list(dict_energy_demand_cost['Transport'].keys())
            for sec in list(dict_energy_demand_cost.keys()):
                for this_fuel in list(dict_energy_demand_cost[sec].keys()):
                    dict_energy_demand_cost[sec][this_fuel] = [0] * len(time_vector)
                    dict_energy_demand_cost_disc[sec][this_fuel] = [0] * len(time_vector)
            fuel_list_demand_w_cost = df_param_related_7['Fuel'].tolist()
            sector_list_demand = list(dict_energy_demand.keys())
            for sec in sector_list_demand:
                for fuel_idx_7 in range(len(fuel_list_demand_w_cost)):
                    this_fuel_in_dem_cost = [0] * len(time_vector)
                    this_fuel_in_dem_cost_disc = [0] * len(time_vector)
                    this_fuel = df_param_related_7['Fuel'].tolist()[fuel_idx_7]
                    this_fuel_in_dem = dict_energy_demand[sec][this_fuel]
                    for y in range(len(time_vector)):
                        disc_constant = 1 / ((1 + r_rate/100)**(float(time_vector[y]) - r_year))
                        this_fuel_in_dem_cost[y] = \
                            deepcopy(this_fuel_in_dem[y]*df_param_related_7.loc[fuel_idx_7, time_vector[y]])
                        this_fuel_in_dem_cost_disc[y] = \
                            this_fuel_in_dem_cost[y] * disc_constant
                    dict_energy_demand_cost[sec][this_fuel] = deepcopy(this_fuel_in_dem_cost)
                    dict_energy_demand_cost_disc[sec][this_fuel] = deepcopy(this_fuel_in_dem_cost_disc)

            # We need to update the capacity factors after the modifications:
            for atech in list(cf_by_tech.keys()):
                idict_net_cap_factor.update({atech:deepcopy(cf_by_tech[atech])})
                idict_net_cap_factor_by_scen_by_country[this_scen][this_country] = deepcopy(idict_net_cap_factor)

            # 3i) Store the transport calculations:
            '''
            *Use these variables:*
            dict_fleet_k
            dict_new_fleet_k
            dict_capex_out
            dict_fopex_out
            dict_vopex_out
            
            Remember to apply this: dict_eq_transport_fuels
            '''
            if overwrite_transport_model:
                dict_tax_out_t1 = dict_tax_out['Imports']
                dict_tax_out_t2 = dict_tax_out['IMESI_Venta']
                dict_tax_out_t3 = dict_tax_out['IVA_Venta']
                dict_tax_out_t4 = dict_tax_out['Patente']
                dict_tax_out_t5 = dict_tax_out['IMESI_Combust']
                dict_tax_out_t6 = dict_tax_out['IVA_Gasoil']
                dict_tax_out_t7 = dict_tax_out['IVA_Elec']
                dict_tax_out_t8 = dict_tax_out['Impuesto_Carbono']
                dict_tax_out_t9 = dict_tax_out['Otros_Gasoil']
                dict_tax_out_t10 = dict_tax_out['Tasa_Consular']

                dict_local_country[this_country].update({'Fleet': deepcopy(dict_fleet_k)})
                dict_local_country[this_country].update({'New Fleet': deepcopy(dict_new_fleet_k)})
                dict_local_country[this_country].update({'Transport CAPEX [$]': deepcopy(dict_capex_out)})
                dict_local_country[this_country].update({'Transport Fixed OPEX [$]': deepcopy(dict_fopex_out)})
                dict_local_country[this_country].update({'Transport Variable OPEX [$]': deepcopy(dict_vopex_out)})
                dict_local_country[this_country].update({'Transport Tax Imports [$]': deepcopy(dict_tax_out_t1)})
                dict_local_country[this_country].update({'Transport Tax IMESI_Venta [$]': deepcopy(dict_tax_out_t2)})
                dict_local_country[this_country].update({'Transport Tax IVA_Venta [$]': deepcopy(dict_tax_out_t3)})
                dict_local_country[this_country].update({'Transport Tax Patente [$]': deepcopy(dict_tax_out_t4)})
                dict_local_country[this_country].update({'Transport Tax IMESI_Combust [$]': deepcopy(dict_tax_out_t5)})
                dict_local_country[this_country].update({'Transport Tax IVA_Gasoil [$]': deepcopy(dict_tax_out_t6)})
                dict_local_country[this_country].update({'Transport Tax IVA_Elec [$]': deepcopy(dict_tax_out_t7)})
                dict_local_country[this_country].update({'Transport Tax IC [$]': deepcopy(dict_tax_out_t8)})
                dict_local_country[this_country].update({'Transport Tax Otros_Gasoil [$]': deepcopy(dict_tax_out_t9)})
                dict_local_country[this_country].update({'Transport Tax Tasa_Consular [$]': deepcopy(dict_tax_out_t10)})

            # 3j) Store the data for printing:
            dict_local_country[this_country].update({'Electricity fuel use': deepcopy(fuel_use_electricity)})
            dict_local_country[this_country].update({'Global warming externalities in electricity': deepcopy(externalities_globalwarming_electricity)})
            dict_local_country[this_country].update({'Local pollution externalities in electricity': deepcopy(externalities_localpollution_electricity)})
            dict_local_country[this_country].update({'Emissions in electricity': deepcopy(emissions_electricity)})
            dict_local_country[this_country].update({'Electricity total capacity': deepcopy(total_capacity)})
            dict_local_country[this_country].update({'Electricity residual capacity': deepcopy(residual_capacity)})
            dict_local_country[this_country].update({'Electricity new capacity': deepcopy(new_capacity)})
            dict_local_country[this_country].update({'Electricity total production': deepcopy(total_production)})
            dict_local_country[this_country].update({'Electricity CAPEX': deepcopy(capex)})
            dict_local_country[this_country].update({'Electricity Fixed OPEX': deepcopy(fopex)})
            dict_local_country[this_country].update({'Electricity Variable OPEX': deepcopy(vopex)})
            dict_local_country[this_country].update({'Electricity Grid Connection Cost': deepcopy(gcc)})

            # ...disaggregate the new capacity:
            dict_local_country[this_country].update({'Electricity new capacity unplanned': deepcopy(cap_new_unplanned)})
            dict_local_country[this_country].update({'Electricity new capacity planned': deepcopy(cap_new_planned)})
            dict_local_country[this_country].update({'Electricity phase out capacity': deepcopy(cap_phase_out)})

            # ...let's store additional debugging variables per power plant (1):
            dict_local_country[this_country].update({'Electricity production share (unplanned)': deepcopy(ele_prod_share)})
            dict_local_country[this_country].update({'New energy assign': deepcopy(ele_endogenous)})
            dict_local_country[this_country].update({'Accumulated new capacity': deepcopy(cap_accum)})

            # ...let's store the "required energy" components:
            dict_local_country[this_country].update({'Electricity demand to supply': deepcopy(electrical_demand_to_supply)})
            dict_local_country[this_country].update({'Electricity planned supply': deepcopy(forced_newcap_energy_all)})

            # ...let's store additional debugging variables per power plant (2):
            dict_local_country[this_country].update({'Accumulated forced new capacity': deepcopy(accum_forced_newcap_by_tech)})
            dict_local_country[this_country].update({'Electricity planned supply per technology': deepcopy(forced_newcap_energy_by_tech)})
            dict_local_country[this_country].update({'Electricity residual supply': deepcopy(store_res_energy_all)})
            dict_local_country[this_country].update({'Electricity residual supply per tech': deepcopy(store_res_energy)})

            # *...here we need a supporting variable*:
            dict_local_country[this_country].update({'Electricity new production per tech': deepcopy(new_production)})

            # *...here we add a non-energy cost*:
            dict_local_country[this_country].update({'Energy demand cost': deepcopy(dict_energy_demand_cost)})
            dict_local_country[this_country].update({'Energy demand cost (disc)': deepcopy(dict_energy_demand_cost_disc)})

            # ...here we can execute the discount rate to 5 variables:
            '''
            'Global warming externalities in electricity'
            'Local pollution externalities in electricity'
            'Electricity CAPEX'
            'Electricity Fixed OPEX'
            'Electricity Variable OPEX'
            'Electricity Grid Connection Cost'
            '''
            disc_capex = deepcopy(capex)
            disc_fopex = deepcopy(fopex)
            disc_vopex = deepcopy(vopex)
            disc_gcc = deepcopy(gcc)
            disc_externalities_globalwarming_electricity = deepcopy(externalities_globalwarming_electricity)
            disc_externalities_localpollution_electricity = deepcopy(externalities_localpollution_electricity)

            '''
            # This is the generic equation you must apply:
            this_val_disc = this_value / ((1 + r_rate/100)**(float(this_year) - r_year))
            '''

            for y in range(len(time_vector)):
                this_year = int(time_vector[y])
                disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                for tech in list_electric_sets_3:  # extract the references
                    disc_capex[tech][y] *= disc_constant
                    disc_fopex[tech][y] *= disc_constant
                    disc_vopex[tech][y] *= disc_constant
                    disc_gcc[tech][y] *= disc_constant
                    for use_fuel in externality_fuels_list:
                        try:
                            disc_externalities_globalwarming_electricity[tech][use_fuel][y] *= disc_constant
                        except Exception:
                            pass  # in case the technology does not have an externality
                        try:
                            disc_externalities_localpollution_electricity[tech][use_fuel][y] *= disc_constant
                        except Exception:
                            pass  # in case the technology does not have an externality

            dict_local_country[this_country].update({'Electricity CAPEX (disc)': deepcopy(disc_capex)})
            dict_local_country[this_country].update({'Electricity Fixed OPEX (disc)': deepcopy(disc_fopex)})
            dict_local_country[this_country].update({'Electricity Variable OPEX (disc)': deepcopy(disc_vopex)})
            dict_local_country[this_country].update({'Electricity Grid Connection Cost (disc)': deepcopy(disc_gcc)})
            dict_local_country[this_country].update({'Global warming externalities in electricity (disc)': deepcopy(disc_externalities_globalwarming_electricity)})
            dict_local_country[this_country].update({'Local pollution externalities in electricity (disc)': deepcopy(disc_externalities_localpollution_electricity)})

            # At this point, we want to deal with the sum of local consumption of fossil
            # fuels for an assumption of exports change to the IROTE

            for a_fuel in list_fuel:
                dict_energy_demand_by_fuel_sum[a_fuel] = [
                    a + b for a, b in zip(
                        dict_energy_demand_by_fuel_sum[a_fuel],
                        dict_energy_demand_by_fuel[a_fuel])]

            # print('Record correct storage of overarching fuels')
            # sys.exit()

        dict_local_reg.update({this_reg: deepcopy(dict_local_country)})

    ###########################################################################
    # *Here we will implement the new IROTE

    for r in range(len(regions_list)):
        this_reg = regions_list[r]

        country_list = dict_regs_and_countries[this_reg]
        country_list.sort()

        # Add a filter to include countries with transport data only:
        country_list = [c for c in country_list if c in tr_list_app_countries_u]

        for c in range(len(country_list)):
            this_country = country_list[c]
            this_country_2 = dict_equiv_country_2[this_country]

            # At this point, we can add the IROTE estimation based on the previously stored dictionaries and lists
            '''
            IROTE_num = Renewable energy use + biomass
            IROTE_den = IROTE_num + Electricity fuel use + Non-electrical and non-biomass demand
            '''

            # First, we need to calculate the renewable energy use by reverse-engineering the renewable production
            use_ref_eb = dict_database['EB'][this_reg][this_country_2]
            use_ref_eb_transf = use_ref_eb['Total transformation']['Power plants']
            all_eb_transf_keys = list(use_ref_eb_transf.keys())

            '''
            freeze for developing production expansion normalzied share
            dictionary
            '''

            # Unpack useful dictionaries previously stored
            total_production = dict_local_reg[this_reg][this_country]['Electricity total production']
            total_ele_prod = [0] * len(time_vector)
            for atech in list(total_production.keys()):
                total_ele_prod = [a + b for a, b in zip(total_ele_prod, total_production[atech])]
            dict_energy_demand_by_fuel = dict_local_reg[this_reg][this_country]['Energy demand by fuel']
            fuel_use_electricity = dict_local_reg[this_reg][this_country]['Electricity fuel use']
            supply_dict = dict_database['EB'][this_reg][this_country_2]['Total supply']
            transf_dict = dict_database['EB'][this_reg][this_country_2]['Total transformation']
            dem_dict = dict_database['EB'][this_reg][this_country_2]['Final consumption']
            selfcon_dict = dict_database['EB'][this_reg][this_country_2]['Self-consumption']
            loss_dict = dict_database['EB'][this_reg][this_country_2]['Losses']

            # Calculate new exports
            export_dict_local = {}
            
            # Note: Deal with losses and self consumption that add to production + imports               
            
            # Find the relative change of consumption for specific fuel
            list_all_fuels = list(dict_energy_demand_by_fuel.keys())
            list_all_fuels = list_all_fuels[:list_all_fuels.index('Other secondary')+1]
            split_index = list_all_fuels.index('Electricity')
            list_primary_fuels = list_all_fuels[:split_index]
            list_secondary_fuels = list_all_fuels[split_index:]
            list_all_fuels_rev = list_secondary_fuels + list_primary_fuels

            # We need to find the fuel use electricy, but by fuel
            fuel_use_electricity_by_fuel = {}
            for atech in list(fuel_use_electricity.keys()):
                if type(fuel_use_electricity[atech]) is not str:
                    # for afuel in list_all_fuels:
                    for afuel in list(fuel_use_electricity[atech].keys()):
                        if afuel not in list(fuel_use_electricity_by_fuel.keys()):
                            fuel_use_electricity_by_fuel.update({afuel: fuel_use_electricity[atech][afuel]})
                        else:
                            fuel_use_electricity_by_fuel[afuel] = [a + b for a, b in zip(
                                fuel_use_electricity_by_fuel[afuel] + fuel_use_electricity[atech][afuel])]
                       
            # We need to have a preliminary assessment about transformations
            refinery_data = transf_dict['Refineries']
            refinery_ratio_and_sets = {'ratio':{}, 'sets':{}}
            for primfuel in list_primary_fuels:
                prim_val_by = refinery_data[primfuel][str(time_vector[0])]
                sec_val_by_sum = 0
                set_list = []
                for secfuel in list_secondary_fuels:
                    sec_val_by = refinery_data[secfuel][str(time_vector[0])]
                    if float(sec_val_by) > 0.0:
                        sec_val_by_sum += sec_val_by
                        set_list.append(secfuel)
                if float(prim_val_by) != 0 and sec_val_by_sum > 0:
                    refinery_ratio_and_sets['ratio'].update({primfuel: -1*float(prim_val_by)/sec_val_by_sum})
                    refinery_ratio_and_sets['sets'].update({primfuel: set_list})

            # We need to store the total supply with the general production/import approach:
            total_supply_dict, total_prod_dict = {}, {}

            # Apply the exports to secondary fuels first, then primary fuels:
            for lif in list_all_fuels_rev:               
                # Dealing with export projection by extracting the overall demand
                list_lif = dict_energy_demand_by_fuel_sum[lif]
                if list_lif[0] != 0:
                    # For the specific fuel, let's calculate exports and production
                    dict_energy_demand_by_fuel_sum_norm = [
                        v/list_lif[0] for v in list_lif]

                    export_fuel_by = supply_dict['Exports'][lif][str(time_vector[0])]
                    export_fuel_list = [-1 * v * export_fuel_by
                        for v in dict_energy_demand_by_fuel_sum_norm]
                
                    if sum(dict_energy_demand_by_fuel_sum_norm)/len(dict_energy_demand_by_fuel_sum_norm) == 1:
                        print('please review this case if it happens')
                        sys.exit()
                
                else:
                    export_fuel_list = [0] * len(time_vector)

                # Store value of exports
                export_dict_local.update({lif: deepcopy(export_fuel_list)})

                if lif in list(refinery_ratio_and_sets['ratio'].keys()):
                    iter_sec_sets = refinery_ratio_and_sets['sets'][lif]
                    sum_sec_set_list_exp = [0]*len(time_vector)
                    sum_sec_set_list_loc = [0]*len(time_vector)
                    for lif2 in iter_sec_sets:
                        sum_sec_set_list_exp = [a + b for a, b in zip(
                            export_dict_local[lif2], sum_sec_set_list_exp)]
                        sum_sec_set_list_loc = [a + b for a, b in zip(
                            dict_energy_demand_by_fuel[lif2], sum_sec_set_list_loc)] 

                    sum_sec_set_list = [a + b for a, b in zip(
                        sum_sec_set_list_exp, sum_sec_set_list_loc)]
                    
                    refinery_demand = [refinery_ratio_and_sets['ratio'][lif]*v
                        for v in sum_sec_set_list]
                else:
                    refinery_demand = [0]*len(time_vector)

                # Extracting the local demand:
                list_demand = dict_energy_demand_by_fuel[lif]

                # Extracting fuel demand for power plants:
                try:
                    fuel_demand = fuel_use_electricity_by_fuel[lif]
                except Exception:
                    fuel_demand = [0] * len(time_vector)

                # Dealing with the production structure:
                selfcon_by = selfcon_dict['none'][lif][str(time_vector[0])]
                loss_by = loss_dict['none'][lif][str(time_vector[0])]

                prod_fuel_by = supply_dict['Production'][lif][str(time_vector[0])]
                import_fuel_by = supply_dict['Imports'][lif][str(time_vector[0])]
                if prod_fuel_by + import_fuel_by == 0:
                    prod_fuel_ratio = 0
                    import_fuel_ratio = 0
                    loss_ratio = 0
                else:
                    prod_fuel_ratio = prod_fuel_by / (prod_fuel_by + import_fuel_by)
                    import_fuel_ratio = import_fuel_by / (prod_fuel_by + import_fuel_by)

                    # We have the losses as transformation + demand as a share,
                    # but prefer the losses to production + imports ratio
                    loss_ratio = (selfcon_by + loss_by) / (prod_fuel_by + import_fuel_by)

                '''
                We have adopted the following approach:
                total_supply = production + imports
                demand + refinery + fuel_powerplant + loss = total_supply
                loss / total_supply = loss_ratio
                demand + refinery + fuel_powerplant = subtotal_supply
                
                => subtotal_supply/total_supply + loss_ratio = 1
                => subtotal_supply/(1-loss_ratio) = total_supply
                '''

                subtotal_supply = [a + b + c for a, b, c in zip(
                    list_demand, refinery_demand, fuel_demand)]
                
                if loss_ratio < 1:
                    total_supply = [v/(1-loss_ratio) for v in subtotal_supply]
                else:
                    total_supply = deepcopy(subtotal_supply)
                total_prod = [v*prod_fuel_ratio for v in total_supply]
                total_imports = [v*import_fuel_ratio for v in total_supply]

                total_supply_dict.update({lif: deepcopy(total_supply)})
                total_prod_dict.update({lif: deepcopy(total_prod)})

                # print('Check this please!!!')
                # sys.exit()

            # print('Stopping for adding the base energy balance shares')
            # sys.exit()

            useful_pj_inputs_dict = {}
            for a_transf_key in all_eb_transf_keys:
                add_trasf_val = -1*float(use_ref_eb_transf[a_transf_key]['2021'])
                if add_trasf_val > 0:
                    if a_transf_key == 'Sugar cane and derivatives':
                        use_transf_key = dict_equiv_pp_fuel_rev['Renewable Thermal']
                        useful_pj_inputs_dict.update({use_transf_key:add_trasf_val})
                    else:
                        try:
                            use_transf_key = dict_equiv_pp_fuel_rev[a_transf_key]
                            useful_pj_inputs_dict.update({use_transf_key:add_trasf_val})
                        except Exception:
                            print('No equivalence found for', a_transf_key)
                            # pass

            primary_inv_effic_dict = {}
            renewable_key_list = []
            iter_tot_prod_keys = list(total_production.keys())
            store_res_energy_orig = dict_store_res_energy_orig[this_country_2]
            for itpk in iter_tot_prod_keys:
                # print('hey', itpk, renewable_key_list)
                # get_src_gen = total_production[itpk][0]
                get_src_gen = store_res_energy_orig[itpk][0]
                if itpk in list(useful_pj_inputs_dict.keys()):
                    get_src_prim_ene = useful_pj_inputs_dict[itpk]
                    if get_src_gen == 0 and get_src_prim_ene == 0:
                        primary_inv_effic_dict.update({itpk: 0})
                    elif get_src_gen == 0 and get_src_prim_ene != 0:
                        print('Anomaly', get_src_gen, round(get_src_prim_ene, 2))
                        # sys.exit()
                    else:
                        primary_inv_effic_dict.update({itpk: get_src_prim_ene/get_src_gen})
                if 'Solar' in itpk:
                    # primary_inv_effic_dict.update({itpk: 1/0.15})
                    primary_inv_effic_dict.update({itpk: 1/0.8})
                    renewable_key_list.append(itpk)
                if 'Wind' in itpk:
                    # primary_inv_effic_dict.update({itpk: 1/0.3})
                    primary_inv_effic_dict.update({itpk: 1/0.8})
                    renewable_key_list.append(itpk)
                if 'Geothermal' in itpk and 'PP_Geothermal' not in list(useful_pj_inputs_dict.keys()):
                    primary_inv_effic_dict.update({itpk: 1/((1601.7292 * 0.0036)/50.9971403743161)})
                    renewable_key_list.append(itpk)
                if 'Hydro' in itpk and 'PP_Hydro' not in list(useful_pj_inputs_dict.keys()):
                    primary_inv_effic_dict.update({itpk: 1})
                    renewable_key_list.append(itpk)
                elif 'Hydro' in itpk and primary_inv_effic_dict[itpk] < 1:
                    primary_inv_effic_dict.update({itpk: 1})
                    renewable_key_list.append(itpk)
                elif 'Hydro' in itpk:
                    renewable_key_list.append(itpk)
                    
                    # print('happens__')
                    # sys.exit()
                    
                #if 'Sugar cane' in itpk:
                #    print('please review this')
                #    sys.exit()

                if itpk in renewable_key_list:
                    # print('happens 1', itpk, primary_inv_effic_dict[itpk])
                    if primary_inv_effic_dict[itpk] < 1:
                        primary_inv_effic_dict[itpk] = 1
                        # print('happens 2', itpk)

            # Now we need to iteratively calculate an indicator that has the renewable information for all years:
            reno_gen_list = [0] * len(time_vector)
            for rlk in range(len(renewable_key_list)):
                this_total_production = total_production[renewable_key_list[rlk]]
                this_primary_use = [primary_inv_effic_dict[renewable_key_list[rlk]]*val for val in this_total_production]
                reno_gen_list = [a + b for a, b in zip(reno_gen_list, this_primary_use)]

            # We need to add biomass to the numerator, the one consumed on the demand side:
            other_biomass = [0] * len(time_vector)
            sug_cane_ene = dict_energy_demand_by_fuel['Sugar cane and derivatives']
            firewood_ene = dict_energy_demand_by_fuel['Firewood']
            otherprim_ene = dict_energy_demand_by_fuel['Other primary sources']
            
            other_biomass = [a + b + c for a, b, c in zip(sug_cane_ene, firewood_ene, otherprim_ene)]
            
            # Non-electricity and non-biomass demand:
            non_ele_bio_list = [0] * len(time_vector)
            for a_dedbf in list(dict_energy_demand_by_fuel.keys()):
                if a_dedbf not in ['Electricity', 'Sugar cane and derivatives', 'Firewood', 'Other primary sources']:
                    non_ele_bio_list = [a + b for a, b in zip(non_ele_bio_list, dict_energy_demand_by_fuel[a_dedbf])]
                
                    # print(a_dedbf, [round(x, 2) for x in dict_energy_demand_by_fuel[a_dedbf]])
                
            # Add fuel use by power plants:
            fuel_use_pp_list = [0] * len(time_vector)
            for a_fue in list(fuel_use_electricity.keys()):
                if type(fuel_use_electricity[a_fue]) != str:
                    for a_fue_2 in list(fuel_use_electricity[a_fue].keys()):
                        this_fue_list = fuel_use_electricity[a_fue][a_fue_2]
                        fuel_use_pp_list = [a + b for a, b in zip(fuel_use_pp_list, this_fue_list)]

            # Extract the sum of primary supply not considered by renewables (pick fuel names):
            nonreno_primary_fuels_pick = ['Oil', 'Natural Gas', 'Coal']
            nonreno_second_fuels_pick = ['Electricity', 'Liquified Gas', 'Gasoline and alcohol',
                 'Kerose and jet fuel', 'Diesel', 'Fuel Oil', 'Coke', 'Gases']

            total_supply_primary = [0] * len(time_vector)
            total_supply_secondary = [0] * len(time_vector)
            total_production_secondary = [0] * len(time_vector)
            for pfp in nonreno_primary_fuels_pick:
                total_supply_primary = [a + b for a, b in zip(
                    total_supply_primary, total_supply_dict[pfp])]

            for sfp in nonreno_second_fuels_pick:
                total_supply_secondary = [a + b for a, b in zip(
                    total_supply_secondary, total_supply_dict[sfp])]
                total_production_secondary = [a + b for a, b in zip(
                    total_production_secondary, total_prod_dict[sfp])]

            # Now calculate the irote index:
            num_irote = [a + b for a, b in zip(reno_gen_list, other_biomass)]
            den_irote = [a + b + c for a, b, c in zip(num_irote, non_ele_bio_list, fuel_use_pp_list)]
            irote = [100*a / b for a, b in zip(num_irote, den_irote)]


            # Store the irote results
            dict_local_reg[this_reg][this_country].update({'IROTE': deepcopy(irote)})
            dict_local_reg[this_reg][this_country].update({'IROTE_NUM': deepcopy(num_irote)})
            dict_local_reg[this_reg][this_country].update({'IROTE_DEN': deepcopy(den_irote)})

            # print('review if this finished - last part')
            # sys.exit()

    ###########################################################################
    # *Here it is crucial to implement the exports as share of total LAC demand:

    # ...calculate the total natural gas demand
    lac_ng_dem = [0 for y in range(len(time_vector))]
    keys_2list_regions = list(dict_local_reg.keys())
    country_2_region = {}
    for ng_reg in keys_2list_regions:
        keys_2list_countries = list(dict_local_reg[ng_reg].keys())
        for ng_cntry in keys_2list_countries:
            query_dict = dict_local_reg[ng_reg][ng_cntry]
            local_ng_dem = []
            for y in range(len(time_vector)):
                add_val = \
                    query_dict['Energy demand by fuel']['Natural Gas'][y] + \
                    query_dict['Electricity fuel use']['PP_Thermal_Natural Gas']['Natural Gas'][y]  # this is a list
                local_ng_dem.append(add_val)

            for y in range(len(time_vector)):
                lac_ng_dem[y] += deepcopy(local_ng_dem[y])

            # ...store the dictionary below to quickly store the export values
            country_2_region.update({ng_cntry:ng_reg})

    # ...extract the exporting countries to LAC // assume *df_scen* is correct from previous loops
    mask_exports = \
        (df_scen['Parameter'] == '% Exports for production')
    mask_exports_pipeline = \
        (df_scen['Parameter'] == '% Exports for production through pipeline')

    df_scen_exports = df_scen.loc[mask_exports]
    df_scen_exports_countries = \
        df_scen_exports['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_scen_exports_countries = \
        [c for c in df_scen_exports_countries if c in tr_list_app_countries_u]

    df_scen_exports_pipeline = df_scen.loc[mask_exports_pipeline]
    df_scen_exports_countries_pipeline = \
        df_scen_exports_pipeline['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_scen_exports_countries_pipeline = \
        [c for c in df_scen_exports_countries_pipeline if c in tr_list_app_countries_u]

    # ...now we must extract all natural gas prices:
    mask_ngas_prices = \
        ((df_scen['Parameter'].isin(['Fuel prices sales through pipeline',
                                     'Fuel prices sales liquified'])) &
         (df_scen['Fuel'] == 'Natural Gas'))
    df_ngas_prices = df_scen.loc[mask_ngas_prices]
    df_ngas_prices.reset_index(drop=True, inplace=True)

    # ...now we must extract the quantitiy of natural gas exports to LAC!
    # In a loop, iterate across countries:
    for this_con in df_scen_exports_countries:
        mask_con = (df_scen_exports['Application_Countries'] == this_con)
        df_scen_exports_select = df_scen_exports.loc[mask_con]
        df_scen_exports_select.reset_index(drop=True, inplace=True)

        mask_con_pipe = (df_scen_exports_pipeline['Application_Countries'] == this_con)
        df_scen_exports_select_pipe = df_scen_exports_pipeline.loc[mask_con_pipe]
        df_scen_exports_select_pipe.reset_index(drop=True, inplace=True)

        exports_country = [0 for y in range(len(time_vector))]  # in PJ
        exports_country_pipe = [0 for y in range(len(time_vector))]
        exports_country_liq = [0 for y in range(len(time_vector))]
        exports_income = [0 for y in range(len(time_vector))]

        for y in range(len(time_vector)):
            this_year = int(time_vector[y])

            disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))

            export_price_pipe = df_ngas_prices.loc[0, int(time_vector[0])]  # FOR NOW ASSUME THE PRICE IS CONSTANT
            export_price_liq = df_ngas_prices.loc[1, int(time_vector[0])]

            # ...here we must calculate the natural gas exports for the country:
            exports_country[y] = \
                lac_ng_dem[y]*df_scen_exports_select.loc[0, this_year]/100

            if len(df_scen_exports_select_pipe.index.tolist()) != 0:
                # here we need to discriminate pipeline and non-pipeline elements
                q_ngas_pipe = \
                    exports_country[y]*df_scen_exports_select_pipe.loc[0, this_year]/100
            else:
                q_ngas_pipe = 0

            exports_country_pipe[y] = q_ngas_pipe
            exports_country_liq[y] = exports_country[y] - q_ngas_pipe

            exports_income[y] = \
                exports_country_pipe[y]*export_price_pipe + \
                exports_country_liq[y]*export_price_liq
            exports_income[y] *= disc_constant

        # ...now we must store the result and intermediary info to the dictionary
        this_reg = country_2_region[this_con]
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports (PJ)':deepcopy(exports_country)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports via Pipeline (PJ)':deepcopy(exports_country_pipe)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports Liquified (PJ)':deepcopy(exports_country_liq)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports Income (M USD)':deepcopy(exports_income)})  # only print the disocunted value

    ###########################################################################
    # *For fugitive emissions, we must use the "imports" sheet, with a similar approach as above

    # ...extract fugitive emissions of natural gas (exclusively):
    mask_fugitive_emissions = \
        ((df4_ef['Apply'] == 'Production') &
         (df4_ef['Fuel'] == 'Natural Gas'))
    this_df_fugef_ngas = df4_ef.loc[mask_fugitive_emissions]
    this_df_fugef_ngas.reset_index(drop=True, inplace=True)
    fugef_ngas = this_df_fugef_ngas.iloc[0][per_first_yr]  # assume this is a constant

    mask_fugitive_emissions_2 = \
        ((df4_ef['Apply'] == 'Imports') &
         (df4_ef['Fuel'] == 'Natural Gas'))
    this_df_fugef_ngas_2 = df4_ef.loc[mask_fugitive_emissions_2]
    this_df_fugef_ngas_2.reset_index(drop=True, inplace=True)
    fugef_ngas_2 = this_df_fugef_ngas_2.iloc[0][per_first_yr]  # assume this is a constant

    # ...extract the dataframe with imports information:
    mask_ngas_imports = \
        (df_scen['Parameter'] == '% Imports for consumption')
    df_ngas_imports = df_scen.loc[mask_ngas_imports]
    df_ngas_imports.reset_index(drop=True, inplace=True)
    df_ngas_imports_countries = \
        df_ngas_imports['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_ngas_imports_countries = \
        [c for c in df_ngas_imports_countries if c in tr_list_app_countries_u]

    # ...iterate across country-wide consumption and find the imports, local production, and add the exports from above
    for acon in range(len(df_ngas_imports_countries)):
        this_con = df_ngas_imports_countries[acon]
        this_reg = country_2_region[this_con]

        query_dict = dict_local_reg[this_reg][this_con]
        local_ng_dem = []
        for y in range(len(time_vector)):
            add_val = \
                query_dict['Energy demand by fuel']['Natural Gas'][y] + \
                query_dict['Electricity fuel use']['PP_Thermal_Natural Gas']['Natural Gas'][y]  # this is a list
            local_ng_dem.append(add_val)

        try:
            local_ng_exp = \
                query_dict['Natural Gas Exports (PJ)']
        except Exception:
            local_ng_exp = [0 for y in range(len(time_vector))]

        local_ng_production = []
        local_ng_fugitive_emissions = []

        imps_ng = []
        imps_ng_fugitive_emissions = []

        for y in range(len(time_vector)):
            this_year = int(time_vector[y])

            imports_share = df_ngas_imports.loc[acon, this_year]
            imports_PJ = local_ng_dem[y]*imports_share/100
            local_prod_PJ = local_ng_dem[y] - imports_PJ
            local_prod_PJ += local_ng_exp[y]

            local_ng_production.append(local_prod_PJ)
            local_ng_fugitive_emissions.append(local_prod_PJ*fugef_ngas)

            imps_ng.append(imports_PJ)
            imps_ng_fugitive_emissions.append(imports_PJ*fugef_ngas_2)

        dict_local_reg[this_reg][this_con].update({'Natural Gas Production (PJ)':deepcopy(local_ng_production)})  # aggregate
        dict_local_reg[this_reg][this_con].update({'Natural Gas Production Fugitive Emissions (MTon)':deepcopy(local_ng_fugitive_emissions)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Imports (PJ)':deepcopy(imps_ng)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Imports Fugitive Emissions (MTon)':deepcopy(imps_ng_fugitive_emissions)})

    ###########################################################################
    # *For job estimates, we must multiply times the installed capacity:
    # *For T&D estimates, we must check the electricity supply:

    # ...iterate across all countries and estimate the jobs
    for ng_reg in keys_2list_regions:
        keys_2list_countries = list(dict_local_reg[ng_reg].keys())
        this_reg = ng_reg
        for ng_cntry in keys_2list_countries:
            this_con = ng_cntry

            # ...now we must iterate across technologies with the technological capacity
            list_electric_sets_3.sort()  # this must work
            tech_counter = 0
            for tech in list_electric_sets_3:
                # -------------------------------------------------------------
                # >>> this section is for JOBS:
                try:
                    list_cap = \
                        dict_local_reg[this_reg][this_con]['Electricity total capacity'][tech]
                except Exception:
                    list_cap = [0 for y in range(len(time_vector))]

                try:
                    list_new_cap = \
                        dict_local_reg[this_reg][this_con]['Electricity new capacity'][tech]
                except Exception:
                    list_new_cap = [0 for y in range(len(time_vector))]

                try:
                    list_new_prod = \
                        dict_local_reg[this_reg][this_con]['Electricity new production per tech'][tech]
                except Exception:
                    list_new_prod = [0 for y in range(len(time_vector))]

                try:
                    list_demand_2_supply = \
                        dict_local_reg[this_reg][this_con]['Electricity demand to supply'][tech]
                except Exception:
                    list_demand_2_supply = [0 for y in range(len(time_vector))]

                # ...we must also extract the jobs per unit of installed capacity
                mask_df4_job_fac = \
                    (df4_job_fac['Tech'] == tech)
                this_df4_job_fac = \
                    df4_job_fac.loc[mask_df4_job_fac]

                if len(this_df4_job_fac.index.tolist()) != 0:
                    jobs_factor_constru = this_df4_job_fac['Construction/installation (Job years/ MW)'].iloc[0]
                    jobs_factor_manufac = this_df4_job_fac['Manufacturing (Job years/ MW)'].iloc[0]
                    jobs_factor_opeyman = this_df4_job_fac['Operations & maintenance (Jobs/MW)'].iloc[0]
                    jobs_factor_decom = this_df4_job_fac['Decommissioning (Jobs/MW)'].iloc[0]
                else:
                    jobs_factor_constru = 0
                    jobs_factor_manufac = 0
                    jobs_factor_opeyman = 0
                    jobs_factor_decom = 0

                # ...we must create a LAC multiplier (based on the paper
                # https://link.springer.com/content/pdf/10.1007%2F978-3-030-05843-2_10.pdf)
                jobs_LAC_mult_vector_raw = ['' for x in range(len(time_vector))]
                for x in range(len(time_vector)):
                    if int(time_vector[x]) <= 2030:
                        jobs_LAC_mult_vector_raw[x] = 3.4
                    elif int(time_vector[x]) == 2040:
                        jobs_LAC_mult_vector_raw[x] = 3.1
                    elif int(time_vector[x]) == 2050:
                        jobs_LAC_mult_vector_raw[x] = 2.9
                    else:
                        pass
                jobs_LAC_mult_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr,
                                         jobs_LAC_mult_vector_raw, 'ini',
                                         this_scen, '')

                # ...we must estimate the jobs
                jobs_con_list_per_tech = \
                    [jobs_factor_constru*(1000*list_new_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_man_list_per_tech = \
                    [jobs_factor_manufac*(1000*list_new_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_oym_list_per_tech = \
                    [jobs_factor_opeyman*(1000*list_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_dec_list_per_tech = \
                    [jobs_factor_decom*(1000*list_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]

                # ...and store the results
                if tech_counter == 0:
                    dict_local_reg[this_reg][this_con].update({'Related construction jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related manufacturing jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related O&M jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related decommissioning jobs':{}})
                dict_local_reg[this_reg][this_con]['Related construction jobs'].update({tech: jobs_con_list_per_tech})  # per tech
                dict_local_reg[this_reg][this_con]['Related manufacturing jobs'].update({tech: jobs_man_list_per_tech})
                dict_local_reg[this_reg][this_con]['Related O&M jobs'].update({tech: jobs_oym_list_per_tech})
                dict_local_reg[this_reg][this_con]['Related decommissioning jobs'].update({tech: jobs_dec_list_per_tech})

                # -------------------------------------------------------------
                # >>> this section is for T&D:
                try:
                    list_generation = \
                        dict_local_reg[this_reg][this_con]['Electricity total production'][tech]
                except Exception:
                    list_generation = [0 for y in range(len(time_vector))]

                # ...we must also extract the costs per unit of generated electricity
                mask_df4_tran_dist_fac = \
                    (df4_tran_dist_fac['Tech'] == tech)
                this_df4_tran_dist_fac = \
                    df4_tran_dist_fac.loc[mask_df4_tran_dist_fac]

                if len(this_df4_tran_dist_fac.index.tolist()) != 0:
                    transmi_capex = this_df4_tran_dist_fac['Transmission Capital Cost (M US$/PJ produced)'].iloc[0]
                    transmi_fopex = this_df4_tran_dist_fac['Transmission 2% Fixed Cost (M US$/PJ produced)'].iloc[0]
                    distri_capex = this_df4_tran_dist_fac['Distribution Capital Cost (M US$/PJ produced)'].iloc[0]
                    distri_fopex = this_df4_tran_dist_fac['Distribution 2% Fixed Cost (M US$/PJ produced)'].iloc[0]
                else:
                    transmi_capex = 0
                    transmi_fopex = 0
                    distri_capex = 0
                    distri_fopex = 0

                # ...we must estimate the t&d costs
                transmi_capex_list_per_tech = \
                    [transmi_capex*(list_new_prod[y]) for y in range(len(time_vector))]
                transmi_fopex_list_per_tech = \
                    [transmi_fopex*(list_generation[y]) for y in range(len(time_vector))]
                distri_capex_list_per_tech = \
                    [distri_capex*(list_new_prod[y]) for y in range(len(time_vector))]
                distri_fopex_list_per_tech = \
                    [distri_fopex*(list_generation[y]) for y in range(len(time_vector))]

                transmi_capex_list_per_tech_disc = deepcopy(transmi_capex_list_per_tech)
                transmi_fopex_list_per_tech_disc = deepcopy(transmi_fopex_list_per_tech)
                distri_capex_list_per_tech_disc = deepcopy(distri_capex_list_per_tech)
                distri_fopex_list_per_tech_disc = deepcopy(distri_fopex_list_per_tech)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    transmi_capex_list_per_tech_disc[y] *= disc_constant
                    transmi_fopex_list_per_tech_disc[y] *= disc_constant
                    distri_capex_list_per_tech_disc[y] *= disc_constant
                    distri_fopex_list_per_tech_disc[y] *= disc_constant

                # ...and store the results
                if tech_counter == 0:
                    dict_local_reg[this_reg][this_con].update({'Transmission CAPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Transmission Fixed OPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution CAPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution Fixed OPEX':{}})

                    dict_local_reg[this_reg][this_con].update({'Transmission CAPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Transmission Fixed OPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution CAPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution Fixed OPEX (disc)':{}})

                dict_local_reg[this_reg][this_con]['Transmission CAPEX'].update({tech: transmi_capex_list_per_tech})  # per tech
                dict_local_reg[this_reg][this_con]['Transmission Fixed OPEX'].update({tech: transmi_fopex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Distribution CAPEX'].update({tech: distri_capex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Distribution Fixed OPEX'].update({tech: distri_fopex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Transmission CAPEX (disc)'].update({tech: transmi_capex_list_per_tech_disc})  # per tech
                dict_local_reg[this_reg][this_con]['Transmission Fixed OPEX (disc)'].update({tech: transmi_fopex_list_per_tech_disc})
                dict_local_reg[this_reg][this_con]['Distribution CAPEX (disc)'].update({tech: distri_capex_list_per_tech_disc})
                dict_local_reg[this_reg][this_con]['Distribution Fixed OPEX (disc)'].update({tech: distri_fopex_list_per_tech_disc})

                ###############################################################
                # ...increase the tech count:
                tech_counter += 1

    ###########################################################################
    # Store the elements
    dict_scen.update({this_scen: deepcopy(dict_local_reg)})

# print('End the simulation here. The printing is done below.')
# sys.exit()

# ...our scenarios have run here
# 4) Now we can print the results file

# ...but first, review everything we have:
# Enlist the names of your output and the contained keys:
'''
OUTPUT:
'Global warming externalities by demand': tech (demand) / fuel / list with values as long as year
'Local pollution externalities by demand': tech (demand) / fuel / list with values as long as year
'Emissions by demand': tech (demand) / fuel / list with values as long as year
'Electricity fuel use': tech / fuel / list with values as long as year
'Global warming externalities in electricity': tech / fuel / list with values as long as year
'Local pollution externalities in electricity': tech / fuel / list with values as long as year
'Emissions in electricity': tech / fuel / list with values as long as year
'Electricity total capacity': tech / list with values as long as year
'Electricity residual capacity': tech / list with values as long as year
'Electricity new capacity': tech / list with values as long as year
'Electricity total production': tech / list with values as long as year || implicitly, electricity
'Electricity CAPEX': tech / list with values as long as year
'Electricity Fixed OPEX': tech / list with values as long as year
'Electricity Variable OPEX': tech / list with values as long as year
'Electricity Grid Connection Cost': tech / list with values as long as year
'''


# Enlist the names of your input and the contained keys:
'''
PARAMETERS:
this_externality_dict[fuel]['Global warming']  # constant
this_externality_dict[fuel]['Local pollution']  # constant
emissions_fuels_dict[fuel]  # list with values as long as year
idict_u_capex[tech]  # list with values as long as year
idict_u_fopex[tech]  # list with values as long as year
idict_u_vopex[tech]  # list with values as long as year
idict_u_gcc[tech]  # list with values as long as year
idict_cau[tech]  # list with values as long as year
idict_net_cap_factor[tech]  # list with values as long as year
idict_hr[tech]  # list with values as long as year
idict_oplife[tech]  # list with values as long as year
'''

### THIS IS ALL THE DATA THAT WE HAVE AVAILABLE:
# ...now, iterate and create a list of things you want to store:
list_dimensions = ['Strategy', 'Region', 'Country', 'Technology', 'Technology type', 'Fuel', 'Year']
list_inputs = ['Unit ext. global warming', #1
               'Unit ext. local pollution', #2
               'Emission factor', #3
               'Unit CAPEX', #4
               'Unit fixed OPEX', #5
               'Unit variable OPEX', #6
               'Unit grid connection cost', #7
               'Net capacity factor', #8
               'Heat rate', #9
               'Operational life'] #10
'''
list_outputs = ['Global warming externalities by demand', #1
                'Local pollution externalities by demand', #2
                'Emissions by demand', #3
                'Electricity fuel use', #4
                'Global warming externalities in electricity', #5
                'Local pollution externalities in electricity', #6
                'Emissions in electricity', #7
                'Electricity total capacity', #8
                'Electricity new capacity', #9
                'Electricity total production', #10
                'Electricity CAPEX', #11
                'Electricity Fixed OPEX', #12
                'Electricity Variable OPEX', #13
                'Electricity Grid Connection Cost'] #14
'''
list_outputs = ['Electricity CAPEX', #1
                'Electricity Fixed OPEX', #2
                'Electricity fuel use', #3
                'Electricity Grid Connection Cost', #4
                'Electricity new capacity', #5
                'Electricity residual capacity', #6
                'Electricity total capacity', #7
                'Electricity total production', #8
                'Electricity Variable OPEX', #9
                'Emissions by demand', #10
                'Emissions in electricity', #11
                'Energy demand by sector', #12
                'Energy demand by fuel', #13
                'Energy intensity by sector', #14
                'Global warming externalities by demand', #15
                'Global warming externalities in electricity', #16
                'Local pollution externalities by demand', #17
                'Local pollution externalities in electricity', #18
                'Electricity CAPEX (disc)', # 19 // discounted costs
                'Electricity Fixed OPEX (disc)', # 20
                'Electricity Variable OPEX (disc)', # 21
                'Electricity Grid Connection Cost (disc)', # 22
                'Global warming externalities in electricity (disc)', # 23
                'Local pollution externalities in electricity (disc)', # 24
                'Natural Gas Exports (PJ)' , # 25 // natural gas exports
                'Natural Gas Exports via Pipeline (PJ)' , # 26
                'Natural Gas Exports Liquified (PJ)' , # 27
                'Natural Gas Exports Income (M USD)' , # 28
                'Natural Gas Production (PJ)' , # 29 // production and emission factors
                'Natural Gas Production Fugitive Emissions (MTon)' , # 30
                'Natural Gas Imports (PJ)' , # 31 // production and emission factors
                'Natural Gas Imports Fugitive Emissions (MTon)' , # 32
                'Related construction jobs' , # 33
                'Related manufacturing jobs' , # 34
                'Related O&M jobs' , # 35
                'Related decommissioning jobs' , # 36
                'Transmission CAPEX' , # 37
                'Transmission Fixed OPEX' , # 38
                'Distribution CAPEX' , # 39
                'Distribution Fixed OPEX' , # 40
                'Transmission CAPEX (disc)' , # 41
                'Transmission Fixed OPEX (disc)' , # 42
                'Distribution CAPEX (disc)' , # 43
                'Distribution Fixed OPEX (disc)', # 44
                'Electricity new capacity unplanned', # 45
                'Electricity new capacity planned', # 46
                'Electricity phase out capacity', # 47
                'Electricity production share (unplanned)', # 48
                'New energy assign', # 49
                'Accumulated new capacity', # 50
                'Electricity demand to supply', # 51
                'Electricity planned supply', # 52
                'Accumulated forced new capacity', # 53
                'Electricity planned supply per technology', # 54
                'Electricity residual supply', # 55
                'Electricity residual supply per tech', # 56
                'Fleet',  # 57
                'New Fleet',  # 58
                'Transport CAPEX [$]',  # 59
                'Transport Fixed OPEX [$]',  # 60
                'Transport Variable OPEX [$]',  # 61
                'Transport Tax Imports [$]',  # 62
                'Transport Tax IMESI_Venta [$]',  # 63
                'Transport Tax IVA_Venta [$]',  # 64
                'Transport Tax Patente [$]',  # 65
                'Transport Tax IMESI_Combust [$]',  # 66
                'Transport Tax IVA_Gasoil [$]',  # 67
                'Transport Tax IVA_Elec [$]',  # 68
                'Transport Tax IC [$]',  # 69
                'Transport Tax Otros_Gasoil [$]',  # 70
                'Transport Tax Tasa_Consular [$]',  # 71
                'IROTE',  # 72
                'IROTE_NUM',  # 73
                'IROTE_DEN',  # 74
                'Energy demand cost',  # 75
                'Energy demand cost (disc)'  # 76
                ]

list_inputs_add = [i + ' (input)' for i in list_inputs]
list_outputs_add = [i + ' (output)' for i in list_outputs]

h_strategy, h_region, h_country, h_tech, h_techtype, h_fuel, h_yr = \
    [], [], [], [], [], [], []
h_i1, h_i2, h_i3, h_i4, h_i5, h_i6, h_i7, h_i8, h_i9, h_i10 = \
    [], [], [], [], [], [], [], [], [], []
h_o1, h_o2, h_o3, h_o4, h_o5, h_o6, h_o7, h_o8, h_o9, h_o10 = \
    [], [], [], [], [], [], [], [], [], []
h_o11, h_o12, h_o13, h_o14, h_o15, h_o16, h_o17, h_o18, h_o19, h_o20 = \
    [], [], [], [], [], [], [], [], [], []
h_o21, h_o22, h_o23, h_o24, h_o25, h_o26, h_o27, h_o28, h_o29, h_o30 = \
    [], [], [], [], [], [], [], [], [], []
h_o31, h_o32, h_o33, h_o34, h_o35, h_o36, h_o37, h_o38, h_o39, h_o40 = \
    [], [], [], [], [], [], [], [], [], []
h_o41, h_o42, h_o43, h_o44, h_o45, h_o46, h_o47, h_o48, h_o49, h_o50 = \
    [], [], [], [], [], [], [], [], [], []
h_o51, h_o52, h_o53, h_o54, h_o55, h_o56, h_o57, h_o58, h_o59, h_o60 = \
    [], [], [], [], [], [], [], [], [], []
h_o61, h_o62, h_o63, h_o64, h_o65, h_o66, h_o67, h_o68, h_o69, h_o70 = \
    [], [], [], [], [], [], [], [], [], []
h_o71, h_o72, h_o73, h_o74, h_o75, h_o76 = \
    [], [], [], [], [], []
# ...here, clean up the fuels:
list_fuel_clean = [i for i in list_fuel_ALL if 'Total' not in i]
if overwrite_transport_model:
    list_fuel_clean += list(dict_eq_transport_fuels.keys())
list_fuel_clean += ['']

# print('End the simulation here. The printing is done below. 2')
# sys.exit()

print('\n')
print('PROCESS 2 - PRINTING THE INPUTS AND OUTPUTS')
for s in range(len(scenario_list)):
    this_scen = scenario_list[s]

    # regions_list = list(dict_regs_and_countries.keys())
    # regions_list = ['4_The Amazon', '5_Southern Cone']
    # regions_list = ['1_Mexico', '2_Central America', '4_The Amazon', '5_Southern Cone']
    regions_list = ['1_Mexico', '2_Central America', '3_Caribbean', '4_The Amazon', '5_Southern Cone']

    for r in range(len(regions_list)):
        this_reg = regions_list[r]

        country_list = dict_regs_and_countries[this_reg]
        country_list.sort()

        # Add a filter to include countries with transport data only:
        country_list = [c for c in country_list if c in tr_list_app_countries_u]

        for c in range(len(country_list)):
            this_country = country_list[c]

            # First iterable: list_demand_sector_techs
            # Second iterable: list_electric_sets_3
            # inner iterable 1: list_fuel_clean
            # inner iterable 2: time_vector
            tech_iterable = list_demand_sector_techs + \
                list_electric_sets_3 + types_all + ['']
            for tech in tech_iterable:
                for fuel in list_fuel_clean:
                    for y in range(len(time_vector)):
                        count_empties = 0

                        #if y == 0 and (tech == '' or fuel == ''):
                        #    print(this_scen, this_reg, tech, fuel)

                        if tech in list_demand_sector_techs:
                            tech_type = 'demand'
                        elif tech in list_electric_sets_3:
                            tech_type = 'power_plant'
                        elif tech in types_all:
                            tech_type = 'transport'
                        elif tech == '':
                            tech_type = ''
                        else:
                            tech_type = 'none'

                        this_data_dict = \
                            dict_scen[this_scen][this_reg][this_country]

                        # Store inputs:
                        # Input 1:
                        if fuel != '' and tech == '':
                            try:
                                h_i1.append(ext_by_country[this_country][fuel]['Global warming'])
                            except Exception:
                                h_i1.append(0)
                                count_empties += 1
                        else:
                            h_i1.append(0)
                            count_empties += 1

                        # Input 2:
                        if fuel != '' and tech == '':
                            try:
                                h_i2.append(ext_by_country[this_country][fuel]['Local pollution'])
                            except Exception:
                                h_i2.append(0)
                                count_empties += 1
                        else:
                            h_i2.append(0)
                            count_empties += 1

                        # Input 3:
                        if fuel != '' and tech == '':
                            try:
                                h_i3.append(emissions_fuels_dict[fuel][y])
                            except Exception:
                                h_i3.append(0)
                                count_empties += 1
                        else:
                            h_i3.append(0)
                            count_empties += 1

                        # Input 4:
                        if fuel == '' and tech != '':
                            try:
                                h_i4.append(idict_u_capex[tech][y])
                            except Exception:
                                h_i4.append(0)
                                count_empties += 1
                        else:
                            h_i4.append(0)
                            count_empties += 1

                        # Input 5:
                        if fuel == '' and tech != '':
                            try:
                                h_i5.append(idict_u_fopex[tech][y])
                            except Exception:
                                h_i5.append(0)
                                count_empties += 1
                        else:
                            h_i5.append(0)
                            count_empties += 1

                        # Input 6:
                        if fuel == '' and tech != '':
                            try:
                                h_i6.append(idict_u_vopex[tech][y])
                            except Exception:
                                h_i6.append(0)
                                count_empties += 1
                        else:
                            h_i6.append(0)
                            count_empties += 1

                        # Input 7:
                        if fuel == '' and tech != '':
                            try:
                                h_i7.append(idict_u_gcc[tech][y])
                            except Exception:
                                h_i7.append(0)
                                count_empties += 1
                        else:
                            h_i7.append(0)
                            count_empties += 1

                        # Input 8:
                        if fuel == '' and tech != '':
                            try:
                                h_i8.append(idict_net_cap_factor_by_scen_by_country[this_scen][this_country][tech][y])
                            except Exception:
                                h_i8.append(0)
                                count_empties += 1
                        else:
                            h_i8.append(0)
                            count_empties += 1

                        # Input 9:
                        if fuel == '' and tech != '':
                            try:
                                h_i9.append(idict_hr[tech][y])
                            except Exception:
                                h_i9.append(0)
                                count_empties += 1
                        else:
                            h_i9.append(0)
                            count_empties += 1

                        # Input 10:
                        if fuel == '' and tech != '':
                            try:
                                h_i10.append(idict_oplife[tech][y])
                            except Exception:
                                h_i10.append(0)
                                count_empties += 1
                        else:
                            h_i10.append(0)
                            count_empties += 1

                        # Store outputs:
                        # Output ID: 1
                        if fuel == '' and tech != '':
                            try:
                                h_o1.append(this_data_dict[list_outputs[0]][tech][y])
                            except Exception:
                                h_o1.append(0)
                                count_empties += 1
                        else:
                            h_o1.append(0)
                            count_empties += 1

                        # Output ID: 2
                        if fuel == '' and tech != '':
                            try:
                                h_o2.append(this_data_dict[list_outputs[1]][tech][y])
                            except Exception:
                                h_o2.append(0)
                                count_empties += 1
                        else:
                            h_o2.append(0)
                            count_empties += 1

                        # Output ID: 3
                        if fuel != '' and tech != '':
                            
                            # if fuel == 'Oil' and tech == 'PP_Thermal_Crude' and this_country == 'Chile':
                            #     print('review this')
                            #     sys.exit()
                            try:
                                h_o3.append(this_data_dict[list_outputs[2]][tech][fuel][y])
                            except Exception:
                                h_o3.append(0)
                                count_empties += 1
                        else:
                            h_o3.append(0)
                            count_empties += 1

                        # Output ID: 4
                        if fuel == '' and tech != '':
                            try:
                                h_o4.append(this_data_dict[list_outputs[3]][tech][y])
                            except Exception:
                                h_o4.append(0)
                                count_empties += 1
                        else:
                            h_o4.append(0)
                            count_empties += 1

                        # Output ID: 5
                        if fuel == '' and tech != '':
                            try:
                                h_o5.append(this_data_dict[list_outputs[4]][tech][y])
                            except Exception:
                                h_o5.append(0)
                                count_empties += 1
                        else:
                            h_o5.append(0)
                            count_empties += 1

                        # Output ID: 6
                        if fuel == '' and tech != '':
                            try:
                                h_o6.append(this_data_dict[list_outputs[5]][tech][y])
                            except Exception:
                                h_o6.append(0)
                                count_empties += 1
                        else:
                            h_o6.append(0)
                            count_empties += 1

                        # Output ID: 7
                        if fuel == '' and tech != '':
                            try:
                                h_o7.append(this_data_dict[list_outputs[6]][tech][y])
                            except Exception:
                                h_o7.append(0)
                                count_empties += 1
                        else:
                            h_o7.append(0)
                            count_empties += 1

                        # Output ID: 8
                        if fuel == '' and tech != '':
                            try:
                                h_o8.append(this_data_dict[list_outputs[7]][tech][y])
                            except Exception:
                                h_o8.append(0)
                                count_empties += 1
                        else:
                            h_o8.append(0)
                            count_empties += 1

                        # Output ID: 9
                        if fuel == '' and tech != '':
                            try:
                                h_o9.append(this_data_dict[list_outputs[8]][tech][y])
                            except Exception:
                                h_o9.append(0)
                                count_empties += 1
                        else:
                            h_o9.append(0)
                            count_empties += 1

                        # Output ID: 10
                        if fuel != '' and tech != '':
                            try:
                                h_o10.append(this_data_dict[list_outputs[9]][tech][fuel][y])
                            except Exception:
                                h_o10.append(0)
                                count_empties += 1
                        else:
                            h_o10.append(0)
                            count_empties += 1

                        # Output ID: 11
                        if fuel != '' and tech != '':
                            #if fuel == 'Oil' and tech == 'PP_Thermal_Crude' and this_country == 'Chile':
                            #    print('review this 2')
                            #    sys.exit()
                            try:
                                h_o11.append(this_data_dict[list_outputs[10]][tech][fuel][y])
                            except Exception:
                                h_o11.append(0)
                                count_empties += 1
                        else:
                            h_o11.append(0)
                            count_empties += 1

                        # Output ID: 12
                        if fuel != '' and tech != '':  # check energy demand
                            try:
                                h_o12.append(this_data_dict[list_outputs[11]][tech][fuel][y])
                            except Exception:
                                h_o12.append(0)
                                count_empties += 1
                        else:
                            h_o12.append(0)
                            count_empties += 1

                        # Output ID: 13
                        if fuel != '' and tech == '':
                            try:
                                h_o13.append(this_data_dict[list_outputs[12]][fuel][y])
                            except Exception:
                                h_o13.append(0)
                                count_empties += 1
                        else:
                            h_o13.append(0)
                            count_empties += 1

                        # Output ID: 14
                        if fuel == '' and tech != '':
                            try:
                                h_o14.append(this_data_dict[list_outputs[13]][tech]['Total'][y])
                            except Exception:
                                h_o14.append(0)
                                count_empties += 1
                        else:
                            h_o14.append(0)
                            count_empties += 1

                        # Output ID: 15
                        if fuel != '' and tech != '':
                            try:
                                h_o15.append(this_data_dict[list_outputs[14]][tech][fuel][y])
                            except Exception:
                                h_o15.append(0)
                                count_empties += 1
                        else:
                            h_o15.append(0)
                            count_empties += 1

                        # Output ID: 16
                        if fuel != '' and tech != '':
                            try:
                                h_o16.append(this_data_dict[list_outputs[15]][tech][fuel][y])
                            except Exception:
                                h_o16.append(0)
                                count_empties += 1
                        else:
                            h_o16.append(0)
                            count_empties += 1

                        # Output ID: 17
                        if fuel != '' and tech != '':
                            try:
                                h_o17.append(this_data_dict[list_outputs[16]][tech][fuel][y])
                            except Exception:
                                h_o17.append(0)
                                count_empties += 1
                        else:
                            h_o17.append(0)
                            count_empties += 1

                        # Output ID: 18
                        if fuel != '' and tech != '':
                            try:
                                h_o18.append(this_data_dict[list_outputs[17]][tech][fuel][y])
                            except Exception:
                                h_o18.append(0)
                                count_empties += 1
                        else:
                            h_o18.append(0)
                            count_empties += 1

                        # Output ID: 19***
                        if fuel == '' and tech != '':
                            try:
                                h_o19.append(this_data_dict[list_outputs[18]][tech][y])
                            except Exception:
                                h_o19.append(0)
                                count_empties += 1
                        else:
                            h_o19.append(0)
                            count_empties += 1

                        # Output ID: 20***
                        if fuel == '' and tech != '':
                            try:
                                h_o20.append(this_data_dict[list_outputs[19]][tech][y])
                            except Exception:
                                h_o20.append(0)
                                count_empties += 1
                        else:
                            h_o20.append(0)
                            count_empties += 1

                        # Output ID: 21***
                        if fuel == '' and tech != '':
                            try:
                                h_o21.append(this_data_dict[list_outputs[20]][tech][y])
                            except Exception:
                                h_o21.append(0)
                                count_empties += 1
                        else:
                            h_o21.append(0)
                            count_empties += 1

                        # Output ID: 22***
                        if fuel == '' and tech != '':
                            try:
                                h_o22.append(this_data_dict[list_outputs[21]][tech][y])
                            except Exception:
                                h_o22.append(0)
                                count_empties += 1
                        else:
                            h_o22.append(0)
                            count_empties += 1

                        # Output ID: 23***
                        if fuel != '' and tech != '':
                            try:
                                h_o23.append(this_data_dict[list_outputs[22]][tech][fuel][y])
                            except Exception:
                                h_o23.append(0)
                                count_empties += 1
                        else:
                            h_o23.append(0)
                            count_empties += 1

                        # Output ID: 24***
                        if fuel != '' and tech != '':
                            try:
                                h_o24.append(this_data_dict[list_outputs[23]][tech][fuel][y])
                            except Exception:
                                h_o24.append(0)
                                count_empties += 1
                        else:
                            h_o24.append(0)
                            count_empties += 1

                        # Output ID: 25***
                        if fuel == '' and tech == '':
                            try:
                                h_o25.append(this_data_dict[list_outputs[24]][y])
                            except Exception:
                                h_o25.append(0)
                                count_empties += 1
                        else:
                            h_o25.append(0)
                            count_empties += 1

                        # Output ID: 26***
                        if fuel == '' and tech == '':
                            try:
                                h_o26.append(this_data_dict[list_outputs[25]][y])
                            except Exception:
                                h_o26.append(0)
                                count_empties += 1
                        else:
                            h_o26.append(0)
                            count_empties += 1

                        # Output ID: 27***
                        if fuel == '' and tech == '':
                            try:
                                h_o27.append(this_data_dict[list_outputs[26]][y])
                            except Exception:
                                h_o27.append(0)
                                count_empties += 1
                        else:
                            h_o27.append(0)
                            count_empties += 1

                        # Output ID: 28***
                        if fuel == '' and tech == '':
                            try:
                                h_o28.append(this_data_dict[list_outputs[27]][y])
                            except Exception:
                                h_o28.append(0)
                                count_empties += 1
                        else:
                            h_o28.append(0)
                            count_empties += 1

                        # Output ID: 29***
                        if fuel == '' and tech == '':
                            try:
                                h_o29.append(this_data_dict[list_outputs[28]][y])
                            except Exception:
                                h_o29.append(0)
                                count_empties += 1
                        else:
                            h_o29.append(0)
                            count_empties += 1

                        # Output ID: 30***
                        if fuel == '' and tech == '':
                            try:
                                h_o30.append(this_data_dict[list_outputs[29]][y])
                            except Exception:
                                h_o30.append(0)
                                count_empties += 1
                        else:
                            h_o30.append(0)
                            count_empties += 1

                        # Output ID: 31***
                        if fuel == '' and tech == '':
                            try:
                                h_o31.append(this_data_dict[list_outputs[30]][y])
                            except Exception:
                                h_o31.append(0)
                                count_empties += 1
                        else:
                            h_o31.append(0)
                            count_empties += 1

                        # Output ID: 32***
                        if fuel == '' and tech == '':
                            try:
                                h_o32.append(this_data_dict[list_outputs[31]][y])
                            except Exception:
                                h_o32.append(0)
                                count_empties += 1
                        else:
                            h_o32.append(0)
                            count_empties += 1

                        # Output ID: 33***
                        if fuel == '' and tech != '':
                            try:
                                h_o33.append(this_data_dict[list_outputs[32]][tech][y])
                            except Exception:
                                h_o33.append(0)
                                count_empties += 1
                        else:
                            h_o33.append(0)
                            count_empties += 1

                        # Output ID: 34***
                        if fuel == '' and tech != '':
                            try:
                                h_o34.append(this_data_dict[list_outputs[33]][tech][y])
                            except Exception:
                                h_o34.append(0)
                                count_empties += 1
                        else:
                            h_o34.append(0)
                            count_empties += 1

                        # Output ID: 35***
                        if fuel == '' and tech != '':
                            try:
                                h_o35.append(this_data_dict[list_outputs[34]][tech][y])
                            except Exception:
                                h_o35.append(0)
                                count_empties += 1
                        else:
                            h_o35.append(0)
                            count_empties += 1

                        # Output ID: 36***
                        if fuel == '' and tech != '':
                            try:
                                h_o36.append(this_data_dict[list_outputs[35]][tech][y])
                            except Exception:
                                h_o36.append(0)
                                count_empties += 1
                        else:
                            h_o36.append(0)
                            count_empties += 1

                        # Output ID: 37***
                        if fuel == '' and tech != '':
                            try:
                                h_o37.append(this_data_dict[list_outputs[36]][tech][y])
                            except Exception:
                                h_o37.append(0)
                                count_empties += 1
                        else:
                            h_o37.append(0)
                            count_empties += 1

                        # Output ID: 38***
                        if fuel == '' and tech != '':
                            try:
                                h_o38.append(this_data_dict[list_outputs[37]][tech][y])
                            except Exception:
                                h_o38.append(0)
                                count_empties += 1
                        else:
                            h_o38.append(0)
                            count_empties += 1

                        # Output ID: 39***
                        if fuel == '' and tech != '':
                            try:
                                h_o39.append(this_data_dict[list_outputs[38]][tech][y])
                            except Exception:
                                h_o39.append(0)
                                count_empties += 1
                        else:
                            h_o39.append(0)
                            count_empties += 1

                        # Output ID: 40***
                        if fuel == '' and tech != '':
                            try:
                                h_o40.append(this_data_dict[list_outputs[39]][tech][y])
                            except Exception:
                                h_o40.append(0)
                                count_empties += 1
                        else:
                            h_o40.append(0)
                            count_empties += 1

                        # Output ID: 41***
                        if fuel == '' and tech != '':
                            try:
                                h_o41.append(this_data_dict[list_outputs[40]][tech][y])
                            except Exception:
                                h_o41.append(0)
                                count_empties += 1
                        else:
                            h_o41.append(0)
                            count_empties += 1

                        # Output ID: 42***
                        if fuel == '' and tech != '':
                            try:
                                h_o42.append(this_data_dict[list_outputs[41]][tech][y])
                            except Exception:
                                h_o42.append(0)
                                count_empties += 1
                        else:
                            h_o42.append(0)
                            count_empties += 1

                        # Output ID: 43***
                        if fuel == '' and tech != '':
                            try:
                                h_o43.append(this_data_dict[list_outputs[42]][tech][y])
                            except Exception:
                                h_o43.append(0)
                                count_empties += 1
                        else:
                            h_o43.append(0)
                            count_empties += 1

                        # Output ID: 44***
                        if fuel == '' and tech != '':
                            try:
                                h_o44.append(this_data_dict[list_outputs[43]][tech][y])
                            except Exception:
                                h_o44.append(0)
                                count_empties += 1
                        else:
                            h_o44.append(0)
                            count_empties += 1

                        # Output ID: 45***
                        if fuel == '' and tech != '':
                            try:
                                h_o45.append(this_data_dict[list_outputs[44]][tech][y])
                            except Exception:
                                h_o45.append(0)
                                count_empties += 1
                        else:
                            h_o45.append(0)
                            count_empties += 1

                        # Output ID: 46***
                        if fuel == '' and tech != '':
                            try:
                                h_o46.append(this_data_dict[list_outputs[45]][tech][y])
                            except Exception:
                                h_o46.append(0)
                                count_empties += 1
                        else:
                            h_o46.append(0)
                            count_empties += 1

                        # Output ID: 47***
                        if fuel == '' and tech != '':
                            try:
                                h_o47.append(this_data_dict[list_outputs[46]][tech][y])
                            except Exception:
                                h_o47.append(0)
                                count_empties += 1
                        else:
                            h_o47.append(0)
                            count_empties += 1

                        # Output ID: 48***
                        if fuel == '' and tech != '':
                            try:
                                h_o48.append(this_data_dict[list_outputs[47]][tech][y])
                            except Exception:
                                h_o48.append(0)
                                count_empties += 1
                        else:
                            h_o48.append(0)
                            count_empties += 1

                        # Output ID: 49***
                        if fuel == '' and tech != '':
                            try:
                                h_o49.append(this_data_dict[list_outputs[48]][tech][y])
                            except Exception:
                                h_o49.append(0)
                                count_empties += 1
                        else:
                            h_o49.append(0)
                            count_empties += 1

                        # Output ID: 50***
                        if fuel == '' and tech != '':
                            try:
                                h_o50.append(this_data_dict[list_outputs[49]][tech][y])
                            except Exception:
                                h_o50.append(0)
                                count_empties += 1
                        else:
                            h_o50.append(0)
                            count_empties += 1

                        # Output ID: 51***
                        if fuel == '' and tech == '':
                            try:
                                h_o51.append(this_data_dict[list_outputs[50]][y])
                            except Exception:
                                h_o51.append(0)
                                count_empties += 1
                        else:
                            h_o51.append(0)
                            count_empties += 1

                        # Output ID: 52***
                        if fuel == '' and tech == '':
                            try:
                                h_o52.append(this_data_dict[list_outputs[51]][y])
                            except Exception:
                                h_o52.append(0)
                                count_empties += 1
                        else:
                            h_o52.append(0)
                            count_empties += 1

                        # Output ID: 53***
                        if fuel == '' and tech != '':
                            try:
                                h_o53.append(this_data_dict[list_outputs[52]][tech][y])
                            except Exception:
                                h_o53.append(0)
                                count_empties += 1
                        else:
                            h_o53.append(0)
                            count_empties += 1

                        # Output ID: 54***
                        if fuel == '' and tech != '':
                            try:
                                h_o54.append(this_data_dict[list_outputs[53]][tech][y])
                            except Exception:
                                h_o54.append(0)
                                count_empties += 1
                        else:
                            h_o54.append(0)
                            count_empties += 1

                        # Output ID: 55***
                        if fuel == '' and tech == '':
                            try:
                                h_o55.append(this_data_dict[list_outputs[54]][y])
                            except Exception:
                                h_o55.append(0)
                                count_empties += 1
                        else:
                            h_o55.append(0)
                            count_empties += 1

                        # Output ID: 56***
                        if fuel == '' and tech != '':
                            try:
                                h_o56.append(this_data_dict[list_outputs[55]][tech][y])
                            except Exception:
                                h_o56.append(0)
                                count_empties += 1
                        else:
                            h_o56.append(0)
                            count_empties += 1

                        # Output ID: 57
                        if fuel != '' and tech != '':
                            #if tech in types_all:
                            #    print('review!')
                            #    sys.exit()
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o57.append(this_data_dict[list_outputs[56]][tech][fuel][y])
                                # print('happens¡¡¡')
                            except Exception:
                                h_o57.append(0)
                                count_empties += 1
                        else:
                            h_o57.append(0)
                            count_empties += 1

                        # Output ID: 58
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o58.append(this_data_dict[list_outputs[57]][tech][fuel][y])
                            except Exception:
                                h_o58.append(0)
                                count_empties += 1
                        else:
                            h_o58.append(0)
                            count_empties += 1

                        # Output ID: 59
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o59.append(this_data_dict[list_outputs[58]][tech][fuel][y])
                            except Exception:
                                h_o59.append(0)
                                count_empties += 1
                        else:
                            h_o59.append(0)
                            count_empties += 1

                        # Output ID: 60
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o60.append(this_data_dict[list_outputs[59]][tech][fuel][y])
                            except Exception:
                                h_o60.append(0)
                                count_empties += 1
                        else:
                            h_o60.append(0)
                            count_empties += 1

                        # Output ID: 61
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o61.append(this_data_dict[list_outputs[60]][tech][fuel][y])
                            except Exception:
                                h_o61.append(0)
                                count_empties += 1
                        else:
                            h_o61.append(0)
                            count_empties += 1

                        # Output ID: 62
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o62.append(this_data_dict[list_outputs[61]][tech][fuel][y])
                            except Exception:
                                h_o62.append(0)
                                count_empties += 1
                        else:
                            h_o62.append(0)
                            count_empties += 1

                        # Output ID: 63
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o63.append(this_data_dict[list_outputs[62]][tech][fuel][y])
                            except Exception:
                                h_o63.append(0)
                                count_empties += 1
                        else:
                            h_o63.append(0)
                            count_empties += 1

                        # Output ID: 64
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o64.append(this_data_dict[list_outputs[63]][tech][fuel][y])
                            except Exception:
                                h_o64.append(0)
                                count_empties += 1
                        else:
                            h_o64.append(0)
                            count_empties += 1

                        # Output ID: 65
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o65.append(this_data_dict[list_outputs[64]][tech][fuel][y])
                            except Exception:
                                h_o65.append(0)
                                count_empties += 1
                        else:
                            h_o65.append(0)
                            count_empties += 1

                        # Output ID: 66
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o66.append(this_data_dict[list_outputs[65]][tech][fuel][y])
                            except Exception:
                                h_o66.append(0)
                                count_empties += 1
                        else:
                            h_o66.append(0)
                            count_empties += 1

                        # Output ID: 67 
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o67.append(this_data_dict[list_outputs[66]][tech][fuel][y])
                            except Exception:
                                h_o67.append(0)
                                count_empties += 1
                        else:
                            h_o67.append(0)
                            count_empties += 1

                        # Output ID: 68
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o68.append(this_data_dict[list_outputs[67]][tech][fuel][y])
                            except Exception:
                                h_o68.append(0)
                                count_empties += 1
                        else:
                            h_o68.append(0)
                            count_empties += 1

                        # Output ID: 69
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o69.append(this_data_dict[list_outputs[68]][tech][fuel][y])
                            except Exception:
                                h_o69.append(0)
                                count_empties += 1
                        else:
                            h_o69.append(0)
                            count_empties += 1

                        # Output ID: 70
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o70.append(this_data_dict[list_outputs[69]][tech][fuel][y])
                            except Exception:
                                h_o70.append(0)
                                count_empties += 1
                        else:
                            h_o70.append(0)
                            count_empties += 1

                        # Output ID: 71
                        if fuel != '' and tech != '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o71.append(this_data_dict[list_outputs[70]][tech][fuel][y])
                            except Exception:
                                h_o71.append(0)
                                count_empties += 1
                        else:
                            h_o71.append(0)
                            count_empties += 1

                        # Output ID: 72
                        if fuel == '' and tech == '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o72.append(this_data_dict[list_outputs[71]][y])
                            except Exception:
                                h_o72.append(0)
                                count_empties += 1
                        else:
                            h_o72.append(0)
                            count_empties += 1

                        # Output ID: 73
                        if fuel == '' and tech == '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o73.append(this_data_dict[list_outputs[72]][y])
                            except Exception:
                                h_o73.append(0)
                                count_empties += 1
                        else:
                            h_o73.append(0)
                            count_empties += 1

                        # Output ID: 74
                        if fuel == '' and tech == '':
                            try:
                                # fuel_app = dict_eq_trn_fuels_rev[fuel]
                                h_o74.append(this_data_dict[list_outputs[73]][y])
                            except Exception:
                                h_o74.append(0)
                                count_empties += 1
                        else:
                            h_o74.append(0)
                            count_empties += 1

                        # Output ID: 75
                        if fuel != '' and tech != '':  # check energy demand
                            try:
                                h_o75.append(this_data_dict[list_outputs[74]][tech][fuel][y])
                            except Exception:
                                h_o75.append(0)
                                count_empties += 1
                        else:
                            h_o75.append(0)
                            count_empties += 1

                        # Output ID: 76
                        if fuel != '' and tech != '':  # check energy demand
                            try:
                                h_o76.append(this_data_dict[list_outputs[75]][tech][fuel][y])
                            except Exception:
                                h_o76.append(0)
                                count_empties += 1
                        else:
                            h_o76.append(0)
                            count_empties += 1

                        if count_empties == 86:  # gotta pop, because it is an empty row:
                            h_i1.pop() #1
                            h_i2.pop() #2
                            h_i3.pop() #3
                            h_i4.pop() #4
                            h_i5.pop() #5
                            h_i6.pop() #6
                            h_i7.pop() #7
                            h_i8.pop() #8
                            h_i9.pop() #9
                            h_i10.pop() #10
                            h_o1.pop() #11
                            h_o2.pop() #12
                            h_o3.pop() #13
                            h_o4.pop() #14
                            h_o5.pop() #15
                            h_o6.pop() #16
                            h_o7.pop() #17
                            h_o8.pop() #18
                            h_o9.pop() #19
                            h_o10.pop() #20
                            h_o11.pop() #21
                            h_o12.pop() #22
                            h_o13.pop() #23
                            h_o14.pop() #24
                            h_o15.pop() #25
                            h_o16.pop() #26
                            h_o17.pop() #27
                            h_o18.pop() #28* (old)
                            h_o19.pop() #29
                            h_o20.pop() #30
                            h_o21.pop() #31
                            h_o22.pop() #32
                            h_o23.pop() #33
                            h_o24.pop() #34
                            h_o25.pop() #35
                            h_o26.pop() #36
                            h_o27.pop() #37
                            h_o28.pop() #38
                            h_o29.pop() #39
                            h_o30.pop() #40* (old)
                            h_o31.pop() #41
                            h_o32.pop() #42
                            h_o33.pop() #43
                            h_o34.pop() #44
                            h_o35.pop() #45
                            h_o36.pop() #46
                            h_o37.pop() #47
                            h_o38.pop() #48
                            h_o39.pop() #49
                            h_o40.pop() #50
                            h_o41.pop() #51
                            h_o42.pop() #52
                            h_o43.pop() #53
                            h_o44.pop() #54* (old)
                            h_o45.pop() #55
                            h_o46.pop() #56
                            h_o47.pop() #57* (old)
                            h_o48.pop() #58
                            h_o49.pop() #59
                            h_o50.pop() #60
                            h_o51.pop() #61
                            h_o52.pop() #62
                            h_o53.pop() #63* (old)
                            h_o54.pop() #64* (old)
                            h_o55.pop() #65
                            h_o56.pop() #66
                            h_o57.pop() #67
                            h_o58.pop() #68
                            h_o59.pop() #69
                            h_o60.pop() #70
                            h_o61.pop() #71
                            h_o62.pop() #72
                            h_o63.pop() #73
                            h_o64.pop() #74
                            h_o65.pop() #75
                            h_o66.pop() #76
                            h_o67.pop() #77
                            h_o68.pop() #78
                            h_o69.pop() #79
                            h_o70.pop() #80
                            h_o71.pop() #81
                            h_o72.pop() #81
                            h_o73.pop() #82
                            h_o74.pop() #83
                            h_o75.pop() #84
                            h_o76.pop() #85

                        else:
                            h_strategy.append(this_scen)
                            h_region.append(this_reg)
                            h_country.append(this_country)
                            h_tech.append(tech)
                            h_techtype.append(tech_type)
                            h_fuel.append(fuel)
                            h_yr.append(time_vector[y])

# Review if *zero* elements exist:
list_variables = \
    [h_i1, h_i2, h_i3, h_i4, h_i5, h_i6, h_i7, h_i8, h_i9, h_i10,
     h_o1, h_o2, h_o3, h_o4, h_o5, h_o6, h_o7, h_o8, h_o9, h_o10,
     h_o11, h_o12, h_o13, h_o14, h_o15, h_o16, h_o17, h_o18, h_o19, h_o20,
     h_o21, h_o22, h_o23, h_o24, h_o25, h_o26, h_o27, h_o28, h_o29, h_o30,
     h_o31, h_o32, h_o33, h_o34, h_o35, h_o36, h_o37, h_o38, h_o39, h_o40,
     h_o41, h_o42, h_o43, h_o44, h_o45, h_o46, h_o47, h_o48, h_o49, h_o50,
     h_o51, h_o52, h_o53, h_o54, h_o55, h_o56, h_o57, h_o58, h_o59, h_o60,
     h_o61, h_o62, h_o63, h_o64, h_o65, h_o66, h_o67, h_o68, h_o69, h_o70,
     h_o71, h_o72, h_o73, h_o74, h_o75, h_o76]

h_count = 0
h_zeros = []
for h in list_variables:
    if sum(h) == 0.0:
        h_zeros.append(h_count)
    h_count += 1

# Review the lengths:
print(1, list_dimensions[0], len(h_strategy)) #1
print(2, list_dimensions[1], len(h_region)) #2
print(3, list_dimensions[2], len(h_country)) #3
print(4, list_dimensions[3], len(h_tech)) #4
print(5, list_dimensions[4], len(h_techtype)) #5
print(6, list_dimensions[5], len(h_fuel)) #6
print(7, list_dimensions[6], len(h_yr)) #7
print(8, list_inputs_add[0], len(h_i1)) #8
print(9, list_inputs_add[1], len(h_i2)) #9
print(10, list_inputs_add[2], len(h_i3)) #10
print(11, list_inputs_add[3], len(h_i4)) #11
print(12, list_inputs_add[4], len(h_i5)) #12
print(13, list_inputs_add[5], len(h_i6)) #13
print(14, list_inputs_add[6], len(h_i7)) #14
print(15, list_inputs_add[7], len(h_i8)) #15
print(16, list_inputs_add[8], len(h_i9)) #16
print(17, list_inputs_add[9], len(h_i10)) #17
print(18, list_outputs_add[0], len(h_o1)) #18
print(19, list_outputs_add[1], len(h_o2)) #19
print(20, list_outputs_add[2], len(h_o3)) #20
print(21, list_outputs_add[3], len(h_o4)) #21
print(22, list_outputs_add[4], len(h_o5)) #22
print(23, list_outputs_add[5], len(h_o6)) #23
print(24, list_outputs_add[6], len(h_o7)) #24
print(25, list_outputs_add[7], len(h_o8)) #25
print(26, list_outputs_add[8], len(h_o9)) #26
print(27, list_outputs_add[9], len(h_o10)) #27
print(28, list_outputs_add[10], len(h_o11)) #28
print(29, list_outputs_add[11], len(h_o12)) #29
print(30, list_outputs_add[12], len(h_o13)) #30
print(31, list_outputs_add[13], len(h_o14)) #31
print(32, list_outputs_add[14], len(h_o15)) #32
print(33, list_outputs_add[15], len(h_o16)) #33
print(34, list_outputs_add[16], len(h_o17)) #34
print(35, list_outputs_add[17], len(h_o18)) #35
print(36, list_outputs_add[18], len(h_o19)) #36
print(37, list_outputs_add[19], len(h_o20)) #37
print(38, list_outputs_add[20], len(h_o21)) #38
print(39, list_outputs_add[21], len(h_o22)) #39
print(40, list_outputs_add[22], len(h_o23)) #40
print(41, list_outputs_add[23], len(h_o24)) #41
print(42, list_outputs_add[24], len(h_o25)) #42
print(43, list_outputs_add[25], len(h_o26)) #43
print(44, list_outputs_add[26], len(h_o27)) #44
print(45, list_outputs_add[27], len(h_o28)) #45
print(46, list_outputs_add[28], len(h_o29)) #46
print(47, list_outputs_add[29], len(h_o30)) #47

print(48, list_outputs_add[30], len(h_o31)) #48
print(49, list_outputs_add[31], len(h_o32)) #49
print(50, list_outputs_add[32], len(h_o33)) #50
print(51, list_outputs_add[33], len(h_o34)) #51
print(52, list_outputs_add[34], len(h_o35)) #52
print(53, list_outputs_add[35], len(h_o36)) #53
print(54, list_outputs_add[36], len(h_o37)) #54
print(55, list_outputs_add[37], len(h_o38)) #55
print(56, list_outputs_add[38], len(h_o39)) #56
print(57, list_outputs_add[39], len(h_o40)) #57
print(58, list_outputs_add[40], len(h_o41)) #58
print(59, list_outputs_add[41], len(h_o42)) #59
print(60, list_outputs_add[42], len(h_o43)) #60
print(61, list_outputs_add[43], len(h_o44)) #61

print(62, list_outputs_add[44], len(h_o45)) #62
print(63, list_outputs_add[45], len(h_o46)) #63
print(64, list_outputs_add[46], len(h_o47)) #64

print(65, list_outputs_add[47], len(h_o48)) #65
print(66, list_outputs_add[48], len(h_o49)) #66
print(67, list_outputs_add[49], len(h_o50)) #67
print(68, list_outputs_add[50], len(h_o51)) #68
print(69, list_outputs_add[51], len(h_o52)) #69
print(70, list_outputs_add[52], len(h_o53)) #70
print(71, list_outputs_add[53], len(h_o54)) #71
print(72, list_outputs_add[54], len(h_o55)) #72
print(73, list_outputs_add[55], len(h_o56)) #73

print(74, list_outputs_add[56], len(h_o57)) #74
print(75, list_outputs_add[57], len(h_o58)) #75
print(76, list_outputs_add[58], len(h_o59)) #76
print(77, list_outputs_add[59], len(h_o60)) #77
print(78, list_outputs_add[60], len(h_o61)) #78
print(79, list_outputs_add[61], len(h_o62)) #79
print(80, list_outputs_add[62], len(h_o63)) #80
print(81, list_outputs_add[63], len(h_o64)) #81
print(82, list_outputs_add[64], len(h_o65)) #82
print(83, list_outputs_add[65], len(h_o66)) #83
print(84, list_outputs_add[66], len(h_o67)) #84
print(85, list_outputs_add[67], len(h_o68)) #85
print(86, list_outputs_add[68], len(h_o69)) #86
print(87, list_outputs_add[69], len(h_o70)) #87
print(88, list_outputs_add[70], len(h_o71)) #88

print(89, list_outputs_add[71], len(h_o72)) #89
print(90, list_outputs_add[72], len(h_o73)) #90
print(91, list_outputs_add[73], len(h_o74)) #91

print(92, list_outputs_add[74], len(h_o75)) #92
print(93, list_outputs_add[75], len(h_o76)) #93

# Convert to output:
print('\n')
print('Convert lists to dataframe for printing:')
dict_output = {list_dimensions[0]: h_strategy, #1
               list_dimensions[1]: h_region, #2
               list_dimensions[2]: h_country, #3
               list_dimensions[3]: h_tech, #4
               list_dimensions[4]: h_techtype, #5
               list_dimensions[5]: h_fuel, #6
               list_dimensions[6]: h_yr, #7
               list_inputs_add[0]: h_i1, #8
               list_inputs_add[1]: h_i2, #9
               list_inputs_add[2]: h_i3, #10
               list_inputs_add[3]: h_i4, #11
               list_inputs_add[4]: h_i5, #12
               list_inputs_add[5]: h_i6, #13
               list_inputs_add[6]: h_i7, #14
               list_inputs_add[7]: h_i8, #15
               list_inputs_add[8]: h_i9, #16
               list_inputs_add[9]: h_i10, #17
               list_outputs_add[0]: h_o1, #18
               list_outputs_add[1]: h_o2, #19
               list_outputs_add[2]: h_o3, #20
               list_outputs_add[3]: h_o4, #21
               list_outputs_add[4]: h_o5, #22
               list_outputs_add[5]: h_o6, #23
               list_outputs_add[6]: h_o7, #24
               list_outputs_add[7]: h_o8, #25
               list_outputs_add[8]: h_o9, #26
               list_outputs_add[9]: h_o10, #27
               list_outputs_add[10]: h_o11, #28
               list_outputs_add[11]: h_o12, #29
               list_outputs_add[12]: h_o13, #30
               list_outputs_add[13]: h_o14, #31
               list_outputs_add[14]: h_o15, #32
               list_outputs_add[15]: h_o16, #33
               list_outputs_add[16]: h_o17, #34
               list_outputs_add[17]: h_o18, #35
               list_outputs_add[18]: h_o19, #36
               list_outputs_add[19]: h_o20, #37
               list_outputs_add[20]: h_o21, #38
               list_outputs_add[21]: h_o22, #39
               list_outputs_add[22]: h_o23, #40
               list_outputs_add[23]: h_o24, #41
               list_outputs_add[24]: h_o25, #42
               list_outputs_add[25]: h_o26, #43
               list_outputs_add[26]: h_o27, #44
               list_outputs_add[27]: h_o28, #45
               list_outputs_add[28]: h_o29, #46
               list_outputs_add[29]: h_o30, #47

               list_outputs_add[30]: h_o31, #48
               list_outputs_add[31]: h_o32, #49
               list_outputs_add[32]: h_o33, #50
               list_outputs_add[33]: h_o34, #51
               list_outputs_add[34]: h_o35, #52
               list_outputs_add[35]: h_o36, #53
               list_outputs_add[36]: h_o37, #54
               list_outputs_add[37]: h_o38, #55
               list_outputs_add[38]: h_o39, #56
               list_outputs_add[39]: h_o40, #57
               list_outputs_add[40]: h_o41, #58
               list_outputs_add[41]: h_o42, #59
               list_outputs_add[42]: h_o43, #60
               list_outputs_add[43]: h_o44, #61

               list_outputs_add[44]: h_o45, #62
               list_outputs_add[45]: h_o46, #63
               list_outputs_add[46]: h_o47, #64

               list_outputs_add[47]: h_o48, #65
               list_outputs_add[48]: h_o49, #66
               list_outputs_add[49]: h_o50, #67
               list_outputs_add[50]: h_o51, #68
               list_outputs_add[51]: h_o52, #69
               list_outputs_add[52]: h_o53, #70
               list_outputs_add[53]: h_o54, #71
               list_outputs_add[54]: h_o55, #72
               list_outputs_add[55]: h_o56, #73

               list_outputs_add[56]: h_o57, #74
               list_outputs_add[57]: h_o58, #75
               list_outputs_add[58]: h_o59, #76
               list_outputs_add[59]: h_o60, #77
               list_outputs_add[60]: h_o61, #78
               list_outputs_add[61]: h_o62, #79
               list_outputs_add[62]: h_o63, #80
               list_outputs_add[63]: h_o64, #81
               list_outputs_add[64]: h_o65, #82
               list_outputs_add[65]: h_o66, #83
               list_outputs_add[66]: h_o67, #84
               list_outputs_add[67]: h_o68, #85
               list_outputs_add[68]: h_o69, #86
               list_outputs_add[69]: h_o70, #87
               list_outputs_add[70]: h_o71, #88
               
               list_outputs_add[71]: h_o72, #89
               list_outputs_add[72]: h_o73, #90
               list_outputs_add[73]: h_o74, #91
               
               list_outputs_add[74]: h_o75, #92
               list_outputs_add[75]: h_o76, #93
               }

df_output_name = 'model_BULAC_simulation.csv'
df_output = pd.DataFrame.from_dict(dict_output)
df_output.to_csv(path + '/' + df_output_name, index=None, header=True)

df_output_name_f0 = 'model_BULAC_simulation_0.csv'
df_output_f0 = deepcopy(df_output)
df_output_f0['Future'] = 0
list_inner = list_dimensions + [
                    'Emissions by demand (output)',
                    'Emissions in electricity (output)',
                    'Energy demand by sector (output)',
                    'Energy demand by fuel (output)',
                    'Energy intensity by sector (output)',
                    'Global warming externalities by demand (output)',
                    'Global warming externalities in electricity (output)',
                    'Local pollution externalities by demand (output)',
                    'Local pollution externalities in electricity (output)',
                    'Global warming externalities in electricity (disc) (output)',
                    'Local pollution externalities in electricity (disc) (output)',
                    'Fleet (output)',
                    'New Fleet (output)',
                    'Transport CAPEX [$] (output)',
                    'Transport Fixed OPEX [$] (output)',
                    'Transport Variable OPEX [$] (output)',
                    'Transport Tax Imports [$] (output)',
                    'Transport Tax IMESI_Venta [$] (output)',
                    'Transport Tax IVA_Venta [$] (output)',
                    'Transport Tax Patente [$] (output)',
                    'Transport Tax IMESI_Combust [$] (output)',
                    'Transport Tax IVA_Gasoil [$] (output)',
                    'Transport Tax IVA_Elec [$] (output)',
                    'Transport Tax IC [$] (output)',
                    'Transport Tax Otros_Gasoil [$] (output)',
                    'Transport Tax Tasa_Consular [$] (output)',
                    'Future']
df_output_f0_out = df_output_f0[list_inner]
df_output_f0_out.to_csv(path + '/' + df_output_name_f0, index=None, header=True)

# Recording final time of execution:
end_f = time.time()
te_f = -start_1 + end_f  # te: time_elapsed
print(str(te_f) + ' seconds /', str(te_f/60) + ' minutes')
print('*: This automatic analysis is finished.')
