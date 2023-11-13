# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13, 2021
Last updated: Apr. 04, 2023

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


# Function 1: find the intersection between to lists
def intersection_2(lst1, lst2):
    return list(set(lst1) & set(lst2))


# Function 2: find the interpolated values of a list
def interpolation_to_end_debug(time_vector, ini_simu_yr, number_vector,
                         interp_indicator, this_scen, this_action):
    x_coord_tofill, xp_coord_known, yp_coord_known = [], [], []

    initial_year_of_change = ini_simu_yr
    if this_action == 'Population':
        initial_year_of_change = time_vector[1]  # overwrite to year after base year
    initial_year_of_change_index = time_vector.index(initial_year_of_change)

    for y in range(len(time_vector)):
        if y > initial_year_of_change_index:
            if number_vector[y] != '':
                xp_coord_known.append(y)
                yp_coord_known.append(number_vector[y])
            else:
                x_coord_tofill.append(y)
        elif y == initial_year_of_change_index:
            if (interp_indicator == 'last' and this_action == 'power'):
                xp_coord_known.append(y)
                yp_coord_known.append(number_vector[-1])
            else:
                if number_vector[y] != '':
                    xp_coord_known.append(y)
                    yp_coord_known.append(number_vector[y])
                else:
                    x_coord_tofill.append(y)
        elif y < initial_year_of_change_index and number_vector[y] != '':
            xp_coord_known.append(y)
            yp_coord_known.append(number_vector[y])
        else:
            xp_coord_known.append(y)
            if interp_indicator == 'last':
                yp_coord_known.append(number_vector[0])
            if interp_indicator == 'ini':
                yp_coord_known.append(number_vector[0])

    # try:
    y_coord_filled = list(np.interp(x_coord_tofill, xp_coord_known, yp_coord_known))
    # except Exception:
    print(x_coord_tofill, len(x_coord_tofill))
    print(xp_coord_known, len(xp_coord_known))
    print(yp_coord_known, len(yp_coord_known))
    '''
    try:
        y_coord_filled = list(np.interp(x_coord_tofill, xp_coord_known, yp_coord_known))
    except Exception:
        print(x_coord_tofill)
        print(xp_coord_known)
        print(yp_coord_known)
    '''

    interpolated_values = []
    for coord in range(len(time_vector)):
        if coord in xp_coord_known:
            value_index = xp_coord_known.index(coord)
            interpolated_values.append(float(yp_coord_known[value_index]))
        elif coord in x_coord_tofill:
            value_index = x_coord_tofill.index(coord)
            interpolated_values.append(float(y_coord_filled[value_index]))

    return deepcopy(interpolated_values)



# Function 2: find the interpolated values of a list
def interpolation_to_end(time_vector, ini_simu_yr, number_vector,
                         interp_indicator, this_scen, this_action):
    x_coord_tofill, xp_coord_known, yp_coord_known = [], [], []

    initial_year_of_change = ini_simu_yr
    if this_action == 'Population':
        initial_year_of_change = time_vector[1]  # overwrite to year after base year
    initial_year_of_change_index = time_vector.index(initial_year_of_change)

    for y in range(len(time_vector)):
        if y > initial_year_of_change_index:
            if number_vector[y] != '':
                xp_coord_known.append(y)
                yp_coord_known.append(number_vector[y])
            else:
                x_coord_tofill.append(y)
        elif y == initial_year_of_change_index:
            if (interp_indicator == 'last' and this_action == 'power'):
                xp_coord_known.append(y)
                yp_coord_known.append(number_vector[-1])
            else:
                if number_vector[y] != '':
                    xp_coord_known.append(y)
                    yp_coord_known.append(number_vector[y])
                else:
                    x_coord_tofill.append(y)
        elif y < initial_year_of_change_index and number_vector[y] != '':
            xp_coord_known.append(y)
            yp_coord_known.append(number_vector[y])
        else:
            xp_coord_known.append(y)
            if interp_indicator == 'last':
                yp_coord_known.append(number_vector[0])
            if interp_indicator == 'ini':
                yp_coord_known.append(number_vector[0])

    try:
        y_coord_filled = list(np.interp(x_coord_tofill, xp_coord_known, yp_coord_known))
    except Exception:
        print(x_coord_tofill, len(x_coord_tofill))
        print(xp_coord_known, len(xp_coord_known))
        print(yp_coord_known, len(yp_coord_known))
    '''
    try:
        y_coord_filled = list(np.interp(x_coord_tofill, xp_coord_known, yp_coord_known))
    except Exception:
        print(x_coord_tofill)
        print(xp_coord_known)
        print(yp_coord_known)
    '''

    interpolated_values = []
    for coord in range(len(time_vector)):
        if coord in xp_coord_known:
            value_index = xp_coord_known.index(coord)
            interpolated_values.append(float(yp_coord_known[value_index]))
        elif coord in x_coord_tofill:
            value_index = x_coord_tofill.index(coord)
            interpolated_values.append(float(y_coord_filled[value_index]))

    return deepcopy(interpolated_values)


# Function 3: print the dictionary in table format for a single country
#   This way, we can add a new sheet that a user can modify with new
# variables and sets.
def fun_reverse_dict_data(dict_database_frz, apply_reg, apply_country,
                          print_new_fuels, new_fuel_list, pack_fe):
    # Analyze the energy balance and create an equivalent df:
    dict_EB_print = {'Region':[], 'Country':[], 'Variable':[],
                     'Sector':[], 'Fuel':[], 'Year':[], 'Value':[]}
    apply_EB_dict = dict_database_frz['EB'][apply_reg][apply_country]
    var_list = list(dict.fromkeys(apply_EB_dict))
    for var in var_list:
        sec_list = list(dict.fromkeys(apply_EB_dict[var]))
        for sec in sec_list:
            fuel_list = list(dict.fromkeys(apply_EB_dict[var][sec]))
            for fuel in fuel_list:
                years_list = \
                    list(dict.fromkeys(apply_EB_dict[var][sec][fuel]))
                years_list_EB = deepcopy(years_list)
                for y in years_list:
                    a_value = apply_EB_dict[var][sec][fuel][y]
                    dict_EB_print['Region'].append(apply_reg)
                    dict_EB_print['Country'].append(apply_country)
                    dict_EB_print['Variable'].append(var)
                    dict_EB_print['Sector'].append(sec)
                    dict_EB_print['Fuel'].append(fuel)
                    dict_EB_print['Year'].append(y)
                    dict_EB_print['Value'].append(a_value)
    df_EB_print = pd.DataFrame.from_dict(dict_EB_print)

    # Analyze the installed capacity and create an equivalent df:
    dict_IC_print = {'Region':[], 'Country':[], 'Technology':[],
                     'Year':[], 'Value':[]}
    apply_IC_dict = dict_database_frz['Cap'][apply_reg][apply_country]
    techs_list = list(dict.fromkeys(apply_IC_dict))
    for tech in techs_list:
        years_list = list(dict.fromkeys(apply_IC_dict[tech]))
        for y in years_list:
            a_value = apply_IC_dict[tech][y]
            dict_IC_print['Region'].append(apply_reg)
            dict_IC_print['Country'].append(apply_country)
            dict_IC_print['Technology'].append(tech)
            dict_IC_print['Year'].append(y)
            dict_IC_print['Value'].append(a_value)
    df_IC_print = pd.DataFrame.from_dict(dict_IC_print)

    # Print an energy balance that has the updated fuel list:
    new_fuel_list += ['Total']  # We need a "Total" key to reuse code
    if print_new_fuels is True:
        dict_EB2_print = {'Region':[], 'Country':[], 'Variable':[],
                          'Sector':[], 'Fuel':[], 'Year':[], 'Value':[]}
        for var in var_list:
            sec_list = list(dict.fromkeys(apply_EB_dict[var]))
            for sec in sec_list:
                av_fuel_list = \
                    list(dict.fromkeys(apply_EB_dict[var][sec]))
                for fuel in new_fuel_list:
                    try:
                        fuel_eq = pack_fe['new2old'][fuel]
                    except Exception:
                        fuel_eq = ''
                    for y in years_list_EB:
                        if fuel_eq in av_fuel_list:
                            a_value = \
                                apply_EB_dict[var][sec][fuel_eq][y]
                        else:
                            a_value = 0
                        dict_EB2_print['Region'].append(apply_reg)
                        dict_EB2_print['Country'].append(apply_country)
                        dict_EB2_print['Variable'].append(var)
                        dict_EB2_print['Sector'].append(sec)
                        dict_EB2_print['Fuel'].append(fuel)
                        dict_EB2_print['Year'].append(y)
                        dict_EB2_print['Value'].append(a_value)
        df_EB2_print = pd.DataFrame.from_dict(dict_EB2_print)

    # Print the dataframes for review:
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    a_writer = pd.ExcelWriter('dict_data.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    df_EB_print.to_excel(a_writer, sheet_name='EneBal', index=False)
    df_IC_print.to_excel(a_writer, sheet_name='InsCap', index=False)
    if print_new_fuels is True:
        df_EB2_print.to_excel(a_writer, sheet_name='EneBalAdj',
                              index=False)

    # Close the Pandas Excel writer and output the Excel file.
    a_writer.save()

    # print("Print this if you got here")
    # sys.exit()

    # return None


# Function 4: define the new dictionary data.
def fun_extract_new_dict_data(df2_EB, df2_InsCap, per_first_yr):
    start_overwrite = time.time()

    # Define the lists of the dataframe (or columns):
    col_EB_reg = df2_EB['Region'].tolist()
    col_EB_cou = df2_EB['Country'].tolist()
    col_EB_var = df2_EB['Variable'].tolist()
    col_EB_sec = df2_EB['Sector'].tolist()
    col_EB_fue = df2_EB['Fuel'].tolist()
    col_EB_yea = df2_EB['Year'].tolist()
    col_EB_val = df2_EB['Value'].tolist()

    d_EB = {}
    col_EB_reg_u = list(dict.fromkeys(col_EB_reg))
    col_EB_cou_u = list(dict.fromkeys(col_EB_cou))
    col_EB_var_u = list(dict.fromkeys(col_EB_var))
    col_EB_sec_u = list(dict.fromkeys(col_EB_sec))
    col_EB_fue_u = list(dict.fromkeys(col_EB_fue))
    # col_EB_yea_u = list(dict.fromkeys(col_EB_yea))
    col_EB_yea_u = [per_first_yr]
    
    #print(col_EB_yea_u)
    #sys.exit()
    
    for r in col_EB_reg_u:
        for cou in col_EB_cou_u:
            for var in col_EB_var_u:
                for s in col_EB_sec_u:
                    for f in col_EB_fue_u:
                        for y in col_EB_yea_u:
                            # The code below must find the indices
                            # of an occurrence on a list.
                            #idx_reg = \
                            #    [i for i, v in enumerate(col_EB_reg)
                            #     if v == r]
                            #idx_cou = \
                            #    [i for i, v in enumerate(col_EB_cou)
                            #     if v == cou]
                            idx_var = \
                                [i for i, v in enumerate(col_EB_var)
                                 if v == var]
                            idx_sec = \
                                [i for i, v in enumerate(col_EB_sec)
                                 if v == s]
                            idx_fue = \
                                [i for i, v in enumerate(col_EB_fue)
                                 if v == f]
                            idx_y = \
                                [i for i, v in enumerate(col_EB_yea)
                                 if v == y]
                            idx_val = \
                                list(set(idx_var) & set(idx_sec) &\
                                     set(idx_fue) & set(idx_y))
                            if len(idx_val) > 1:
                                print('Logic failed (Flag 1)')
                                print(type(idx_val))
                                print(len(idx_val))
                                print(r, cou, var, s, f, y)
                                sys.exit()
                            if len(idx_val) == 1:
                                val = col_EB_val[idx_val[0]]
                                if r not in list(d_EB.keys()):
                                    d_EB.update({r:{}})
                                if cou not in list(d_EB[r].keys()):
                                    d_EB[r].update({cou:{}})
                                if var not in list(d_EB[r][cou].keys()):
                                    d_EB[r][cou].update({var:{}})
                                if s not in list(d_EB[r][cou][var].keys()):
                                    d_EB[r][cou][var].update({s:{}})
                                if f not in list(d_EB[r][cou][var][s].keys()):
                                    d_EB[r][cou][var][s].update({f:{}})
                                d_EB[r][cou][var][s][f].update({str(y):val})
                                # print(list(d_EB.keys()))
                            else:
                                pass
                                # print('happens')

    col_InstCap_reg = df2_InsCap['Region'].tolist()
    col_InstCap_cou = df2_InsCap['Country'].tolist()
    col_InstCap_tec = df2_InsCap['Technology'].tolist()
    col_InstCap_yea = df2_InsCap['Year'].tolist()
    col_InstCap_val = df2_InsCap['Value'].tolist()

    dict_InstCap = {}
    col_InstCap_reg_u = list(dict.fromkeys(col_InstCap_reg))
    col_InstCap_cou_u = list(dict.fromkeys(col_InstCap_cou))
    col_InstCap_tec_u = list(dict.fromkeys(col_InstCap_tec))
    col_InstCap_yea_u = list(dict.fromkeys(col_InstCap_yea))
    for r in col_InstCap_reg_u:
        for cou in col_InstCap_cou_u:
            for tec in col_InstCap_tec_u:
                for y in col_InstCap_yea_u:
                    # The code below must find the indices
                    # of an occurrence on a list.
                    #idx_reg = \
                    #    [i for i, v in enumerate(col_InstCap_reg)
                    #     if v == r]
                    #idx_cou = \
                    #    [i for i, v in enumerate(col_InstCap_cou)
                    #     if v == cou]
                    idx_tec = \
                        [i for i, v in enumerate(col_InstCap_tec)
                         if v == tec]
                    idx_y = \
                        [i for i, v in enumerate(col_InstCap_yea)
                         if v == y]
                    idx_val = list(set(idx_tec) & set(idx_y))
                    if len(idx_val) > 1:
                        print('Logic failed (Flag 1)')
                        print(type(idx_val))
                        print(len(idx_val))
                        print(r, cou, tec, y)
                        sys.exit()
                    if len(idx_val) == 1:
                        val = col_InstCap_val[idx_val[0]]
                        if r not in list(dict_InstCap.keys()):
                            dict_InstCap.update({r:{}})
                        if cou not in list(dict_InstCap[r].keys()):
                            dict_InstCap[r].update({cou:{}})
                        if tec not in list(dict_InstCap[r][cou].keys()):
                            dict_InstCap[r][cou].update({tec:{}})
                        if y not in list(dict_InstCap[r][cou][tec].keys()):
                            dict_InstCap[r][cou][tec].update({str(y):val})
    end_overwrite = time.time()
    time_overwrite = -start_overwrite + end_overwrite  # te: time_elapsed
    print(str(time_overwrite) + ' seconds /',
          str(time_overwrite/60) + ' minutes')
    return d_EB, dict_InstCap


# Function 5: extract a variable from the transport model: 
def fun_dem_model_projtype(app_type, app_country, app_var, app_col,
                           df_trans_data):
    mask_app = \
        (df_trans_data['Type'] == app_type) & \
        (df_trans_data['Application_Countries'] == app_country) & \
        (df_trans_data['Parameter'] == app_var)
    get_dem_model_var = df_trans_data.loc[mask_app][app_col].iloc[0]
    return get_dem_model_var, mask_app


# Function 6: perform projections for the transport model: 
def fun_dem_proj(time_vector, projtype, mask_app, df_trans_data):
    list_a_var = []
    for y in range(len(time_vector)):
        if projtype == 'flat':
            add_a_var = \
                df_trans_data.loc[mask_app][time_vector[0]].iloc[0]
        elif projtype == 'user_defined':
            add_a_var = \
                df_trans_data.loc[mask_app][time_vector[y]].iloc[0]
        else:
            print('Undefined elasticity projection. Review.')
            sys.exit()
        list_a_var.append(add_a_var)
    return list_a_var


# Function 7: unpack cost variables: 
def fun_unpack_costs(param, tech, fuel, df_costs_trn, time_vector):
    mask_costs_trn = (df_costs_trn['Parameter'] == param) & \
        (df_costs_trn['Type'] == tech) & \
        (df_costs_trn['Fuel'] == fuel)
    use_df = df_costs_trn.loc[mask_costs_trn]
    by_val = use_df[time_vector[0]].iloc[0]

    use_proj = use_df['projection'].iloc[0]
    use_unit = use_df['Unit'].iloc[0]

    list_val = []
    
    if use_proj == 'flat':
        for y in range(len(time_vector)):
            list_val.append(by_val)
    elif use_proj == 'user_defined':
        for y in range(len(time_vector)):
            list_val = \
                df_costs_trn.loc[mask_costs_trn][time_vector].iloc[0].tolist()
    elif use_proj == 'normalized-trajectory':
        list_val.append(by_val)
        for y in range(1, len(time_vector)):
            rel_val = use_df[time_vector[y]].iloc[0]
            list_val.append(rel_val*by_val)
    else:
        print('Undefined projection mode. Please check!')
        sys.exit()

    return list_val, use_unit


# Function 8: unpack tax variables: 
def fun_unpack_taxes(param, tech, fuel, df_tax_trn, time_vector):
    '''
    If a tax is undefined for a specific combination of technology (t) and 
    fuel (f), then, the tax is zero.
    '''
    try:
        mask_tax_trn = (df_tax_trn['Parameter'] == param) & \
            (df_tax_trn['Type'] == tech) & \
            (df_tax_trn['Fuel'] == fuel)
        use_df = df_tax_trn.loc[mask_tax_trn]
        by_val = use_df[time_vector[0]].iloc[0]
    
        use_proj = use_df['projection'].iloc[0]
        ref_param = use_df['Ref_Parameter'].iloc[0]
        mult_depr = use_df['depreciation_factor'].iloc[0]

        if math.isnan(mult_depr):
            mult_depr = 1

        list_val = []
        
        if use_proj == 'flat':
            for y in range(len(time_vector)):
                list_val.append(by_val)
        elif use_proj == 'user_defined':
            list_val = \
                df_tax_trn.loc[mask_tax_trn][time_vector].iloc[0].tolist()
        else:
            print('Undefined projection mode. Please check!')
            sys.exit()
    except Exception:
        list_val = [0 for y in range(len(time_vector))]
        ref_param = 'None'
        mult_depr = 1

    return list_val, ref_param, mult_depr


# Function 9: interpolate non_linear final to update list values 
def interpolation_non_linear_final(time_list, value_list,
                                   new_relative_final_value,
                                   Initial_Year_of_Uncertainty):
    # Rememeber that the 'old_relative_final_value' is 1
    old_relative_final_value = 1
    new_value_list = []
    # We select a list that goes from the "Initial_Year_of_Uncertainty" to the Final Year of the Time Series
    initial_year_index = time_list.index( Initial_Year_of_Uncertainty )
    fraction_time_list = time_list[initial_year_index:]
    fraction_value_list = value_list[initial_year_index:]
    # We now perform the 'non-linear OR linear adjustment':
    xdata = [ fraction_time_list[i] - fraction_time_list[0] for i in range( len( fraction_time_list ) ) ]
    ydata = [ float( fraction_value_list[i] ) for i in range( len( fraction_value_list ) ) ]
    delta_ydata = [ ydata[i]-ydata[i-1] for i in range( 1,len( ydata ) ) ]
    #
    m_original = ( ydata[-1]-ydata[0] ) / ( xdata[-1]-xdata[0] )
    #
    m_new = ( ydata[-1]*(new_relative_final_value/old_relative_final_value) - ydata[0] ) / ( xdata[-1]-xdata[0] )
    #
    if int(m_original) == 0:
        delta_ydata_new = [m_new for i in range( 1,len( ydata ) ) ]
    else:
        delta_ydata_new = [ (m_new/m_original)*(ydata[i]-ydata[i-1]) for i in range( 1,len( ydata ) ) ]
    #
    ydata_new = [ 0 for i in range( len( ydata ) ) ]
    ydata_new[0] = ydata[0]
    for i in range( 0, len( delta_ydata ) ):
        ydata_new[i+1] = ydata_new[i] + delta_ydata_new[i]
    #
    # We now recreate the new_value_list considering the fraction before and after the Initial_Year_of_Uncertainty
    fraction_list_counter = 0
    for n in range( len( time_list ) ):
        if time_list[n] >= Initial_Year_of_Uncertainty:
            new_value_list.append( ydata_new[ fraction_list_counter ] )
            fraction_list_counter += 1
        else:
            new_value_list.append( float( value_list[n] ) )
    #
    # We return the list:
    return new_value_list