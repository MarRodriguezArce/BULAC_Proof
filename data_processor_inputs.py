# AUTHOR: LUIS FERNANDO

import pandas as pd
from copy import deepcopy
import os
import sys
import time
import pickle

# Recording initial time of execution
start_1 = time.time()

# 0.a) Open the managing files:
df_mngr_overall = "data_manager_overall.xlsx"
df_country_2_reg = pd.read_excel(df_mngr_overall, sheet_name="country_2_reg")
df_mngr = "data_manager_olade_iea_other.xlsx"

# 0.b) Open the input files:
dir_caps = './data_input_processing/olade_capacities'
caps_filename = dir_caps + '/' + 'all_countries_capacity.xlsx'

# 0.c) A dictionary is required to align the gathered thermal power plant data
### dict_thermal_equiv = {
###    'Fuel oil':['Fuel Oil/Bunker', 'Petróleo/Crudo']}

dict_thermal_equiv = {
    'Fuel oil':['Fuel Oil/Bunker'],
    'Crude':['Petróleo/Crudo'],
    'Diesel':['Diesel'],
    'Natural Gas':['Gas Natural'],
    'Coal':['Carbón']}

dir_thermal_caps = './data_input_processing/thermal_capacities'
thermal_caps_file = dir_thermal_caps + '/' + 'thermal_cap_distribution.xlsx'
df_therm_cap = pd.read_excel(thermal_caps_file, sheet_name="2021")

hyear_vector = [
    2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
hyear_vector_sh = [2018, 2019, 2020, 2021]

# Control which elements you want to process:
process_energy_balances = True
process_electrical_capacities = True
process_electrical_capacities_raw = True

# Control how you want to process the data:
thermal_share_manual_mode = False
thermal_share_automatic = True

###############################################################################
if process_energy_balances is True:
    # 1) First, let's process the energy balance databases
    print('1) Processing energy balance data')
    dir_balances = './data_input_processing/energy_balances'

    # Objective: create a nested dictionary of region/country/variable/set/year
    # Crucially, variable <=> balance rows && set <=> balance cols

    dict_db = {}
    wide_source = []
    wide_file = []
    wide_region = []
    wide_country = []
    wide_variable_main = []
    wide_variable_disag = []
    wide_set = []
    wide_year = []
    wide_value = []

    # Open the nesting and labeling sheets from control:
    df_control_EB_nest = pd.read_excel(df_mngr, sheet_name="EB_nest")
    df_control_EB_labeling = pd.read_excel(df_mngr, sheet_name="EB_labeling")
    list_EB_nest_main = df_control_EB_nest['main'].tolist()
    list_EB_nest_disag = df_control_EB_nest['disag'].tolist()
    df_EB_label_name = df_control_EB_labeling['name'].tolist()
    df_EB_labels = df_control_EB_labeling['label'].tolist() 

    list_regions = list(set(df_country_2_reg['Region'].tolist()))
    list_regions.sort()

    all_lac_country_list = list(set(df_country_2_reg['Country'].tolist()))
    country_2_region = {}
    for index, row in df_country_2_reg.iterrows():
        a_country = row['Country']
        a_region = row['Region']
        country_2_region[a_country] = a_region

    list_of_balances = os.listdir(dir_balances)
    list_of_balances_filter = [i for i in list_of_balances if '$' not in i]
    """
    Note 1: the name of the balances must be equal to the names of countries
    in the "df_country_2_reg" dataframe.
    """

    list_of_countries = [
        a_con.replace('.xlsx', '') for a_con in list_of_balances]

    for reg in list_regions:  # iterate across the list of regions
        dict_db.update({reg:{}})

    for con in list_of_countries:  # iterate across the list of countries
        unique_main = []
        con_idx = list_of_countries.index(con)

        print('Processing: ' + con)

        eb_data = dir_balances + '/' + list_of_balances[con_idx]
        balance_file = pd.ExcelFile(eb_data)
        yearly_sheet = balance_file.sheet_names

        reg = country_2_region[con]

        dict_db[reg].update({con:{}})

        for y in range(len(yearly_sheet)):
            year = yearly_sheet[y].split(' - ')[0]

            balance_data_raw = pd.read_excel(eb_data,
                                             sheet_name=yearly_sheet[y])

            # this is custom; make sure it always applies:
            balance_data = balance_data_raw.drop([0,1,2,4,32,33,34])

            balance_data.fillna(0, inplace=True)
            balance_data.columns = balance_data.iloc[0]
            balance_data.drop([3], inplace=True)
            balance_data = balance_data.reset_index(drop=True)

            # Now we have a couple of key tasks:
            # First, extract the number of sets and find easy labels.
            # Second, nest the data appropiately.
            data_cols_all = balance_data.columns.tolist()  # copy this once
            data_cols = data_cols_all[1:]
            data_rows = balance_data[0].tolist()

            # The next thing to do is store the data nested with the logic:
            #   main row, disag row, col

            # Because of nesting logic,
            #create the dictionary vairable and sets at y == 0
            if y == 0:  # nest
                for a_main_row in list_EB_nest_main:
                    if a_main_row not in unique_main:
                        unique_main.append(a_main_row)

                        # Call main row below:
                        conv_idx = df_EB_label_name.index(a_main_row)
                        main_row_print = df_EB_labels[conv_idx]
                        dict_db[reg][con].update({main_row_print:{}})

                        # Call disaggregated row below:
                        this_index = list_EB_nest_main.index(a_main_row)
                        disag_row_raw = list_EB_nest_disag[this_index]
                        try:
                            conv_idx = df_EB_label_name.index(disag_row_raw)
                            disag_row = df_EB_labels[conv_idx]
                        except Exception:
                            disag_row = disag_row_raw

                        dict_db[reg][con][
                            main_row_print].update({disag_row:{}})

                        for col in data_cols:  # be sure to add the fuels:
                            conv_idx = df_EB_label_name.index(col)
                            col_print = df_EB_labels[conv_idx]

                            dict_db[reg][con][main_row_print][
                                disag_row].update({col_print:{}})

                    else:
                        # Call more disaggregated rows below:
                        unique_main_indices_raw = [
                            i for i in range(len(list_EB_nest_main))
                            if list_EB_nest_main[i] == a_main_row]
                        unique_main_indices = unique_main_indices_raw
                        for a_disag_idx in unique_main_indices:
                            disag_row_raw = list_EB_nest_disag[a_disag_idx]

                            try:
                                conv_idx = \
                                    df_EB_label_name.index(disag_row_raw)
                                disag_row = df_EB_labels[conv_idx]
                            except Exception:
                                disag_row = disag_row_raw

                            dict_db[reg][con][
                                main_row_print].update({disag_row:{}})

                            for col in data_cols:  # add the fuels:
                                conv_idx = df_EB_label_name.index(col)
                                col_print = df_EB_labels[conv_idx]

                                dict_db[reg][con][main_row_print][
                                    disag_row].update({col_print:{}})

            # Now, fill the data:
            list_EB_nest_main_unique = list(set(list_EB_nest_main))
            for a_main_row in list_EB_nest_main_unique:
                conv_idx = df_EB_label_name.index(a_main_row)
                main_row_print = df_EB_labels[conv_idx]

                unique_main_indices_raw = [
                    i for i in range(len(list_EB_nest_main))
                    if list_EB_nest_main[i] == a_main_row]

                unique_main_indices = unique_main_indices_raw
                for a_disag_idx in unique_main_indices:
                    disag_row_raw = list_EB_nest_disag[a_disag_idx]
                    try:
                        conv_idx = df_EB_label_name.index(disag_row_raw)
                        disag_row = df_EB_labels[conv_idx]
                    except Exception:
                        disag_row = disag_row_raw

                    for col in data_cols:
                        # Here we must extract the value:
                        if disag_row_raw != 'none':
                            mask = (balance_data[0] == disag_row_raw)
                        else:
                            mask = (balance_data[0] == a_main_row)
                        data_series = balance_data[mask]
                        if disag_row == 'Exports':
                            add_value = data_series[col].iloc[0]*-1
                        else:
                            add_value = data_series[col].iloc[0]

                        # Here we must update the dictionary
                        conv_idx = df_EB_label_name.index(col)
                        col_print = df_EB_labels[conv_idx]
                        dict_db[reg][con][main_row_print][disag_row][
                            col_print].update({year:add_value})

                        # Append the list to print in wide format:
                        wide_source.append('OLADE')
                        wide_file.append('Energy Balances')
                        wide_region.append(reg)
                        wide_country.append(con)
                        wide_variable_main.append(main_row_print)
                        wide_variable_disag.append(disag_row)
                        wide_set.append(col_print)
                        wide_year.append(year)
                        wide_value.append(add_value)

    print_output_energy_balance = True
    if print_output_energy_balance is True:
        # Store the data for visualization:
        dict_wide_database = {  'Source':deepcopy(wide_source),
                                'File':deepcopy(wide_file),
                                'Region':deepcopy(wide_region),
                                'Country':deepcopy(wide_country),
                                'Variable_Main':deepcopy(wide_variable_main),
                                'Variable_Disag':deepcopy(wide_variable_disag),
                                'Set':deepcopy(wide_set),
                                'Year':deepcopy(wide_year),
                                'Value':deepcopy(wide_value)}
    
        df_EB_name = 'energy_balances_wide.csv'
        df_EB = pd.DataFrame.from_dict(dict_wide_database)
        df_EB.to_csv(df_EB_name, index=None, header=True)

        # We are done with energy balances;
        # we can close everything and move to the next variable.

print('The first stage is complete.\n')
# sys.exit()

###############################################################################
if process_electrical_capacities is True:
    # 2) Second, let's process the electrical capacity databases
    print('2) Processing electrical capacity data')

    dict_db_cap = {}

    # It is also useful to have the geographic distribution df handy:
    df_geo_dist = pd.read_excel(df_mngr, sheet_name="geo_dist")

    df_cntrl_cap_label_raw = pd.read_excel(
        df_mngr, sheet_name="Cap_labeling_CON")
    df_ccl_countries = df_cntrl_cap_label_raw['countries'].tolist()

    caps_file = pd.ExcelFile(caps_filename)
    country_sheet = caps_file.sheet_names
    country_list_get = df_geo_dist['Country_Spanish_IC'].tolist()

    country_list_region = df_geo_dist['Region'].tolist()
    country_list_region_id = df_geo_dist['Reg_ID'].tolist()

    unique_reg = []
    country_list_set = []
    for c in country_sheet:
        con_spa = c.split('.')[-1]

        print('Processing: ' + con_spa)

        if con_spa in country_list_get:
            country_idx = country_list_get.index(con_spa)
            con = df_geo_dist['Country'].tolist()[country_idx]

            mask_0 = (df_therm_cap['Country'] == con)
            df_therm_cap_con = df_therm_cap.loc[mask_0]
            len_therm_cap = len(df_therm_cap_con.index.tolist())
            therm_cap_sum = df_therm_cap_con['Capacity [MW]'].sum()

            # We will now subselect the table according to the country:
            unique_con_list = list(set(df_ccl_countries))
            internal_c_idx = 0
            internal_c_idx_select = 0
            for internal_c_group in unique_con_list:
                unique_country_list = internal_c_group.split(' ; ')
                if con in unique_country_list:
                    internal_c_idx_select = deepcopy(internal_c_idx)
                internal_c_idx += 1
            country_selector = unique_con_list[internal_c_idx_select]
            country_selector_mask = \
                (df_cntrl_cap_label_raw['countries'] == country_selector)
            df_cntrl_cap_label = \
                df_cntrl_cap_label_raw.loc[country_selector_mask]

            # Call the regions below:
            reg = country_list_region[country_idx]
            reg_id = country_list_region_id[country_idx]
            regstr_prev = str(reg_id) + '_' + reg

            regstr = country_2_region[con]

            # print('check regions')
            # sys.exit()
            
            if regstr not in unique_reg:
                unique_reg.append(regstr)
                dict_db_cap.update({regstr:{}})

            # These lists are used below:
            df_ccl_name = df_cntrl_cap_label['name'].tolist()
            df_ccl_label = df_cntrl_cap_label['label'].tolist()
            df_ccl_fuel = df_cntrl_cap_label['fuel'].tolist()
            df_ccl_cap_share = df_cntrl_cap_label['cap_share_2019'].tolist()
            df_ccl_suffix = df_cntrl_cap_label['suffix'].tolist()

            country_list_set.append(con)
            dict_db_cap[regstr].update({con:{}})

            # Now, open the dataframe:
            df_cap_raw = pd.read_excel(caps_filename, sheet_name=c)
            df_cap = df_cap_raw.drop([0,1,2,12,13,14,15])
            df_cap.columns = df_cap.iloc[0]
            df_cap.drop([3], inplace=True)
            df_cap = df_cap.reset_index(drop=True)
            df_cap.fillna(0, inplace=True)

            old_sets = df_cap['Descripción'].tolist()
            new_sets = []
            new_sets_cap = []

            for aset in old_sets:
                setidx_list = [
                    i for i in range(len(df_ccl_name)) if
                    df_ccl_name[i] == aset]

                for setidx in setidx_list:  # This is important for modelling
                    # Naming sets:
                    grab_label = df_ccl_label[setidx]
                    grab_fuel = df_ccl_fuel[setidx]

                    if grab_label != 'as_fuel':
                        new_set = df_ccl_suffix[setidx] + '_' + grab_label + \
                            '_' + df_ccl_fuel[setidx]
                    else:
                        new_set = \
                            df_ccl_suffix[setidx] + '_' + df_ccl_fuel[setidx] 
                    new_sets.append(new_set)

                    mask = (df_cap['Descripción'] == aset)
                    df_cap_grab = df_cap[mask]

                    dict_db_cap[regstr][con].update({new_set:{}})

                    for hy in hyear_vector_sh:
                        grab_cap = df_cap_grab[float(hy)].iloc[0]

                        # Here we can modify the capacity percentages:
                        if grab_label == 'Thermal':
                            grab_eq_fuels = dict_thermal_equiv[grab_fuel]
                            mask_2 = (df_therm_cap_con["Fuel"].isin(grab_eq_fuels))
                            therm_cap_fuel = df_therm_cap_con.loc[
                                mask_2]['Capacity [MW]'].sum()

                            if thermal_share_manual_mode:
                                ### print('Happen 1')
                                new_sets_cap.append(
                                    grab_cap*df_ccl_cap_share[setidx]/100)

                            if thermal_share_automatic and len_therm_cap > 0 and therm_cap_fuel >= 0:
                                ### print('Happen 2', grab_eq_fuels)
                                if therm_cap_sum == 0:
                                    print('Somebody has a zero sum of thermal', con_spa)
                                    therm_cap_sh = 0
                                    ### sys.exit()
                                else:
                                    therm_cap_sh = therm_cap_fuel/therm_cap_sum
                                new_sets_cap.append(grab_cap*therm_cap_sh)


                            else:
                                ### print('Happen 3')
                                new_sets_cap.append(
                                    grab_cap*df_ccl_cap_share[setidx]/100)

                            ### if 'Haiti' in con_spa and 'sel' in grab_eq_fuels[0]:
                            ###    print('check')
                            ###    sys.exit()

                            # print('redistribute percentages')
                            #  sys.exit()

                        else:
                            new_sets_cap.append(
                                grab_cap*df_ccl_cap_share[setidx]/100)

                        dict_db_cap[regstr][con][new_set].update(
                            {str(hy):new_sets_cap[-1]})

                        # Here we must append the list to print in wide format:
                        wide_source.append('OLADE')
                        wide_file.append('Electrical Capacity')
                        wide_region.append(regstr)
                        wide_country.append(con)
                        wide_variable_main.append('Electrical Capacity')
                        wide_variable_disag.append('Electrical Capacity')
                        wide_set.append(new_set)
                        wide_year.append(hy)
                        wide_value.append(new_sets_cap[-1])

    # We are done with electrical capacities;
    # we can close everything and move to the next variable.

print('The second stage is complete.\n')
# sys.exit()

###############################################################################
if process_electrical_capacities_raw is True:
    # 3) Third, let's process the electrical capacity databases (broadly)
    print('3) Process OLADE electrical capacity, without thermal distribution')

    # It is also useful to have the geographic distribution df handy:
    df_geo_dist = pd.read_excel(df_mngr, sheet_name="geo_dist")

    df_cntrl_cap_label = pd.read_excel(df_mngr, sheet_name="Cap_labeling")
    df_ccl_name = df_cntrl_cap_label['name'].tolist()
    df_ccl_name_eng = df_cntrl_cap_label['name_eng'].tolist()
    df_ccl_label = df_cntrl_cap_label['label'].tolist()
    df_ccl_fuel = df_cntrl_cap_label['fuel'].tolist()
    df_ccl_cap_share = df_cntrl_cap_label['cap_share_2019'].tolist()
    df_ccl_suffix = df_cntrl_cap_label['suffix'].tolist()

    caps_file = pd.ExcelFile(caps_filename)
    country_sheet = caps_file.sheet_names
    country_list_get = df_geo_dist['Country_Spanish_IC'].tolist()

    country_list_region = df_geo_dist['Region'].tolist()
    country_list_region_id = df_geo_dist['Reg_ID'].tolist()

    unique_reg = []
    country_list_set = []
    for c in country_sheet:
        con_spa = c.split('.')[-1]

        print('Processing: ' + con_spa)

        if con_spa in country_list_get:
            country_idx = country_list_get.index(con_spa)
            con = df_geo_dist['Country'].tolist()[country_idx]

            # Call the regions below:
            reg = country_list_region[country_idx]
            reg_id = country_list_region_id[country_idx]
            regstr = str(reg_id) + '_' + reg
            if regstr not in unique_reg:
                unique_reg.append(regstr)
  
            country_list_set.append(con)

            # Now, open the dataframe:
            df_cap_raw = pd.read_excel(caps_filename, sheet_name=c)
            df_cap = df_cap_raw.drop([0,1,2,12,13,14,15])
            df_cap.columns = df_cap.iloc[0]
            df_cap.drop([3], inplace=True)
            df_cap = df_cap.reset_index(drop=True)
            df_cap.fillna(0, inplace=True)

            old_sets = df_cap['Descripción'].tolist()
            new_sets_cap = []

            for hy in hyear_vector_sh:
                for aset in old_sets:
                    name_set = df_ccl_name_eng[df_ccl_name.index(aset)]

                    mask = (df_cap['Descripción'] == aset)
                    df_cap_grab = df_cap[mask]
                    grab_cap = df_cap_grab[float(hy)].iloc[0]

                    new_sets_cap.append(grab_cap)

                    # Here we must append the list to print in wide format:
                    wide_source.append('OLADE')
                    wide_file.append('Electrical Capacity without separation')
                    wide_region.append(regstr)
                    wide_country.append(con)
                    wide_variable_main.append('Electrical Capacity')
                    wide_variable_disag.append('Electrical Capacity')
                    wide_set.append(name_set)
                    wide_year.append(hy)
                    wide_value.append(new_sets_cap[-1])

    # We are done with electrical capacities;
    # we can close everything and move to the next variable.

print('The third stage is complete.\n')
# sys.exit()

###############################################################################
# Final) Print final dictionary:
print_final_pickle = True
if print_final_pickle is True:
    out_dict_db = {'EB': dict_db, 'Cap':dict_db_cap}
    fn_out_dict_db = 'dict_db'
    with open(fn_out_dict_db + '.pickle', 'wb') as out_dict:
        pickle.dump(out_dict_db, out_dict, protocol=pickle.HIGHEST_PROTOCOL)
    out_dict.close()

# Final) Print the final csv table:
print_output_all = True
if print_output_all is True:
    # Store the data for visualization:
    dict_wide_database = {  'Source':wide_source,
                            'File':wide_file,
                            'Region':wide_region,
                            'Country':wide_country,
                            'Variable_Main':wide_variable_main,
                            'Variable_Disag':wide_variable_disag,
                            'Set':wide_set,
                            'Year':wide_year,
                            'Value':wide_value,}

    df_all_name = 'input_database.csv'
    df_all = pd.DataFrame.from_dict(dict_wide_database)
    df_all.to_csv(df_all_name, index=None, header=True)

# Recording final time of execution:
end_f = time.time()
te_f = -start_1 + end_f  # te: time_elapsed
print(str(te_f) + ' seconds /', str(te_f/60) + ' minutes')
print('*: This automatic analysis is finished.')

