'''
Produces supervised time-series dataset for in-hospital mortality prediction task

Preconditions
-------------
EICU in-hospital mortality codes extracted on disk

Post-conditions
---------------
Will produce folder with 3 files:
* Time-varying features: features_per_tstep.csv
* Outcomes per sequence : outcomes_per_seq.csv

'''

import argparse
import numpy as np
import os
import pandas as pd
import glob
from progressbar import ProgressBar

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_raw_path',
        default="/cluster/tufts/hugheslab/datasets/eicu_v2.0/v20210518/eicu_extract/",
        help='Path to the top folder of mimic3benchmarks in-hospital mortality dataset')
    parser.add_argument(
        '--output_dir',
        default="/cluster/tufts/hugheslab/datasets/eicu_v2.0/standardized_data",
        help='Path to preprocessed csv files for tidy time-series of ICU bedside sensors data, features_per_tstep.csv and outcomes.csv')
    
    parser.add_argument(
        '--keep_first_48_hours_only',
        default='true',
        help='Path to preprocessed csv files for tidy time-series of ICU bedside sensors data, features_per_tstep.csv and outcomes.csv')
    
    # original standardized data is stored in /cluster/tufts/hugheslab/datasets/eicu_v2.0/standardized_data
    
    args = parser.parse_args()    

    # get other labs
    print('Extracting all labs and vitals...')
    labs_df = pd.read_csv('/cluster/tufts/hugheslab/datasets/eicu_v2.0/v20210518/lab.csv.gz')
    
#     labs_df = labs_df[labs_df.patientunitstayid.isin(lvm_df.icustay_id.unique())].copy().reset_index(drop=True)
    labs_df.rename(columns={'patientunitstayid': 'icustay_id'}, inplace=True)
    
    
    print('Getting patient info...')
    patient_df = pd.read_csv('/cluster/tufts/hugheslab/datasets/eicu_v2.0/v20210518/patient.csv.gz')
    
    # get the length of stay of patient in days
    patient_df['los_icu'] = patient_df['unitdischargeoffset']/(60*24) 
    patient_df.rename(columns={'patientunitstayid' : 'icustay_id', 
                               'patienthealthsystemstayid':'hadm_id'}, 
                               inplace=True)
    
    if args.keep_first_48_hours_only=='true':
        print('Keeping only first 48 hours of data for each stay...')
        keep_inds = (labs_df.labresultoffset>=0)&(labs_df.labresultoffset<=48*60)
        labs_df = labs_df.loc[keep_inds].reset_index(drop=True)
        suffix = '_first_48_hours_irregular_ts'
    else:
        suffix = '_irregular_ts'    
    
    # keep only some vitals and lab measurements
    keep_columns = [
#         'BUN',
#         'bedside glucose',
        'creatinine',
        'potassium',
        'sodium', 
#         'chloride',  
#         'magnesium',
#         'Hgb',
#         'Hct',
#         'phosphate', 
#         'WBC x 1000',
        'platelets x 1000', 
        'glucose',
#         'lactate',
#         'PT',
#         'fibrinogen',
#         'pH',
        'HCO3',
#         'total protein',
        'paO2',
#         'paCO2',
#         'uric acid',
        'albumin',
#         'calcium',
#         'CRP',
        'ALT (SGPT)',
        'AST (SGOT)',
        'direct bilirubin', 
        'total bilirubin',
        'troponin - T',
#         'Total CO2',
#         'FiO2',
#         'MCH',
#         'anion gap',
#         'RBC'
    ]
    
    
    # keep only columns that are listed
    keep_inds = labs_df['labname'].isin(keep_columns)
    labs_df = labs_df.loc[keep_inds].reset_index(drop=True)
    
    
    fp_id_cols = ['icustay_id', 'labname'] 
    labs_df = labs_df.sort_values(by=['icustay_id', 'labname']).copy().reset_index(drop=True)
    
    # convert lab collection times to hours from time of ICU admission
    labs_df['hours_in'] = labs_df['labresultoffset']/60
    labs_df['minutes_from_admission'] = labs_df['labresultoffset']   
    
#     data_df = data_df.rename(columns={'valuenum':'value'})
    print('transforming the dataframe where we have a measurement for every time point')
    unique_labs_vitals = labs_df['labname'].unique()
    for ii, lv in enumerate(unique_labs_vitals): 
        curr_df = labs_df.loc[labs_df['labname']==lv, ['icustay_id', 'minutes_from_admission', 'labresult']].rename(columns={'labresult' : lv}) 
        if ii==0: 
            final_labs_df = curr_df.copy() 
        else: 
            final_labs_df = pd.merge(final_labs_df, curr_df, on=['icustay_id', 'minutes_from_admission'], how='outer') 
            
    final_labs_df = final_labs_df.sort_values(by=['icustay_id', 'minutes_from_admission']).reset_index(drop=True)  
    
    id_cols = ['hadm_id', 'icustay_id']
    features_df = pd.merge(final_labs_df, patient_df[id_cols + ['age', 'gender']], on='icustay_id', how='left') 
    features_df['gender_is_male']=(features_df['gender'].values==1)*1
#     features_df['gender_is_unknown']=features_df['gender'].values*1
    features_df.drop(columns={'gender'}, inplace=True)
        
    
    patient_df['mort'] = (patient_df['unitdischargestatus']=='Expired')*1
    
    # get the outcomes dataframe
#     outcomes_df = data_df[id_cols + ['mort_hosp', 'mort_icu', 'los_icu', 'hospitalid', 'unittype']]
    
    outcomes_df = patient_df[id_cols + ['mort', 'los_icu', 'hospitalid', 'unittype']]
    # assume that patients with nan outcomes don't die
    outcomes_df = outcomes_df.fillna(0)
    
    
    
    # save the files to csv
    features_csv = os.path.join(args.output_dir, 'features_per_tstep%s.csv.gz'%suffix)
    features_df.to_csv(features_csv, index=False, compression='gzip')
    print("Wrote features per timestep to CSV file: %s"%features_csv)
    
    outcomes_csv = os.path.join(args.output_dir, 'outcomes_per_seq%s.csv'%suffix)
    outcomes_df.to_csv(outcomes_csv, index=False)
    print("Wrote outcomes per sequence to CSV file: %s"%outcomes_csv)