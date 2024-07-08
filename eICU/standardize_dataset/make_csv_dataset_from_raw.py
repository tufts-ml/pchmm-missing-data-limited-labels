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
        default="/data/eicu",
        help='Path to the top folder of eicu dataset')
    parser.add_argument(
        '--output_dir',
        default="data/eicu",
        help='Path to preprocessed csv files for tidy time-series of ICU bedside sensors data, features_per_tstep.csv and outcomes.csv')
        
    args = parser.parse_args()    

    # get other labs
    print('Extracting all labs and vitals...')
        # extract only relevant vitals from eICU
    load_vitals_cols = ['patientunitstayid', 'observationoffset',
                        'temperature', 'sao2', 'heartrate', 
                        'respiration', 'systemicsystolic', 'systemicdiastolic']
    
    data_dir = args.dataset_raw_path
    vitals_df = pd.read_csv(os.path.join(data_dir, 'vitalPeriodic.csv.gz'),
                           usecols=load_vitals_cols)
    vitals_df['hours_in'] = vitals_df['observationoffset']/60
    vitals_df = vitals_df[(vitals_df.hours_in>=0)&(vitals_df.hours_in<=24)].copy().reset_index(drop=True)
    
    
    keep_vitals = ['temperature', 'sao2', 'heartrate', 
                   'respiration', 'systemicsystolic', 
                   'systemicdiastolic']
    
    
    vitals_df.rename(columns={'patientunitstayid': 'icustay_id'}, 
                   inplace=True)
    vitals_df = vitals_df.sort_values(by=['icustay_id', 'hours_in']).copy().reset_index(drop=True)
    
    vid_col = ['icustay_id']
    keys_df = vitals_df[vid_col].copy()
    for col in vid_col:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fps = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 
    
    nrows = len(fps)-1
    dt = 1 # hourly buckets 
    labels_list = [] 
    vals_list = [] 
    t_list = [] 
    stay_id_list = [] 
    pbar = ProgressBar()
    dt=1
    print('Transforming vitals into %s hour buckets'%dt)    
    for ii in pbar(range(nrows)): 
        curr_t = vitals_df.iloc[fps[ii]:fps[ii+1]]['hours_in'].values
        curr_vals = vitals_df.iloc[fps[ii]:fps[ii+1]][keep_vitals].values
        t_start = 0
        t_end = np.ceil(curr_t.max())
        if t_end==t_start:
            t_end=t_start+1e-5
        Tnew = np.arange(t_start, t_end, dt)
        Xnew = np.nan*np.ones((len(Tnew), curr_vals.shape[1]))
#         if len(curr_vals)==1: 
#             Xnew[-1] = curr_vals 
#         else: 

        curr_bins = np.digitize(curr_t, Tnew )-1
        for bin_id in np.unique(curr_bins): 
            keep_inds = curr_bins==bin_id
            Xnew[bin_id, :]=np.nanmean(curr_vals[keep_inds], axis=0)
        
        
#         labels_new = [vitals_df.iloc[fps[ii]]['labname']]*len(Tnew) 
        stay_id_new = [vitals_df.iloc[fps[ii]]['icustay_id']]*len(Tnew) 
        vals_list.append(Xnew) 
        t_list.append(Tnew) 
#         labels_list.append(labels_new) 
        stay_id_list.append(stay_id_new) 
        
    del vitals_df
    vitals_ids_df = pd.DataFrame({'icustay_id' : np.hstack(stay_id_list).astype(int), 
                            'hours_in':np.hstack(t_list)})
    
    vital_vals_df = pd.DataFrame(np.vstack(vals_list), columns=keep_vitals)
    vitals_df = pd.concat([vitals_ids_df, vital_vals_df], axis=1)
    del vals_list
    ## done pre-processing vitals
    
    labs_df = pd.read_csv(os.path.join(data_dir, 'lab.csv.gz'))    
    
    labs_df.rename(columns={'patientunitstayid': 'icustay_id'}, 
                   inplace=True)
    
    
    print('Getting patient info...')
    patient_df = pd.read_csv(os.path.join(data_dir, 'patient.csv.gz'))
    
    # get the length of stay of patient in days
    patient_df['los_icu'] = patient_df['unitdischargeoffset']/(60*24) 
    patient_df.rename(columns={'patientunitstayid' : 'icustay_id', 
                               'patienthealthsystemstayid':'hadm_id',
                               'uniquepid' : 'subject_id'}, 
                               inplace=True)
    
    # keep only labs from first 24 hours
    labs_df = labs_df[(labs_df.labresultoffset>=0)&(labs_df.labresultoffset<=24*60)].copy().reset_index(drop=True)
    
    
    # keep only some vitals and lab measurements
    keep_columns = [
        'BUN',
        'bedside glucose',
        'creatinine',
        'potassium',
       'sodium', 
        'chloride',  
        'magnesium',
        'Hgb',
        'Hct',
        'phosphate', 
        'WBC x 1000',
        'platelets x 1000', 
        'glucose',
        'lactate',
        'PT',
        'fibrinogen',
        'pH',
        'HCO3',
        'total protein',
        'paO2'
        'paCO2',
        'uric acid',
        'albumin',
        'calcium',
        'CRP',
        'ALT (SGPT)',
        'AST (SGOT)',
        'direct bilirubin', 
        'total bilirubin',
        'troponin - T',
        'Total CO2',
        'FiO2',
        'MCH',
        'anion gap',
        'RBC']
    
    # keep only columns that are listed
    keep_inds = labs_df['labname'].isin(keep_columns)
    labs_df = labs_df.loc[keep_inds].reset_index(drop=True)
    
    
    fp_id_cols = ['icustay_id', 'labname'] 
    labs_df = labs_df.sort_values(by=['icustay_id', 'labname']).copy().reset_index(drop=True)
    
    # convert lab collection times to hours from time of ICU admission
    labs_df['hours_in'] = labs_df['labresultoffset']/60
    
    keys_df = labs_df[fp_id_cols].copy()
    for col in fp_id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fps = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 
    
    nrows = len(fps)-1
    dt = 1 # hourly buckets 
    labels_list = [] 
    vals_list = [] 
    t_list = [] 
    stay_id_list = [] 
    pbar = ProgressBar()
    
    print('Transforming data into %s hour buckets'%dt)
    for ii in pbar(range(nrows)): 
        curr_t = labs_df.iloc[fps[ii]:fps[ii+1]]['hours_in'].values 
        curr_vals = labs_df.iloc[fps[ii]:fps[ii+1]]['labresult'].values
        t_start = 0
        t_end = np.ceil(curr_t.max())
        if t_end==t_start:
            t_end=t_start+1e-5
        Tnew = np.arange(t_start, t_end, dt)
        Xnew = np.nan*np.ones_like(Tnew)
        if len(curr_vals)==1: 
            Xnew[-1] = curr_vals 
        else: 
            curr_bins = np.digitize(curr_t, Tnew )-1
            Xnew[curr_bins] = curr_vals
#             F = interp1d(curr_t, curr_vals, kind='previous', bounds_error=False, fill_value=np.nan)   
#             Xnew = F(Tnew) 
        labels_new = [labs_df.iloc[fps[ii]]['labname']]*len(Tnew) 
        stay_id_new = [labs_df.iloc[fps[ii]]['icustay_id']]*len(Tnew) 
        vals_list.append(Xnew) 
        t_list.append(Tnew) 
        labels_list.append(labels_new) 
        stay_id_list.append(stay_id_new) 
        
    del labs_df
    labs_df = pd.DataFrame({'icustay_id' : np.hstack(stay_id_list), 
                            'hours_in':np.hstack(t_list), 
                            'label':np.hstack(labels_list),
                            'value' : np.hstack(vals_list)})

    keep_inds = ~np.isinf(labs_df['value'])
    labs_df = labs_df.loc[keep_inds]    
    
    del vals_list, labels_list
    
    
#     data_df = data_df.rename(columns={'valuenum':'value'})
    print('transforming the dataframe where we have a measurement for every time point')
    unique_labs_vitals = labs_df['label'].unique()
    for ii, lv in enumerate(unique_labs_vitals): 
        curr_df = labs_df.loc[labs_df.label==lv, ['icustay_id', 'hours_in', 'value']].rename(columns={'value' : lv}) 
        if ii==0: 
            final_labs_df = curr_df.copy() 
        else: 
            final_labs_df = pd.merge(final_labs_df, curr_df, on=['icustay_id', 'hours_in'], how='outer') 
            
    
    final_labs_df = final_labs_df.sort_values(by=['icustay_id', 'hours_in']).reset_index(drop=True)  
    
        
    # merge the labs and vitals 
    features_df = pd.merge(vitals_df, final_labs_df, on=['icustay_id', 'hours_in'], how='outer')
    features_df = features_df.sort_values(by=['icustay_id', 'hours_in']).reset_index(drop=True)
    
    
    id_cols = ['subject_id', 'hadm_id', 'icustay_id']
    features_df = pd.merge(features_df, patient_df[id_cols + ['age', 'gender']], on='icustay_id', how='left') 
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
    print('Saving data...')
    
    features_csv = os.path.join(args.output_dir, 'features_per_tstep.csv.gz')
    features_df.to_csv(features_csv, index=False, compression='gzip')
    print("Wrote features per timestep to CSV file: %s"%features_csv)
    
    outcomes_csv = os.path.join(args.output_dir, 'outcomes_per_seq.csv')
    outcomes_df.to_csv(outcomes_csv, index=False)
    print("Wrote outcomes per sequence to CSV file: %s"%outcomes_csv)
    
    