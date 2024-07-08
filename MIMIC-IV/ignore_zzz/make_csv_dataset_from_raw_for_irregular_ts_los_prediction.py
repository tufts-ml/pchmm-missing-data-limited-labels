'''
Produces supervised time-series dataset for in-hospital mortality prediction task

Preconditions
-------------
mimic-iv in-icu features extracted on disk

Post-conditions
---------------
Will produce folder with 3 files:
* Time-varying features: data_per_tstamp.csv
* Per-sequence features: data_per_seq.csv

'''


import argparse
import numpy as np
import os
import pandas as pd
import glob
from scipy.interpolate import interp1d   
from progressbar import ProgressBar

def get_hours_from_adm(example_time): 
    days = example_time.split(' ')[0] 
    hrs, minutes, seconds = example_time.split('days ')[-1].split(':')  
    hours_from_admission = int(days)*24 + int(hrs) + int(minutes)/60 + float(seconds)/3600 
    return hours_from_admission 

def get_minutes_from_adm(example_time): 
    days = example_time.split(' ')[0] 
    hrs, minutes, seconds = example_time.split('days ')[-1].split(':')  
    mins_from_admission = int(days)*24*60 + int(hrs)*60 + int(minutes) + float(seconds)/60 
    return mins_from_admission 


def calc_time_between_measurements(df, id_cols, feature_cols, time_col):
    keys_df = df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    
    if len(id_cols)>1:
        fp = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]])
    else:
        fp = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0)), keys_df.shape[0]])
    n_stays = len(fp)-1
    
    timestamp_arr = np.asarray(df[time_col].values.copy(), dtype=np.float64)
    vitals_arr = df[feature_cols].values
    
    
    tdiff_list = [[] for i in range(len(feature_cols))]
    for stay in range(n_stays):
        fp_start = fp[stay]
        fp_end = fp[stay+1]
        curr_vitals_arr = vitals_arr[fp_start:fp_end].copy()
        curr_timestamp_arr = timestamp_arr[fp_start:fp_end]
        for vital_ind, vital in enumerate(feature_cols):
            curr_vital_arr = curr_vitals_arr[:, vital_ind]
            non_nan_inds = ~np.isnan(curr_vital_arr)
            non_nan_t = curr_timestamp_arr[non_nan_inds]
            curr_vital_tdiff = np.diff(non_nan_t[:, 0])
            tdiff_list[vital_ind].extend(curr_vital_tdiff)
    
    vitals_tdiff_df = pd.DataFrame()
    for vital_ind, vital in enumerate(feature_cols):
        if len(tdiff_list[vital_ind]):
            vitals_tdiff_df.loc[vital, 'tdiff_min'] = min(tdiff_list[vital_ind])
            vitals_tdiff_df.loc[vital, 'tdiff_5%'] = np.percentile(tdiff_list[vital_ind], 5)
            vitals_tdiff_df.loc[vital, 'tdiff_median'] = np.median(tdiff_list[vital_ind])
            vitals_tdiff_df.loc[vital, 'tdiff_95%'] = np.percentile(tdiff_list[vital_ind], 95)
            vitals_tdiff_df.loc[vital, 'tdiff_max'] = max(tdiff_list[vital_ind])
        else:
            vitals_tdiff_df.loc[vital, :] = 'once per stay'
        
    
    return vitals_tdiff_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_raw_path',
        default=None,
        help='Path to the top folder of mimic3benchmarks in-hospital mortality dataset')
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Path to preprocessed csv files for tidy time-series of ICU bedside sensors data, features_per_tstep.csv and outcomes.csv')
    parser.add_argument(
        '--keep_first_24_hours_only',
        default='true',
        help='Path to preprocessed csv files for tidy time-series of ICU bedside sensors data, features_per_tstep.csv and outcomes.csv')
    args = parser.parse_args()
    
    
    # get the raw data
    print('Loading raw data...')
    root_dir = '/cluster/tufts/hugheslab/prath01/datasets/MIMIC-IV/physionet.org/files/mimiciv/MIMIC-IV-Data-Pipeline'
#     root_dir = args.dataset_raw_path
    data_dir = os.path.join(root_dir, 'data', 'features')
    data_df = pd.read_csv(os.path.join(data_dir, 'preproc_chart.csv.gz'))
    
    
    # merge with the items file to get the labs and vitals names
    print('Getting chartevent names...')
    items_csv = os.path.join(root_dir, 'mimic-iv-2.0', 'icu', 'd_items.csv.gz')
    items_df = pd.read_csv(items_csv)
    data_df = pd.merge(items_df[['itemid', 'label']], data_df, on='itemid')
    
    print('Converting the chartevent times to minutes from admission...')
    data_df['hours_from_admission'] = data_df['event_time_from_admit'].apply(get_hours_from_adm)
    data_df['minutes_from_admission'] = data_df['event_time_from_admit'].apply(get_minutes_from_adm)  
    
    
    # keep only some vitals and lab measurements
    keep_columns = ['Heart Rate', 
                    'Respiratory Rate', 
                    'O2 saturation pulseoxymetry',
       'Non Invasive Blood Pressure systolic',
       'Non Invasive Blood Pressure diastolic',
        'Temperature Fahrenheit',
        'Height (cm)',
       'Respiratory Rate (Total)', 
       'Potassium (serum)',
       'Sodium (serum)', 
        'Chloride (serum)', 
        'Hematocrit (serum)',
       'Hemoglobin', 
        'Creatinine (serum)', 
        'Glucose (serum)', 
        'Magnesium', 
       'Phosphorous', 
        'Platelet Count', 
        'Glucose (whole blood)',
        'Daily Weight', 
        'Absolute Neutrophil Count',
        'Prothrombin time',
        'Fibrinogen',
        'PH (Arterial)',
        'PH (Venous)',
        'HCO3 (serum)',
        'Arterial O2 pressure',
        'Arterial CO2 Pressure',
        'Lactic Acid',
        'Albumin',
        'Calcium non-ionized',
        'C Reactive Protein (CRP)',
        'ALT',
        'AST',
        'Direct Bilirubin', 
        'Total Bilirubin',
        'Troponin-T',
        'Venous CO2 Pressure']
    
    # keep only the vitals and labs of interest and keep only first 24 hours of data
    print('Keeping only chosen vitals and labs')
    keep_inds = data_df['label'].isin(keep_columns)

    data_df = data_df.loc[keep_inds]
    data_df = data_df.drop(columns={'itemid'}) 
    
    # keep only measurements taken after admission
    keep_inds = data_df.hours_from_admission>=0  
    data_df = data_df.loc[keep_inds]
    
    outcomes_df = pd.read_csv('/cluster/tufts/hugheslab/prath01/datasets/MIMIC-IV/physionet.org/files/mimiciv/MIMIC-IV-Data-Pipeline/data/cohort/cohort_icu_mortality.csv.gz')
    
    data_df = pd.merge(data_df, outcomes_df[['stay_id', 'intime']], on=['stay_id'], how='inner') 
    data_df['timestamp'] = pd.to_datetime(data_df['intime'])+pd.to_timedelta(data_df['event_time_from_admit']) 
    
    if args.keep_first_48_hours_only=='true':
        print('Keeping only first 48 hours of data for each stay...')
        keep_inds = data_df['hours_from_admission']<48.0
        data_df = data_df.loc[keep_inds].reset_index(drop=True)
        suffix = '_first_48_hours_irregular_ts'
    else:
        suffix = '_irregular_ts'
    
    
    id_cols = ['stay_id', 'label'] 
    keys_df = data_df[id_cols].copy()
    for col in id_cols:
        if not pd.api.types.is_numeric_dtype(keys_df[col].dtype):
            keys_df[col] = keys_df[col].astype('category')
            keys_df[col] = keys_df[col].cat.codes
    fps = np.hstack([0, 1 + np.flatnonzero(np.diff(keys_df.values, axis=0).any(axis=1)), keys_df.shape[0]]) 
        
    data_df = data_df.rename(columns={'valuenum':'value'}).copy()
    
    
#     data_df = data_df.rename(columns={'valuenum':'value'})
    print('transforming the dataframe where we have a measurement for every time point')
    unique_labs_vitals = data_df['label'].unique()
    for ii, lv in enumerate(unique_labs_vitals): 
        curr_df = data_df.loc[data_df.label==lv, ['stay_id', 'minutes_from_admission', 'value']].rename(columns={'value' : lv}) 
        if ii==0: 
            final_df = curr_df.copy() 
        else: 
            final_df = pd.merge(final_df, curr_df, on=['stay_id', 'minutes_from_admission'], how='outer') 
            
    
    final_df = final_df.sort_values(by=['stay_id', 'minutes_from_admission']).reset_index(drop=True)  
    
    
    del data_df
    # calculage the BMI
    print('Calculating BMI...')
    non_nan_inds = ~(final_df['Height (cm)'].isna())|~(final_df['Daily Weight'].isna()) 
    height_weight_df = final_df[['stay_id', 'Height (cm)', 'Daily Weight']].copy() 
    agg_hw_df = height_weight_df.groupby('stay_id').mean().reset_index()
    agg_hw_df['bmi'] = agg_hw_df['Daily Weight']/((0.01*agg_hw_df['Height (cm)'])**2)
    agg_hw_df.loc[np.isinf(agg_hw_df['bmi']), 'bmi']=np.nan
    final_df = pd.merge(final_df, agg_hw_df[['stay_id', 'bmi']], on='stay_id', how='left') 
    final_df.loc[~non_nan_inds, 'bmi']=np.nan
    
    
    # get the frequency of measurements
#     feature_cols = keep_columns + ['bmi']
    time_col = ['minutes_from_admission']
    id_cols = ['stay_id']
    
    
    # ffill height measurements for frequent for tdiff calculation
    h_df = final_df[['stay_id', 'minutes_from_admission', 'Height (cm)']] 
    h_df = h_df.groupby(id_cols).apply(lambda x : x.fillna(method='pad')).copy()
    final_df.loc[:, 'Height (cm)']=h_df['Height (cm)'].copy()
        
    # convert body temperature to celsius
    final_df['Temperature Fahrenheit'] = (final_df['Temperature Fahrenheit'] - 32)*(5/9)    
    
    
    # get the outcomes
    print('Loading the patient stay outcomes from admissions file...')
    
    # calculate length of stay
    print('Calculating length of stay for all patients...')
    td = pd.to_datetime(outcomes_df['outtime']) - pd.to_datetime(outcomes_df['intime'])  
    outcomes_df['length_of_stay_in_hours'] = [ii.total_seconds()/3600 for ii in td] 
    
    # minor pre-processing on outcomes file
    outcomes_df['is_gender_male']=(outcomes_df['gender']=='M')*1
    outcomes_df['is_gender_unknown']=(outcomes_df['gender'].isna())*1
    outcomes_df.rename(columns={'label':'in_icu_mortality', 'intime':'admission_timestamp'}, inplace=True)
    final_df = pd.merge(final_df, outcomes_df[['hadm_id', 'subject_id', 
                                                                   'stay_id', 
                                                                   'in_icu_mortality', 
                                                                   'length_of_stay_in_hours', 
                                                                   'admission_timestamp',
                                                                   'is_gender_male',
                                                                   'is_gender_unknown',
                                                                   'Age']], on=['stay_id'])
    
    final_df['timestamp'] = pd.to_datetime(final_df['admission_timestamp'])+pd.to_timedelta(final_df.minutes_from_admission, unit='minutes') 
    
    
    id_cols = ['subject_id', 'hadm_id', 'stay_id']
    outcome_cols = ['in_icu_mortality', 'length_of_stay_in_hours']
    time_col = ['minutes_from_admission', 'timestamp']
    demographics_cols = ['Age', 'is_gender_male', 'is_gender_unknown']
    adm_col = ['admission_timestamp']
    
    feature_cols = keep_columns
    
    print('Creating tidy features_per_tstep and outcomes_per_seq tables')
    features_df = final_df[id_cols+time_col+feature_cols+adm_col].copy()
    outcomes_df = final_df[id_cols+outcome_cols+adm_col].copy()
    outcomes_df = outcomes_df.drop_duplicates(subset=id_cols).reset_index(drop=True)
    
    demographics_df = final_df[id_cols+demographics_cols+adm_col].copy()
    demographics_df = demographics_df.drop_duplicates(subset=id_cols).reset_index(drop=True) 
#     demographics_df['is_gender_unknown']=(demographics_df.is_gender_male.isna())*1 
    
    final_demographics_df = pd.merge(outcomes_df[id_cols], demographics_df, on=id_cols).reset_index(drop=True)
    final_features_df = pd.merge(outcomes_df[id_cols], features_df, on=id_cols).reset_index(drop=True)
        
    save_dir = '/cluster/tufts/hugheslab/datasets/MIMIC-IV/'
    features_csv = os.path.join(save_dir, 'features_per_tstep%s_los_prediction.csv.gz'%suffix)
    outcomes_csv = os.path.join(save_dir, 'outcomes_per_seq%s_los_prediction.csv'%suffix)
    dem_csv = os.path.join(save_dir, 'demographics%s_los_prediction.csv.gz'%suffix)
    
    print('Saving features per timestep to :\n%s'%features_csv)
    print('Saving outcomes per sequence to :\n%s'%outcomes_csv)
    print('Saving demographics per admission to :\n%s'%dem_csv)
    final_features_df.to_csv(features_csv, index=False, compression='gzip')
    final_demographics_df.to_csv(dem_csv, index=False, compression='gzip')
    outcomes_df.to_csv(outcomes_csv, index=False)
        
    '''
    # create spec-sheet
    features_specs_df = pd.DataFrame(columns=['ColumnName', 'Role', 'Type', 'Minimum', 
                                              'Maximum', 'Units', 'Description', 'Required'])
    features_specs_df.loc[:, 'ColumnName']=features_df.columns 
    
    outcome_specs_df = pd.DataFrame(columns=['ColumnName', 'Role', 'Type', 'Minimum', 
                                             'Maximum', 'Units', 'Description', 'Required']) 
    outcome_specs_df.loc[:, 'ColumnName']=outcomes_df.columns
    '''