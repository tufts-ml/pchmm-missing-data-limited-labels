# Reproducing eICU mortality prediction and LOS ordinal regression experiments


## Workflow

### Downloading the vitals, labs and ICU mortality outcomes
Download the eICU data from Physionet[eICU data from Physionet](https://physionet.org/content/eicu-crd/2.0/)

You should have these 3 main files after download : 

 - vitalPeriodic.csv.gz (all the vitals with timestamps)
 - lab.csv.gz (all the labs with timestamps)
 - patient.csv.gz (patient admission and discharge information)

Make sure to save these files in the "data/eicu" folder

### Pre-process the chart events to extract and downsample relevant vitals and lower outcome rate
Run the "standadize_data/make_csv_dataset_from_raw.py" script to generate 3 files :

 - features_per_tstep.csv.gz (hourly spaced labs and vitals with timestamps)
 - outcomes_per_seq.csv (LOS and mortality outcomes)

`>> python standadize_data/make_csv_dataset_from_raw_for_mortality_and_los_prediction.py --dataset_raw_path "data/eicu" --output_dir "data/eicu"`


### Mask labels out to create versions of the data with 1.2%, 3.7%, 11.1%, 33.3% and 100% of the labels available
Run all the cells in the "notebooks/create_ssl_dataset_for_mortality_prediction.ipynb" and "notebooks/create_ssl_dataset_for_mortality_prediction.ipynb" notebooks to generate the train/valid/test splits with 1.2%, 3.7%, 11.1%, 33.3% and 100% of the labels for the mortality prediction and LOS ordinal regression tasks respectively. 

Note : The train valid test files will be saved in the "data/eicu/ordinal_los_prediction/" and "data/eicu/mortality_prediction" folders

### Train and Evaluate PC-HMM and other supervised and SSL baselines (GRU-D/BRITS/MixMatch/FixMatch)
Run the training snakemake scripts to train the baselines and the PC-HMM models. Replace the {task} with either "mortality_prediction" or "los_prediction" depending on the prediction task. Each model uses the corresponding {model}.json file for hyper parameters. The trained models are saved in the "results" folder.

    >> snakemake --cores 1 --snakefile {task}/train_semi_supervised_pchmm.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile {task}/train_semi_supervised_BRITS.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile {task}/train_gru_d_semi_supervised.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile {task}/train_semi_supervised_FixMatch.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile {task}/train_semi_supervised_MixMatch.smk train_and_evaluate_classifier_many_hyperparams


For training the ordinal regression models, run the following snakemake files

    >> snakemake --cores 1 --snakefile los_prediction/train_semi_supervised_pchmm_for_los_ordinal_regression.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile los_prediction/train_gru_d_semi_supervised_for_los_ordinal_regression.smk train_and_evaluate_classifier_many_hyperparams

Use the "notebooks/evaluate_performance_{task}" to evaluate the performance of each of the models
 

