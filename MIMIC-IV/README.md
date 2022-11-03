# Reproducing MIMIC-IV in-ICU mortality prediction experiment


## Workflow

### Downloading the chart events and ICU mortality outcomes
Clone the [MIMIC-IV Data-Pipeline](https://github.com/healthylaife/mimic-iv-data-pipeline) repo and follow the instructions to extract the chart events and ICU mortality outomes. We used the "data extraction" in the [mainPipeline.ipynb](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/mainPipeline.ipynb) notebook to extract the chart events without any disease filters.

Running the notebook generates 3 files : 

 - preproc_chart.csv.gz (all the vitals with timestamps)
 - d_items.csv.gz (file linking the vital IDs to the vital names)
 - cohort_icu_mortality.csv.gz (file containing in-ICU mortality outcome per admission)

Make sure to save these files in the "data" folder

### Pre-process the chart events to extract and downsample relevant vitals and lower outcome rate
Run the "make_csv_dataset_from_raw.py" script to generate 3 files :

 - features_per_tstep.csv.gz (downsampled vitals with timestamps)
 - outcomes_per_seq.csv (outcomes with lowered rate)
 - demographics.csv.gz (demographics per admission)

`>> python make_csv_dataset_from_raw.py --dataset_raw_path "data" --output_dir "data"`

The full list of features can be found in [this](https://docs.google.com/spreadsheets/d/1Q3GfoC47P7nHhT8pDs73lJ5tGCQw6zhK49eTqn-gtyE/edit?usp=sharing) spec sheet
 

### Align to regularly spaced grid and split into train, valid and test
Run the script "handle_raw_data_and_split_train_test.smk" to align to a grid of equally spaced timestamps spaced 8 hours apart

    >> snakemake --cores 1 --snakefile handle_raw_data_and_split_train_test.smk make_features_and_outcomes_for_custom_times_prediction

This creates 2 files : 

 - features_aligned_to_grid.csv.gz (features spaced 8 hours apart)
 - outcomes_aligned_to_grid.csv.gz (Outcomes every 8 hours)

Generate the train/valid/test files ensuring that the patients in training do not appear in validation or test

    snakemake --cores 1 --snakefile handle_raw_data_and_split_train_test.smk split_into_train_and_test

This generates the following files
 - x_CustomTimes{train/valid/test}_vitals_only_csv.gz (train/valid/test features)
 - y_CustomTimes{train/valid/test}_vitals_only_csv.gz (train/valid/test outcomes)


### Artificially mask labels out to create versions of the data with 1.2%, 3.7%, 11.1%, 33.3% and 100% of the labels available
Run all the cells in the "create_ssl_datasets.ipynb" notebooks to generate the train/valid/test splits with 1.2%, 3.7%, 11.1%, 33.3% and 100% of the labels.

Note : The train valid test files will be saved in the "data/classifier_train_test_split_dir" folder

### Train and Evaluate PC-HMM and other supervised and SSL baselines (GRU-D/BRITS/MixMatch/FixMatch)
Run the training snakemake scripts to train the baselines and the PC-HMM models. Each model uses the corresponding {model}.json file for hyper parameters. The trained models are saved in the "results" folder.

    >> snakemake --cores 1 --snakefile train_semi_supervised_pchmm.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile train_semi_supervised_BRITS.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile train_gru_d_semi_supervised.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile train_semi_supervised_FixMatch.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile train_semi_supervised_MixMatch.smk train_and_evaluate_classifier_many_hyperparams

Run the evaluation scripts to choose the model with the best AUPRC on validation set for each model at multiple %s of missing labels

    >> snakemake --cores 1 --snakefile evaluate_semi_supervised_pchmm_performance.smk evaluate_performance
    >> snakemake --cores 1 --snakefile evaluate_BRITS_performance.smk evaluate_performance
    >> snakemake --cores 1 --snakefile evaluate_semi_supervised_grud_performance.smk evaluate_performance
    >> snakemake --cores 1 --snakefile evaluate_FixMatch_performance.smk evaluate_performance
    >> snakemake --cores 1 --snakefile evaluate_MixMatch_performance.smk evaluate_performance

 The evaluation script generates .csv files in the "results/model" folder with the performance which can be plotted as shown : 
 