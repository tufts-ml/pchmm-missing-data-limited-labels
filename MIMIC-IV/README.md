# Reproducing MIMIC-IV in-ICU mortality prediction and LOS ordinal regression experiments


## Workflow

### Downloading the chart events and ICU mortality outcomes
Clone the [MIMIC-IV Data-Pipeline](https://github.com/healthylaife/mimic-iv-data-pipeline) repo and follow the instructions to extract the chart events and ICU mortality outomes. We used the "data extraction" in the [mainPipeline.ipynb](https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/mainPipeline.ipynb) notebook to extract the chart events without any disease filters.

Running the notebook generates 3 files : 

 - preproc_chart.csv.gz (all the vitals with timestamps)
 - d_items.csv.gz (file linking the vital IDs to the vital names)
 - cohort_icu_mortality.csv.gz (file containing in-ICU mortality outcome per admission)

Make sure to save these files in the "data" folder

### Pre-process the chart events to extract and downsample relevant vitals and outcomes
Run the "make_csv_dataset_from_raw.py" script to generate 3 files :

 - features_per_tstep.csv.gz (hourly spaced vitals with timestamps)
 - outcomes_per_seq.csv (mortality and LOS outcomes)
 - demographics.csv.gz (demographics per admission)

`>> python make_csv_dataset_from_raw_for_mortality_and_los_prediction.py --dataset_raw_path "data/MIMIC-IV" --output_dir "data/MIMIC-IV"`

The full list of features can be found in [this](https://docs.google.com/spreadsheets/d/1Q3GfoC47P7nHhT8pDs73lJ5tGCQw6zhK49eTqn-gtyE/edit?usp=sharing) spec sheet


### Mask labels out to create versions of the data with 1.2%, 3.7%, 11.1%, 33.3% and 100% of the labels available
Run all the cells in the "notebooks/create_ssl_dataset_for_mortality_prediction.ipynb" and "notebooks/create_ssl_dataset_for_mortality_prediction.ipynb" notebooks to generate the train/valid/test splits with 1.2%, 3.7%, 11.1%, 33.3% and 100% of the labels for the mortality prediction and LOS ordinal regression tasks respectively. 

Note : The train valid test files will be saved in the "data/MIMIC-IV/ordinal_los_prediction" and "data/MIMIC-IV/mortality_prediction" folders

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


The notebooks can be used to plot the performance of each model: 
![PCHMM_vs_baselines](https://github.com/tufts-ml/pchmm-missing-data-limited-labels/blob/main/MIMIC-IV/figures/perf_mortality_prediction_mimic.pdf)
 

