# Reproducing toy 2-D experiment


## Workflow

### Create the toy data
Run the toy dataset generation snakemake files. This creates the toy dataset with missing features without imputation/with forward fill imputation/with mean imputation


    >> snakemake --cores 1 --snakefile make_toy_dataset_and_split_train_test.smk build_csv_dataset_for_pchmm

Split the data into train and test

    >> snakemake --cores 1 --snakefile make_toy_dataset_and_split_train_test.smk split_2d_features_into_many_train_and_test

### Train the PC-HMM and other baselines on data with and without imputation

Train and evaluate the PC-HMM : 
The training files are saved in the "training logs" folder. Once trained, the "evaluate_pchmm.smk" script saves the performance of the best model in the results folder. This workflow is followed by all models.

    >> snakemake --cores 1 --snakefile train_pchmm.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile evaluate_pchmm.smk

Train and evaluate the GRU-D

    >> snakemake --cores 1 --snakefile train_GRUD.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile evaluate_GRUD.smk
    
Train and evaluate the LSTM

    >> snakemake --cores 1 --snakefile train_LSTM.smk train_and_evaluate_classifier_many_hyperparams
    >> snakemake --cores 1 --snakefile evaluate_LSTM.smk


### Visualize the PC-HMM at 40% missingness
The script "visualize_pchmm_fits.smk" below chooses the best model after training and visualizes the fit on the toy data
Note : For now, the best fit is saved in the "saved_models" folder. Once you  have followed the training section above, `--fits_dir` argument should be changed to the folder with the saved models.

    >> snakemake --cores 1 --snakefile visualize_pchmm_fits.smk

This creates the visualization of the best model
![PC-HMM toy data best fit visualization](https://github.com/tufts-ml/pchmm-missing-data-limited-labels/blob/main/toydata/figures/pchmm_fits_no_imp_perc_obs=60.png)

