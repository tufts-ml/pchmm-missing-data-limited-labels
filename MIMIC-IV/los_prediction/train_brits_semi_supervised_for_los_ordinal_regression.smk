'''
Train full sequence classifier on mimic inhospital mortality task

Usage:

Schedule as slurm jobs
----------------------
$ snakemake --snakefile train_brits_semi_supervised_for_los_ordinal_regression.smk --profile ../../utils/profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams train_and_evaluate_classifier_many_hyperparams

Train single hyperparam at a time
---------------------------------

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 1 --snakefile train_brits_semi_supervised_for_los_ordinal_regression.smk train_and_evaluate_classifier_many_hyperparams

'''

configfile:"BRITS.json"

import os

PROJECT_REPO_DIR = os.path.abspath("../")
RESULTS_FEAT_PER_TSTEP_PATH=os.path.abspath("../results/BRITS/los_ordinal_regression")
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/datasets/MIMIC-IV/ordinal_los_prediction/"


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"BRITS-semi_supervised-lr={lr}-batch_size={batch_size}-seed={seed}-perc_labelled={perc_labelled}.csv").format(lr=lr, batch_size=batch_size, seed=seed, perc_labelled=perc_labelled) for lr in config['lr'] for batch_size in config['batch_size'] for seed in config['seed'] for perc_labelled in config['perc_labelled']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'BRITS', 'main_ordinal.py'),
        x_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_prediction_ordinal', 'percentage_labelled_sequences={perc_labelled}', 'X_train.npy'),
        x_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_prediction_ordinal','X_valid.npy'),
        x_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH,  'los_prediction_ordinal','X_test.npy'),
        y_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_prediction_ordinal','percentage_labelled_sequences={perc_labelled}', 'y_train.npy'),
        y_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_prediction_ordinal','y_valid.npy'),
        y_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_prediction_ordinal','y_test.npy')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="BRITS-semi_supervised-lr={lr}-batch_size={batch_size}-seed={seed}-perc_labelled={perc_labelled}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "BRITS-semi_supervised-lr={lr}-batch_size={batch_size}-seed={seed}-perc_labelled={perc_labelled}.csv")
        

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --output_dir {params.output_dir} \
            --train_np_files {input.x_train_np},{input.y_train_np} \
            --valid_np_files {input.x_valid_np},{input.y_valid_np} \
            --test_np_files {input.x_test_np},{input.y_test_np} \
            --lr {wildcards.lr} \
            --seed {wildcards.seed} \
            --perc_labelled {wildcards.perc_labelled} \
            --batch_size {wildcards.batch_size} \
            --output_filename_prefix {params.fn_prefix} \
        '''

