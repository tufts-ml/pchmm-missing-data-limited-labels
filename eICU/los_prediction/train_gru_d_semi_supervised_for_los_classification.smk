'''
Train full sequence classifier on eicu los prediction task

Usage:

Schedule as slurm jobs
----------------------
$ snakemake --snakefile train_gru_d_semi_supervised_for_los_classification.smk --profile ../../utils/profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

Train single hyperparam at a time
---------------------------------

To run with multiple random seeds (prespecified in a config file)
$ snakemake --cores 1 --snakefile train_gru_d_semi_supervised_for_los_classification.smk train_and_evaluate_classifier_many_hyperparams

'''

configfile:"semi_supervised_gru_d.json"

import os

PROJECT_REPO_DIR = os.path.abspath("../../MIMIC-IV/")
RESULTS_FEAT_PER_TSTEP_PATH=os.path.abspath("../results/GRUD/los_classification")
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/datasets/eicu_v2.0/ordinal_los_prediction/"


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"final_perf_GRUD-semi_supervised-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-min_los={min_los}-seed={seed}-perc_labelled={perc_labelled}.csv").format(lr=lr, dropout=dropout, weight_decay=weight_decay, batch_size=batch_size, seed=seed, min_los=min_los, perc_labelled=perc_labelled) for lr in config['lr'] for dropout in config['dropout'] for weight_decay in config['weight_decay'] for batch_size in config['batch_size'] for seed in config['param_init_seed'] for min_los in config['min_los'] for perc_labelled in config['perc_labelled']]

rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'GRU_D', 'main_eicu_semi_supervised.py'),
        x_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction', 'percentage_labelled_sequences={perc_labelled}', 'X_train.npy'),
        x_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','X_valid.npy'),
        x_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH,  'los_geq_{min_los}_days_prediction','X_test.npy'),
        y_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction', 'percentage_labelled_sequences={perc_labelled}', 'y_train.npy'),
        y_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','y_valid.npy'),
        y_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','y_test.npy')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="GRUD-semi_supervised-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-min_los={min_los}-seed={seed}-perc_labelled={perc_labelled}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "final_perf_GRUD-semi_supervised-lr={lr}-dropout={dropout}-weight_decay={weight_decay}-batch_size={batch_size}-min_los={min_los}-seed={seed}-perc_labelled={perc_labelled}.csv")
        

    shell:
        '''
        python -u {input.script} \
            --output_dir {params.output_dir} \
            --train_np_files {input.x_train_np},{input.y_train_np} \
            --valid_np_files {input.x_valid_np},{input.y_valid_np} \
            --test_np_files {input.x_test_np},{input.y_test_np} \
            --lr {wildcards.lr} \
            --seed {wildcards.seed} \
            --batch_size {wildcards.batch_size} \
            --dropout {wildcards.dropout} \
            --l2_penalty {wildcards.weight_decay} \
            --output_filename_prefix {params.fn_prefix} \
        '''

