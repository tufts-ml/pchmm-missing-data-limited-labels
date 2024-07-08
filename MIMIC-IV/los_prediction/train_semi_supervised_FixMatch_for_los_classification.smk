'''
Train FixMatch ON MIMIC Inhospital Mortality Task

Usage
-----
snakemake --cores 1 --snakefile train_semi_supervised_FixMatch_for_los_classification.smk train_and_evaluate_classifier_many_hyperparams

Schedule as slurm jobs
----------------------
$ snakemake --snakefile train_semi_supervised_FixMatch_for_los_classification.smk --profile ../../utils/profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

'''

# Default environment variables
# Can override with local env variables
configfile:"FixMatch.json"

PROJECT_REPO_DIR = os.path.abspath("../")
RESULTS_FEAT_PER_TSTEP_PATH=os.path.abspath("../results/FixMatch/los_classification")
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/datasets/MIMIC-IV/ordinal_los_prediction/"


rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"FixMatch-lr={lr}-seed={seed}-min_los={min_los}-batch_size={batch_size}-perc_labelled={perc_labelled}.csv").format(lr=lr, seed=seed, min_los=min_los, batch_size=batch_size, perc_labelled=perc_labelled) for lr in config['lr'] for seed in config['seed'] for batch_size in config['batch_size'] for perc_labelled in config['perc_labelled'] for min_los in config['min_los']]
        
rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'FixMatch', 'main.py'),
        x_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction', 'percentage_labelled_sequences={perc_labelled}', 'X_train.npy'),
        x_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','X_valid.npy'),
        x_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH,  'los_geq_{min_los}_days_prediction','X_test.npy'),
        y_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction', 'percentage_labelled_sequences={perc_labelled}', 'y_train.npy'),
        y_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','y_valid.npy'),
        y_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','y_test.npy')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="FixMatch-lr={lr}-seed={seed}-min_los={min_los}-batch_size={batch_size}-perc_labelled={perc_labelled}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "FixMatch-lr={lr}-seed={seed}-min_los={min_los}-batch_size={batch_size}-perc_labelled={perc_labelled}.csv")
        

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --output_dir {params.output_dir} \
            --train_np_files {input.x_train_np},{input.y_train_np} \
            --valid_np_files {input.x_valid_np},{input.y_valid_np} \
            --test_np_files {input.x_test_np},{input.y_test_np} \
            --lr {wildcards.lr} \
            --manualSeed {wildcards.seed} \
            --perc_labelled {wildcards.perc_labelled} \
            --batch_size {wildcards.batch_size} \
            --output_filename_prefix {params.fn_prefix} \
        '''

