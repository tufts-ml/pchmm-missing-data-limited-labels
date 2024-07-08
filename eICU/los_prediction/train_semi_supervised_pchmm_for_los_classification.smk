'''
Train PC-HMM ON EICU LOS prediction Task

Usage
-----
snakemake --cores 1 --snakefile train_semi_supervised_pchmm_for_los_classification.smk train_and_evaluate_classifier_many_hyperparams


snakemake --snakefile train_semi_supervised_pchmm_for_los_classification.smk --profile ../../utils/profiles/hugheslab_cluster/ train_and_evaluate_classifier_many_hyperparams

Schedule as slurm jobs
----------------------

'''

# Default environment variables
# Can override with local env variables
configfile:"semi_supervised_pchmm_los_classification.json"


RESULTS_FEAT_PER_TSTEP_PATH=os.path.abspath("../results/PCHMM/los_classification")
CLF_TRAIN_TEST_SPLIT_PATH = "/cluster/tufts/hugheslab/datasets/eicu_v2.0/ordinal_los_prediction/"
PROJECT_REPO_DIR = os.path.abspath("../../MIMIC-IV/PC-HMM")

rule train_and_evaluate_classifier_many_hyperparams:
    input:
        [os.path.join(RESULTS_FEAT_PER_TSTEP_PATH,"semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-batch_size={batch_size}-min_los={min_los}-predictor_l2_penalty={predictor_l2_penalty}-perc_labelled={perc_labelled}-n_states={n_states}-lamb={lamb}.csv").format(lr=lr, seed=seed, init_strategy=init_strategy, batch_size=batch_size, min_los=min_los, n_states=n_states, lamb=lamb, predictor_l2_penalty=predictor_l2_penalty, perc_labelled=perc_labelled) for lr in config['lr'] for seed in config['seed'] for init_strategy in config['init_strategy'] for batch_size in config['batch_size'] for min_los in config['min_los'] for n_states in config['n_states'] for lamb in config['lamb'] for predictor_l2_penalty in config['predictor_l2_penalty'] for perc_labelled in config['perc_labelled']]
        
rule train_and_evaluate_classifier:
    input:
        script=os.path.join(PROJECT_REPO_DIR, 'main_eicu_semi_supervised.py'),
        x_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction', 'percentage_labelled_sequences={perc_labelled}', 'X_train.npy'),
        x_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','X_valid.npy'),
        x_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH,  'los_geq_{min_los}_days_prediction','X_test.npy'),
        y_train_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction', 'percentage_labelled_sequences={perc_labelled}', 'y_train.npy'),
        y_valid_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','y_valid.npy'),
        y_test_np=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'los_geq_{min_los}_days_prediction','y_test.npy')

    params:
        output_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        fn_prefix="semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-batch_size={batch_size}-min_los={min_los}-predictor_l2_penalty={predictor_l2_penalty}-perc_labelled={perc_labelled}-n_states={n_states}-lamb={lamb}"
    
    output:
        os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, "semi-supervised-pchmm-lr={lr}-seed={seed}-init_strategy={init_strategy}-batch_size={batch_size}-min_los={min_los}-predictor_l2_penalty={predictor_l2_penalty}-perc_labelled={perc_labelled}-n_states={n_states}-lamb={lamb}.csv")
        

    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script} \
            --output_dir {params.output_dir} \
            --train_np_files {input.x_train_np},{input.y_train_np} \
            --valid_np_files {input.x_valid_np},{input.y_valid_np} \
            --test_np_files {input.x_test_np},{input.y_test_np} \
            --lr {wildcards.lr} \
            --n_states {wildcards.n_states} \
            --seed {wildcards.seed} \
            --batch_size {wildcards.batch_size} \
            --predictor_l2_penalty {wildcards.predictor_l2_penalty} \
            --init_strategy {wildcards.init_strategy} \
            --output_filename_prefix {params.fn_prefix} \
            --lamb {wildcards.lamb} \
        '''


