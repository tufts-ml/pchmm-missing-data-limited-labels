'''
>> snakemake --cores 1 --snakefile evaluate_FixMatch_performance.smk evaluate_performance
'''

import glob
import os
import sys
RESULTS_FEAT_PER_TSTEP_PATH="results/FixMatch/"
CLF_TRAIN_TEST_SPLIT_PATH = "data/classifier_train_test_split_dir/percentage_labelled_sequnces=100/"
CLF_MODELS_PATH = RESULTS_FEAT_PER_TSTEP_PATH
random_seed_list = [42, 1783, 78970, 86787, 8675309]
DATASET_STD_PATH = "data"

rule evaluate_performance:
    input:
        script=os.path.join(os.path.abspath('../'), "src", "evaluate_FixMatch_performance.py")

    params:
        clf_models_dir=CLF_MODELS_PATH,
        clf_train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH,
        tslice_folder=DATASET_SPLIT_FEAT_PER_TSLICE_PATH,
        preproc_data_dir=DATASET_STD_PATH,
        random_seed_list=random_seed_list,
        output_dir=os.path.join(RESULTS_FEAT_PER_TSTEP_PATH, 'classifier_per_tslice_performance')
        
    conda:
        PROJECT_CONDA_ENV_YAML
        
    shell:
        '''
        mkdir -p {params.output_dir} && \
        python -u {input.script}\
        --clf_models_dir {params.clf_models_dir}\
        --clf_train_test_split_dir {params.clf_train_test_split_dir}\
        --tslice_folder {params.tslice_folder}\
        --preproc_data_dir {params.preproc_data_dir}\
        --outcome_column_name {{OUTCOME_COL_NAME}}\
        --random_seed_list "{params.random_seed_list}"\
        --output_dir {params.output_dir}\
        '''.replace("{{OUTCOME_COL_NAME}}", D_CONFIG["OUTCOME_COL_NAME"])