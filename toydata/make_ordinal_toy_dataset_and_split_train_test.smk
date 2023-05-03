'''
Reproducible workflow for building toy dataset

Usage
-----
snakemake --cores 1 --snakefile make_ordinal_toy_dataset_and_split_train_test.smk build_csv_dataset_for_pchmm

snakemake --cores 1 --snakefile make_ordinal_toy_dataset_and_split_train_test.smk split_2d_features_into_many_train_and_test
'''

import json
import glob
configfile:"pchmm_train_test_splitting.json"

DATASET_SPLIT_PATH = os.path.join('ordinal_data', 'train_test_data')
PROJECT_CONDA_ENV_YAML = "pchmm.yml"

rule build_csv_dataset_for_pchmm:
    input:
        script='make_ordinal_toy_dataset.py'

    params:
        output_dir='ordinal_data'

    output:
        x_std_data_csv=os.path.join('ordinal_data', 'features_2d_per_tstep_no_imp_observed=60_perc.csv'),
        y_std_data_csv=os.path.join('ordinal_data', 'outcomes_per_seq.csv')
        
    shell:
        '''
        python -u {input.script} \
            --output_dir {params.output_dir} \
            --Nmax 350 \
            --Tmax 8 \
            --n_states 4 \
            --num_ordinal_labels 4 \
        '''

rule split_2d_features_into_many_train_and_test:
    input:
        [os.path.join(DATASET_SPLIT_PATH, 'x_train_{missing_handling}_observed={perc_obs}_perc.csv').format(perc_obs=perc_obs, missing_handling=missing_handling) for missing_handling in config['missing_handling'] for perc_obs in config['perc_obs']]

rule split_2d_features_into_train_and_test:
    input:
        script=os.path.abspath(os.path.join('../', 'utils', 'split_dataset.py')),
        x_csv=os.path.join('ordinal_data', 'features_2d_per_tstep_{missing_handling}_observed={perc_obs}_perc.csv'),
        x_json=os.path.join('ordinal_data', 'Spec_Features2DPerTimestep.json'),
        y_csv=os.path.join('ordinal_data', 'outcomes_per_seq.csv'),
        y_json=os.path.join('ordinal_data', 'Spec_OutcomesPerSequence.json')

    output:
        x_train_csv=os.path.join(DATASET_SPLIT_PATH, 'x_train_{missing_handling}_observed={perc_obs}_perc.csv'),
        x_test_csv=os.path.join(DATASET_SPLIT_PATH, 'x_test_{missing_handling}_observed={perc_obs}_perc.csv'),
        x_valid_csv=os.path.join(DATASET_SPLIT_PATH, 'x_valid_{missing_handling}_observed={perc_obs}_perc.csv'),
        y_train_csv=os.path.join(DATASET_SPLIT_PATH, 'y_train_{missing_handling}_observed={perc_obs}_perc.csv'),
        y_valid_csv=os.path.join(DATASET_SPLIT_PATH, 'y_valid_{missing_handling}_observed={perc_obs}_perc.csv'),
        y_test_csv=os.path.join(DATASET_SPLIT_PATH, 'y_test_{missing_handling}_observed={perc_obs}_perc.csv'),
        x_json=os.path.join(DATASET_SPLIT_PATH, 'x_dict_{missing_handling}_observed={perc_obs}_perc.json'),
        y_json=os.path.join(DATASET_SPLIT_PATH, 'y_dict_{missing_handling}_observed={perc_obs}_perc.json')

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        mkdir -p DATASET_SPLIT_PATH \
        && python -u {input.script} \
            --input {input.x_csv} \
            --data_dict {input.x_json} \
            --test_size 0.15 \
            --group_cols sequence_id \
            --train_csv_filename {output.x_train_csv} \
            --valid_csv_filename {output.x_valid_csv} \
            --test_csv_filename {output.x_test_csv} \
            --output_data_dict_filename {output.x_json} \
        && python -u {input.script} \
            --input {input.y_csv} \
            --data_dict {input.y_json} \
            --test_size 0.15 \
            --group_cols sequence_id \
            --train_csv_filename {output.y_train_csv} \
            --valid_csv_filename {output.y_valid_csv} \
            --test_csv_filename {output.y_test_csv} \
            --output_data_dict_filename {output.y_json} \
        '''.replace("DATASET_SPLIT_PATH", DATASET_SPLIT_PATH)
