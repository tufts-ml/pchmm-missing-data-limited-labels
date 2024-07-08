'''
Produce a collapsed feature representation for human activities
and produce train/test CSV files

Usage
-----
>> snakemake --cores 1 --snakefile handle_raw_data_and_split_train_test.smk make_features_and_outcomes_for_custom_times_prediction

>> snakemake --cores 1 --snakefile handle_raw_data_and_split_train_test.smk split_into_train_and_test
'''


DATASET_SPLIT_FEAT_PER_TSTEP_PATH = "data"
CLF_TRAIN_TEST_SPLIT_PATH = os.path.join(DATASET_SPLIT_FEAT_PER_TSTEP_PATH, 'classifier_train_test_split_dir')
DATASET_RAW_PATH = "data"


print("Building collapsed dataset")
print("--------------------------")
print("Train/test dataset will go to folder:")
print(CLF_TRAIN_TEST_SPLIT_PATH)

# Default environment variables
# Can override with local env variables

rule make_features_and_outcomes_for_custom_times_prediction:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'make_features_and_outcomes_for_custom_times_prediction.py'),
        features_csv = os.path.join(DATASET_RAW_PATH, "features_per_tstep.csv.gz"),
        outcomes_csv=os.path.join(DATASET_RAW_PATH, "outcomes_per_seq.csv"),
        features_data_dict=os.path.join(DATASET_RAW_PATH, 'Spec_FeaturesPerTimestepIrregularlySampled.json')

    params:
        preproc_data_dir=DATASET_RAW_PATH,
        output_dir=CLF_TRAIN_TEST_SPLIT_PATH

    output:
        output_features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_aligned_to_grid.csv.gz"),
        output_outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_aligned_to_grid.csv.gz")
    
    shell:
        '''
            mkdir -p {{params.output_dir}} && \
            python -u {input.script} \
                --features_csv {input.features_csv} \
                --outcomes_csv {input.outcomes_csv} \
                --preproc_data_dir {params.preproc_data_dir} \
                --output_dir {params.output_dir} \
                --features_data_dict {input.features_data_dict} \
                --output_features_csv {output.output_features_csv} \
                --output_outcomes_csv {output.output_outcomes_csv} \
        '''
        
rule split_into_train_and_test:
    input:
        script=os.path.join(os.path.abspath('../'), 'src', 'split_dataset_by_timestamp.py'),
        features_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_aligned_to_grid.csv.gz"),
        outcomes_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_aligned_to_grid.csv.gz"),
        features_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "features_dict.json"),
        outcomes_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, "outcomes_dict.json"),

    params:
        train_test_split_dir=CLF_TRAIN_TEST_SPLIT_PATH

    output:  
        x_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_trainCustomTimes_10_6_vitals_only.csv.gz'),
        x_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_validCustomTimes_10_6_vitals_only.csv.gz'),
        x_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_testCustomTimes_10_6_vitals_only.csv.gz'),
        y_train_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_trainCustomTimes_10_6_vitals_only.csv.gz'),
        y_valid_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_validCustomTimes_10_6_vitals_only.csv.gz'),
        y_test_csv=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_testCustomTimes_10_6_vitals_only.csv.gz'),
        x_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'x_dictCustomTimes_10_6_vitals_only.json'),
        y_dict_json=os.path.join(CLF_TRAIN_TEST_SPLIT_PATH, 'y_dictCustomTimes_10_6_vitals_only.json')

    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
            mkdir -p {{params.train_test_split_dir}} && \
            python -u {{input.script}} \
                --input {{input.features_csv}} \
                --data_dict {{input.features_json}} \
                --test_size {split_test_size} \
                --train_csv_filename {{output.x_train_csv}} \
                --valid_csv_filename {{output.x_valid_csv}} \
                --test_csv_filename {{output.x_test_csv}} \
                --output_data_dict_filename {{output.x_dict_json}} \

            python -u {{input.script}} \
                --input {{input.outcomes_csv}} \
                --data_dict {{input.outcomes_json}} \
                --test_size {split_test_size} \
                --train_csv_filename {{output.y_train_csv}} \
                --valid_csv_filename {{output.y_valid_csv}} \
                --test_csv_filename {{output.y_test_csv}} \
                --output_data_dict_filename {{output.y_dict_json}} \
        '''.format(
            split_test_size=D_CONFIG['SPLIT_TEST_SIZE'],
            )
