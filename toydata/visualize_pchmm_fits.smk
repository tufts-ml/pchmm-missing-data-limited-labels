'''
Visualize fit pchmm models

Usage
-----
snakemake --cores 1 --snakefile visualize_pchmm_fits.smk

'''
PROJECT_CONDA_ENV_YAML = "pchmm.yaml"
RESULTS_FEAT_PER_TSTEP_PATH = "saved_models"
DATASET_DIR = "data"


rule visualize_pchmm_fits:
    input:
        script='visualize_pchmm_fits.py'

    params:
        fits_dir=RESULTS_FEAT_PER_TSTEP_PATH,
        data_dir=DATASET_DIR
        
    conda:
        PROJECT_CONDA_ENV_YAML

    shell:
        '''
        python -u {input.script} \
            --data_dir {params.data_dir} \
            --fits_dir {params.fits_dir} \
        '''