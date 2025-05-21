# V2C-Long

This repository implements [V2C-Long](https://direct.mit.edu/imag/article/doi/10.1162/imag_a_00500/127971/V2C-Long-Longitudinal-Cortex-Reconstruction-with).

<p float="left">
<img src="/media/v2c-long.png" width="600" /> 
</p>

## Installation
For installation instructions, see our [Vox2Cortex](https://github.com/ai-med/Vox2Cortex) repo. This (V2C-Long) repo is heavily based on it.

## Usage
To get started, add your dataset to `vox2organ/data/supported_datasets.py`. In addition, create a .csv file that describes your longitudinal data; see `supplementary_material/example_data.csv` for an example.
### Inference
We provide pre-trained models for inference, either for the right hemisphere only or for the entire cortex, see `public_experiments/`. The inference process involves three steps (the following examples focus on the right hemisphere and assume the dataset is called `TEST_DATASET_LONG`):
1. Run V2C-Flow
```
cd vox2organ
python main.py --test -n v2c-flow-s-rh_base --dataset TEST_DATASET_LONG --experiment_base_dir ../public_experiments/
```
2. Create within-subject templates
```
python scripts/create_mean_meshes.py ../public_experiments/v2c-flow-s-rh_base/test_template_fsaverage-smooth-rh_TEST_DATASET_LONG_n_5/
```
3. Run V2C-Long
```
python main.py --test -n v2c-long-rh --dataset TEST_DATASET_LONG --experiment_base_dir ../public_experiments/
```

### Training
Training works similar as for V2C-Flow, please refer to [this](https://github.com/ai-med/Vox2Cortex) repo.


## Citation
If you find this work useful, please cite
```
@article{Bongratz2025V2CLong,
    author = {Bongratz, Fabian and Fecht, Jan and Rickmann, Anne-Marie and Wachinger, Christian},
    title = {V2C-Long: Longitudinal cortex reconstruction with spatiotemporal correspondence},
    journal = {Imaging Neuroscience},
    volume = {3},
    pages = {imag_a_00500},
    year = {2025},
    month = {03},
    issn = {2837-6056},
    doi = {10.1162/imag_a_00500},
    url = {https://doi.org/10.1162/imag\_a\_00500},
    eprint = {https://direct.mit.edu/imag/article-pdf/doi/10.1162/imag\_a\_00500/2503305/imag\_a\_00500.pdf},
}

