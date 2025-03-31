# GraFPrint: A GNN-Based Approach for Audio Identification

This is the official repository for our state-of-the-art audio identification framework based on graph neural networks. We demonstrate the code usage for training, audio fingerprint generation and evaluation. For more details, refer to the [paper](https://ieeexplore.ieee.org/abstract/document/10888557) at Interanational Conference on Acoustics, Speech, and Signal Processing (ICASSP) 2025.

## Installation Guide

1. Clone the repository:
    ```bash
    git clone https://github.com/username/GraFP.git
    cd GraFP
    ```
2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```


## Training Setup

As per our experiments, we recommend using the `fma-small` subset of the [Free Music Archive (FMA)](https://github.com/mdeff/fma) dataset. For the noise and room impulse response (RIR) dataset, we recommend using the [MUSAN](https://www.openslr.org/17/) dataset and the [Aachen Impulse Response](https://www.iks.rwth-aachen.de/en/research/tools-downloads/databases/aachen-impulse-response-database/) database, respectively.

1. Setup the config files with paths to datasets
    ```bash
    python setup_config.py --train_dir /PATH/TO/TRAIN/DATA --val_dir /PATH/TO/VALIDATION/DATA --noise_dir /PATH/TO/NOISE/DATA --ir_dir /PATH/TO/IR/DATA
    ```
2. Run the training script:
    ```bash
    python train.py 
    ```

## Generate Fingerprints

We provide a helper code for generating audio fingerprints for a given audio dataset. The pre-trained models are available [here](https://huggingface.co/chymaera96/grafp_db/resolve/main/checkpoint.zip). The primary evaluation benchmarks have been computed using `model_tc_29_best.pth`.

```bash
python generate.py --test_dir /PATH/TO/TEST/DATA --ckp /PATH/TO/MODEL
```

## Evaluation setup

For reproducibility, we have made the dummy fingerprint database available [here](https://huggingface.co/chymaera96/grafp_db/resolve/main/databases.zip). The fingerprint retrieval pipeline utilizes the [FAISS](https://github.com/facebookresearch/faiss) library for the approximate nearest-neighbour (ANN) search in the fingerprint embedding space. Further details about the ANN implementation is available in our pre-print document. The `icassp.sh` script can be used to run the evaluation pipeline for reproducing the published results. 

1. Download and extract the test dataset. The script supports evaluation on both the `fma-medium` and `fma-large` datasets. Note that extracting the compressed `fma_large.zip` can take a while. For quicker evaluation runs, we recommend extracting the `fma_medium.zip`.
2. Download and extract the augmentation dataset. Queries are created using subset of the background noise and impulse response datasets. They can be downloaded [here](https://huggingface.co/chymaera96/grafp_db/resolve/main/aug.zip).
2. Run the evaluation script with the pre-trained model:
    ```bash
    bash icassp.sh /PATH/TO/EVAL/DATASET /PATH/TO/AUG/DATASET
    ```
Note that the evaluation dataset path provided in the above script should be the absolute path to the directory called `fma_medium` or `fma_large`. Logs such as raw outputs and retrieval hit-rates can be found in the `logs/store` directory. Each output run is organized according the filename of the pre-trained model used. Support for running evaluation on private datasets would be made available soon. 

## Citation

If you use this code in repository, please cite our paper:
```bibtex
@inproceedings{grafprint2025,
  title={GraFPrint: A GNN-Based Approach for Audio Identification},
  author={Bhattacharjee, Aditya and Singh, Shubhr and Benetos, Emmanouil},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

```
