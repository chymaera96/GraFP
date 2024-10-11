#!/bin/bash

# Download the pre-trained models and fingerprint databases 
echo "Downloading the trained models..."
wget https://huggingface.co/chymaera96/grafp_db/resolve/main/checkpoint.zip
unzip checkpoint.zip -d data/
mv data/checkpoint/AST/* baselines/checkpoint/
mv data/checkpoint/GraFP/* checkpoint/
rm -r data/checkpoint
rm checkpoint.zip

echo "Downloading the fingerprint databases..."
wget https://huggingface.co/chymaera96/grafp_db/resolve/main/databases.zip
unzip databases.zip -d data/
mkdir -p logs/store
mv data/databases/medium logs/store/
mv data/databases/large logs/store/
rm -r data/databases
rm databases.zip

if [[ $1 == */ ]]; then
    1=${1::-1}
fi
if [[ $2 == */ ]]; then
    2=${2::-1}
fi

# Setup
python setup_config.py --noise_dir=$2/noise --ir_dir=$2/ir
eval=$(basename $1)
python setup_icassp.py --test_dir=$1 --noise_dir=$2/noise --ir_dir=$2/ir --eval_type=$eval

# Evaluation runs
echo "########## Evaluating with without IR corruption ##########"

if [[ $eval == "fma_medium" ]]; then
    for snr in 20 15 10 5 0
    do
        echo "Test with SNR = $snr"
        python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 \
                            --test_snr=$snr --text=sanir_fma_medium_$snr \
                            --test_ids=data/medeval_ids.npy --model=tc_29
    done
    for snr in 20 15 10 5 0
    do
        echo "Test with SNR = $snr"
        python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_dir=data/fma_large.json \
                            --test_snr=$snr --text=sanir_fma_large_$snr \
                            --test_ids=data/largeval_ids.npy --model=tc_29
    done
else
fi

echo "########## Evaluating with with IR corruption ##########"

if [[ $eval == "fma_medium" ]]; then
    for snr in 20 15 10 5 0
    do
        echo "Test with SNR = $snr"
        python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 \
                            --test_snr=$snr --text=withir_fma_medium_$snr \
                            --test_ids=data/medeval_ids.npy --model=tc_29
    done
    for snr in 20 15 10 5 0
    do
        echo "Test with SNR = $snr"
        python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_dir=data/fma_large.json \
                            --test_snr=$snr --text=withir_fma_large_$snr \
                            --test_ids=data/largeval_ids.npy --model=tc_29
    done
else
fi