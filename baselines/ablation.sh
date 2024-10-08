
for snr in 20 15 10 5 0
# for snr in 0
do
    echo "Test with SNR = $snr"
    python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=$snr --text=sanir_fma_medium_$snr \
    --test_ids=../logs/store/test/model_tc_29_best/test_ids.npy \
    --model=$1

done

for snr in 20 15 10 5 0
# for snr in 0
do
    echo "Test with SNR = $snr"
    # python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=$snr --text=1000_fma_medium_$snr --test_ids=4000
    python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=$snr --text=4000_fma_medium_$snr \
    --test_ids=../logs/store/test/model_tc_29_best/test_ids.npy \
     --model=$1

done