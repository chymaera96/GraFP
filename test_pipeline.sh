# Take model name as input and run the test pipeline

# for snr in 20 15 10 5 0
for snr in 0
do
    echo "Test with SNR = $snr"
    # top -b -n 1 -u $1 | grep python | awk '{print $1}' | xargs kill -9
    # python test_fp.py --query_lens=1,2,3,5 --n_dummy_db=100 --test_snr=$snr --text=fma_medium_$snr --small_test=False
    python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=$snr --text=4000_fma_medium_$snr --test_ids=4000 --model=$1
    # python test_fp.py --query_lens=1,2,3,5 --n_query_db=500 --test_snr=$snr --text=4000_fma_medium_$snr --test_ids=4000 --test_dir=/data/scratch/acw723/fma_large --model=$1


done