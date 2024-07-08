# Usage: bash test_pipeline.sh ab017


for snr in 20 15 10 5 0
do
    echo "Test with SNR = $snr"
    # top -b -n 1 -u $1 | grep python | awk '{print $1}' | xargs kill -9
    # python test_fp.py --query_lens=1,2,3,5 --n_dummy_db=100 --test_snr=$snr --text=fma_medium_$snr --small_test=False
    python test_fp.py --query_lens=1,2,3,5 --n_query_db=350 --test_snr=$snr --text=350_fma_medium_$snr

done