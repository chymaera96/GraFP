for k in 5, 7, 9, 11, 15
do
    echo "Test with k = $k"
    python test_fp.py --query_lens=1 --n_query_db=350 --test_snr=0 --text=350_fma_medium_k=$k --k=$k

done