
# Add a loop to run test with a list of test_snr

for snr in 20 10 5 0
do
    echo "Test with SNR = $snr"
    top -b -n 1 -u ab017| grep python | awk '{print $1}' | xargs kill -9
    python test_fp.py --query_lens=1,2,3,5 --n_dummy_db=100 --test_snr=$snr --text=sz_100_test4_snr_nil --small_test=True
done