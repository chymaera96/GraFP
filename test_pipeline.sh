
# Add a loop to run test with a list of test_snr
# Take user as parameter

# Usage: bash test_pipeline.sh ab017


for snr in 20 10 5
do
    echo "Test with SNR = $snr"
    top -b -n 1 -u $1 | grep python | awk '{print $1}' | xargs kill -9
    python test_fp.py --query_lens=1,2,3,5 --n_dummy_db=100 --test_snr=$snr --text=sanity_test_$snr --small_test=False
done