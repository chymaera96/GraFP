
# Add a loop to run test with a list of test_snr

for snr in 0 5 10 20
do
    echo "Test with SNR = $snr"
    python test_fp.py --test_snr $snr
done