rm -rf output
mkdir output
echo [DEMO] running GMiner...
./GMiner -i sample_input/sample.dat -o output/sample.out -w 1 >> output/DEMO.Result
echo [DEMO] frequent itemsets and the summary of demo are found in output
