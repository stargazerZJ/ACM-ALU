#!/usr/bin/bash

# make test directory
mkdir -p test

# Compile
iverilog.exe -o test/fp_adder_test.out test_adder_file.v || exit 1
g++ --std=c++20 -O2 std.cpp -o test/std || exit 1

cd test || exit

# Generate test cases
python ../gen.py > input.txt

# Run std
./std < input.txt > output_std.txt 2> output_std.log

# Run iverilog
vvp.exe fp_adder_test.out > output_iverilog.txt # the verilog reads from input.txt

# Compare
diff -uZB output_std.txt output_iverilog.txt
# cat output_iverilog.txt

if [ $? -eq 0 ]; then
    echo "Test passed"
else
    echo "Test failed, though minimal differences may be due to floating point arithmetic"
fi