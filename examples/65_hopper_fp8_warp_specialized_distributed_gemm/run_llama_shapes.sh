#!/bin/bash

get_tflops() {
    if [ -z "$1" ]; then
        echo "Error: No output provided" >&2
        return -1
    fi 

    local output="$1"
    local tflops

    # Extract TFLOPS value using grep and awk
    tflops=$(echo "$output" | grep -o 'TFLOPS: [0-9]*\.*[0-9]*' | awk '{print $2}')
    # tflops=$(echo "$output" | grep -i "TFLOPS:" | awk '{print $2}')
    
    # Check if TFLOPS was found
    if [ -z "$tflops" ]; then
        echo "Error: TFLOPS value not found in output" >&2
        return -2
    fi 

    # Verify that TFLOPS is a number
    if ! [[ "$tflops" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        echo "Error: Invalid TFLOPS value found: $tflops" >&2
        return -4
    fi

    # Output the TFLOPS value
    echo $tflops
    return 0
}

run_gemm() {
    # Check if correct number of arguments provided
    if [ "$#" -ne 5 ]; then
        echo "Error: Exactly 5 arguments required"
        echo "Usage: run_gemm M N K raster swizzle"
        return -1
    fi 

    # Assign arguments to named variables
    local m="$1"
    local n="$2"
    local k="$3"
    local raster="$4"
    local swizzle="$5"

    # Validate numeric inputs
    if ! [[ "$m" =~ ^[0-9]+$ ]] || ! [[ "$n" =~ ^[0-9]+$ ]] || ! [[ "$k" =~ ^[0-9]+$ ]]; then
        echo "Error: M, N, and K must be positive integers"
        return -1
    fi 

    # Validate raster input (assuming it should be N or other specific values)
    if [[ ! "$raster" =~ ^[MNH]$ ]]; then
        echo "Error: raster must be 'N' 'M' or 'H'"
        return -1
    fi

    # Validate swizzle input (assuming it should be a positive integer)
    if ! [[ "$swizzle" =~ ^[0-9]+$ ]]; then
        echo "Error: swizzle must be a positive integer"
        return -1
    fi

    # Check if the executable exists
    if [ ! -x "./65_hopper_fp8_warp_specialized_distributed_gemm" ]; then
        echo "Error: GEMM executable not found or not executable"
        return -1
    fi 

    # Run the command and capture output
    local output
    output=$(
	./65_hopper_fp8_warp_specialized_distributed_gemm \
	    --m="$m" \
	    --n="$n" \
	    --k="$k" \
	    --raster="$raster" \
	    --swizzle="$swizzle" 2>&1)

    # Capture and return the exit status
    local exit_status=$?
    if [ $exit_status -ne 0 ]; then
        #echo "Error: Command failed with exit status $exit_status"
        return $exit_status
    fi
    
    local tflops
    tflops=$(get_tflops "$output")
    echo "$m,$n,$k,$raster,$swizzle,$tflops"
    return 0
}


for M in 112 3584; do
    for raster in N M H; do
        for swizzle in 1 2 4 8; do
            results=$(run_gemm "$M" 2560 8192 "$raster" "$swizzle")
            echo "$results"
        done
    done
done

for M in 512 3584; do
    for raster in N M H; do
        for swizzle in 1 2 4 8; do
            results=$(run_gemm "$M" 8192 2048 "$raster" "$swizzle")
            echo "$results"
        done
    done
done

for M in 112 3584; do
    for raster in N M H; do
        for swizzle in 1 2 4 8; do
            results=$(run_gemm "$M" 14336 8192 "$raster" "$swizzle")
            echo "$results"
        done
    done
done

for M in 512 3584; do
    for raster in N M H; do
        for swizzle in 1 2 4 8; do
            results=$(run_gemm "$M" 28672 2048 "$raster" "$swizzle")
            echo "$results"
        done
    done
done
