#!/bin/bash
input=$1
first=$2
last=$3
sigma=$4
output=$5
outputSub=$6

#TODO add this parameters to the list of possible parameters
wx=15
px=8
np=64
rank=16
a=0.9
occ=2.3

# Generate the noisy data
./nlkalman -i $input -nisy %06d_noisy.tiff -f $first -l $last -sigma $sigma -px 0 

# Compute the optical flow (using the noisy data)
./tvl1flow.sh %06d_noisy.tiff $first $last tvl1_%06d.flo

# Apply NL-Kalman
./nlkalman -i %06d_noisy.tiff -deno %06d_deno.tiff -sub %06d_sub.tiff -f $first -l $last -sigma $sigma -of tvl1_%06d.flo -wx $wx -px $px -np $np -r $rank -a $a -occ $occ -add_noise false

# Apply DCTdenoising as post-processing on the regular and the subpixelic results
./dct.sh %06d_deno.tiff $first $last $output 3 
./dct.sh %06d_sub.tiff $first $last ${outputSub} 3 
