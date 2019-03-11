#!/bin/bash

DCT_PATH=./dctdenoising

I=${1:-""}
F=${2:-1}
L=${3:-1}
O=${4:-""}
S=${5:-1}

for i in `seq $F $L`;
do
	$DCT_PATH $S `printf $I $i` `printf $O $i` -n 1; 
done
