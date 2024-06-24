#!/bin/bash

## Example run  bash ftnet/helper/run_performace_10_times.sh
PYTHON_SCRIPT="python -m ftnet.helper.performance_test"

for (( c=1; c<=10; c++ ))
do
   echo "Running iteration $c"
   $PYTHON_SCRIPT -c ./ftnet/cfg/infer.toml
done
