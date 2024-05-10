#!/bin/bash
for (( c=1; c<=10; c++ ))
do
   echo "Welcome $c times"
   python performance_test.py --model deeplabv3 --backbone resnet50
done
