#!/bin/bash
for i in 1 2 3; do
  python train.py @experiments/base.txt @experiments/$i.txt @experiments/steps.txt
done
