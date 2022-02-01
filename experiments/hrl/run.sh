#!/bin/bash
for i in 0 1 2; do
  python train.py @experiments/base.txt @experiments/$i.txt @experiments/steps.txt
done
