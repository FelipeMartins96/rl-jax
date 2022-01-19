#!/bin/bash
for i in 0 1; do
  python ddpg_baseline.py -e $i
done
