#!/bin/bash
for i in 0 1; do
  python ddpg_hrl.py -e $i
done
