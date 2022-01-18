#!/bin/bash
for i in 8; do
  python ddpg_hrl.py -e $i
done
