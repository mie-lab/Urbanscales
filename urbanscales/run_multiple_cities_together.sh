#!/bin/bash

for i in {2..9}; do
  nohup python Pipeline.py "$i" > "city$i.csv" &
  sleep 10
done

