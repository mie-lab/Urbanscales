#!/bin/bash

for i in 2 3 4 5 6 7 8; do
   nohup python Pipeline.py "$i" "NON-RECURRENT-MMM" > "NONRECURRENT$i.csv" &
   sleep 10
done

for i in 2 3 4 5 6 7 8; do
   nohup python Pipeline.py "$i" "RECURRENT" > "RECURRENT$i.csv" &
   sleep 10
done


