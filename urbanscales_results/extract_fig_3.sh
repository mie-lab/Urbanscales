#!/bin/bash

for i in {0..6}; do
    grep "RECURRENT recurrent-" RECURRENTcity"$i".csv > RECURRENTFigure3city"$i".csv
done
for i in {0..6}; do
    grep "NON-RECURRENT-MMM non-recurrent-mmm-" NONRECURRENTcity"$i".csv > NONRECURRENTFigure3city"$i".csv
done


