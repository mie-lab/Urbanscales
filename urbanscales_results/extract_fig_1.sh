#!/bin/bash

for i in {0..6}; do
    grep "R2" RECURRENTcity"$i".csv > RECURRENTFigure1city"$i".csv
done
for i in {0..6}; do
    grep "R2" NONRECURRENTcity"$i".csv > NONRECURRENTFigure1city"$i".csv
done


