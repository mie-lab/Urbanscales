#!/bin/bash

for i in {2..8}; do
    grep "config.CONGESTION_TYPE" RECURRENTcity"$i".csv > RECURRENTFigure3city"$i".csv
done
for i in {2..8}; do
    grep "config.CONGESTION_TYPE" NONRECURRENTcity"$i".csv > NONRECURRENTFigure3city"$i".csv
done


