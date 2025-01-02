#!/bin/bash

for i in {2..8}; do
    grep "SHAP--" RECURRENTcity"$i".csv > RECURRENTFigure2city"$i".csv
done
for i in {2..8}; do
    grep "SHAP--" NONRECURRENTcity"$i".csv > NONRECURRENTFigure2city"$i".csv
done


