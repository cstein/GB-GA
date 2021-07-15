#!/usr/bin/env bash
export SCHRODINGER="$SCHRODPATH"
$SCHRODINGER/utilities/maesubset -n ${2} dock_pv.mae > ${1}.mae
$SCHRODINGER/utilities/sdconvert -imae ${1}.mae -osd ${1}.sd
