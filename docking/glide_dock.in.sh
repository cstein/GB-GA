#!/usr/bin/env bash
export SCHRODINGER="$SCHRODPATH"
"${SCHRODINGER}/glide" $GLIDE_IN -OVERWRITE -LOCAL -HOST localhost:1 -NJOBS 1 -WAIT
exit
