#!/usr/bin/env bash
export SCHRODINGER="$SCHRODPATH"
"${SCHRODINGER}/ligprep" -inp ligprep.inp -LOCAL -HOST "localhost" -NJOBS 1 -WAIT
exit