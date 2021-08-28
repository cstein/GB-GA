#!/usr/bin/env bash
export SCHRODINGER="$SCHRODPATH"
"${SCHRODINGER}/ligprep" -inp ligprep.inp -LOCAL -HOST localhost:$NCPUS -NJOBS $NCPUS -WAIT
exit