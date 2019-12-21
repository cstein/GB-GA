#!/usr/bin/env bash
export SCHRODINGER="$SCHRODPATH"
"${SCHRODINGER}/glide" $GLIDE_IN -OVERWRITE -LOCAL -HOST localhost:$NCPUS -NJOBS $NCPUS -WAIT
exit
