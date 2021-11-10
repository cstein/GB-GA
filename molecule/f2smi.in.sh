#!/usr/bin/env bash
# Generates a `subset.smi` file based on output from
# either RDKit or LigPrep.
# If a subset.smi file is not present, an error is thrown
export SCHRODINGER="$SCHRODPATH"

# LigPrep
if [ -e subset.mae ]
then
  $SCHRODINGER/utilities/structconvert subset.mae subset.smi
fi

# RDKit
if [ -e out.sdf ]
then
  $SCHRODINGER/utilities/structconvert out.sdf subset.smi
fi

if [ ! -e subset.smi ]
then
  exit -1
fi
exit
