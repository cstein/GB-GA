#!/usr/bin/env bash
# This script will always generate a subset.smi file based
# on either the results from LigPrep or RDKit. If a subset.smi
# file is not present an error is thrown
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
