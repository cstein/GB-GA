#!/usr/bin/env bash
export SMINAPATH="/Users/css/Development/python/pysmina"
export SMINAEXE="smina.osx"
"${SMINAPATH}/${SMINAEXE}" -r ${RECEPTOR} -l ${LIGAND} --num_modes=${NUMMODES} --autobox_add ${AUTOBOXADD} --center_x ${CX} --center_y ${CY} --center_z ${CZ} --size_x 15 --size_y 15 --size_z 15 --cpu ${NCPUS} -o ${BASENAME}.pdbqt --log ${BASENAME}.log
exit
