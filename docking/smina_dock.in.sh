#!/usr/bin/env bash
export SMINAPATH="$SMINA"
export SMINAEXE="$EXE"
"${SMINAPATH}/${SMINAEXE}" -r ${RECEPTOR} -l ${LIGAND} --num_modes=${NUMMODES} --center_x ${CX} --center_y ${CY} --center_z ${CZ} --size_x ${BOXSIZE} --size_y ${BOXSIZE} --size_z ${BOXSIZE} --cpu ${NCPUS} -o ${BASENAME}.pdbqt --log ${BASENAME}.log
exit
