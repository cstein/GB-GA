#!/usr/bin/env bash
export SCHRODINGER="${SCHRODPATH}"
$SCHRODINGER/utilities/maesubset -n "${INDICES}" "${FILENAME}" > subset.mae

