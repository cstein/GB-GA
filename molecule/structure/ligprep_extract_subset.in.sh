#!/usr/bin/env bash
export SCHRODINGER="${SHRODPATH}"
$SCHRODINGER/utilities/maesubset -n "${INDICES}" "${FILENAME}" > subset.mae

