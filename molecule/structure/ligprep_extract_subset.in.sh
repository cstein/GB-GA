#!/usr/bin/env bash
export SCHRODINGER="/opt/schrodinger/suites2021-2"
$SCHRODINGER/utilities/maesubset -n "${INDICES}" "${FILENAME}" > subset.mae

