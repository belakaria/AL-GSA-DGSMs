#!/bin/bash

TEST_PROBLEM="$1"
METHOD="$2"
NITER="$3"
NINIT="$4"
REPEAT="$5"

mkdir -p "results/"$TEST_PROBLEM"/"
OUT="results/"$TEST_PROBLEM"/"$TEST_PROBLEM"_"$METHOD"_"$NITER
for ((SEED = 0 ; SEED < REPEAT; SEED++)); do
    echo "$TEST_PROBLEM $METHOD $SEED"
    FILEPATH=${OUT}"_"${SEED}".pickle"
    echo $FILEPATH
    if [ -f "$FILEPATH" ]; then
        echo "File $FILEPATH exists, skipping the command."
    else
        python run.py $TEST_PROBLEM $METHOD $NITER $NINIT $SEED 1> ""$OUT"""_"$SEED"_stdout.txt" 2> ""$OUT"""_"$SEED"_stderr.txt";
    fi
done
wait
