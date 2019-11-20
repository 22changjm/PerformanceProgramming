#!/bin/bash

# clean slate & make targets
make clean >/dev/null 2>&1
make benchmark
make baseline

echo ''
echo "[*] DONE MAKING"

# collect new accuracy and execution time
ACCU_NEW="$(./benchmark benchmark | grep "accuracy" | awk '//{print $1}')"
TIME_NEW="$(./benchmark benchmark | grep "microseconds" | awk '//{print $1}')"

echo "[*] RAN NEW BENCHMARK"

# collect old accuracy and execution time
ACCU_OLD="$(./benchmark_baseline benchmark | grep "accuracy" | awk '//{print $1}')"
TIME_OLD="$(./benchmark_baseline benchmark | grep "microseconds" | awk '//{print $1}')"

echo "[*] RAN OLD BENCHMARK"

# calculate the times speedup
SPEEDUP=$(echo "${TIME_OLD} / ${TIME_NEW}" | bc -l)

echo "[*] CALCULATED SPEEDUP"

# print results
echo ''
echo "==========================================="
printf "| VERSION |   ACCURACY    |     TIME      |\n"
echo "==========================================="
printf "|     OLD | %13s | %13s |\n" $ACCU_OLD $TIME_OLD
echo "-------------------------------------------"
printf "|     NEW | %13s | %13s |\n" $ACCU_NEW $TIME_NEW
echo "==========================================="

echo ''
echo "Times Speedup: ${SPEEDUP}"
echo ''

# final cleanup
make clean >/dev/null 2>&1
