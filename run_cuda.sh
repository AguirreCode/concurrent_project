#!/bin/bash

# Define the file paths
FILE1=./data/E_coli.fna
FILE2=./data/Salmonella.fna
RESULTS_FILE=results/cuda/results_pycuda.txt

# Ensure the results file is empty before starting
> $RESULTS_FILE

# Define different sequence lengths to test
SEQUENCE_LENGTHS=(10000 15000 20000 25000 30000)

# Loop over the sequence lengths and run the Python script
for LENGTH in "${SEQUENCE_LENGTHS[@]}"; do
    echo "Running with sequence length $LENGTH..."
    python3 index.py --file1=$FILE1 --file2=$FILE2 --pycuda --seq_length=$LENGTH
    echo "Finished running with sequence length $LENGTH."
done

# After the loop, gather the results and generate performance graphs
echo "Generating performance graphs..."

# Parse the results file to extract times
TIMES=($(grep 'Tiempo de ejecución PyCUDA' $RESULTS_FILE | awk '{print $NF}'))

# Convert bash arrays to JSON arrays for Python
SEQUENCE_LENGTHS_JSON=$(printf '%s\n' "${SEQUENCE_LENGTHS[@]}" | jq -c -s '.')
TIMES_JSON=$(printf '%s\n' "${TIMES[@]}" | jq -c -s '.')

# Plot the performance metrics
python3 <<EOF
import matplotlib.pyplot as plt
import json

sequence_lengths = json.loads('$SEQUENCE_LENGTHS_JSON')
times = json.loads('$TIMES_JSON')

plt.figure(figsize=(10, 10))

# Plot execution times
plt.subplot(1, 2, 1)
plt.plot(sequence_lengths, times, marker='o')
plt.xlabel("Longitud de Secuencia")
plt.ylabel("Tiempo de Ejecución (s)")
plt.title("Tiempo de Ejecución vs Longitud de Secuencia")

# Calculate and plot speedup and efficiency
base_time = times[0]
speedups = [base_time / t for t in times]
efficiencies = [s / (l / sequence_lengths[0]) for s, l in zip(speedups, sequence_lengths)]

plt.subplot(1, 2, 2)
plt.plot(sequence_lengths, speedups, marker='o', label='Aceleración')
plt.plot(sequence_lengths, efficiencies, marker='x', label='Eficiencia')
plt.xlabel("Longitud de Secuencia")
plt.ylabel("Aceleración y Eficiencia")
plt.legend()
plt.title("Aceleración y Eficiencia vs Longitud de Secuencia")

plt.savefig("images/images_pycuda/metrics/graficas_pycuda.png")
plt.show()
EOF


# ejecutar programa con:
# bash ./run_cuda.sh
