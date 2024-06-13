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

# Parse the results file to extract times, accelerations, and efficiencies
TIMES=($(grep 'Tiempo de ejecución PyCUDA' $RESULTS_FILE | awk '{print $NF}'))
ACCELERATIONS=($(grep 'Aceleración con' $RESULTS_FILE | awk '{print $NF}'))
EFFICIENCIES=($(grep 'Eficiencia con' $RESULTS_FILE | awk '{print $NF}'))

# Add default values for base case if not present
if [ ${#ACCELERATIONS[@]} -lt ${#SEQUENCE_LENGTHS[@]} ]; then
    ACCELERATIONS=(1.0 "${ACCELERATIONS[@]}")
    EFFICIENCIES=(1.0 "${EFFICIENCIES[@]}")
fi

# Debugging: print the arrays to check their contents
echo "SEQUENCE_LENGTHS: ${SEQUENCE_LENGTHS[@]}"
echo "TIMES: ${TIMES[@]}"
echo "ACCELERATIONS: ${ACCELERATIONS[@]}"
echo "EFFICIENCIES: ${EFFICIENCIES[@]}"

# Ensure all arrays have the same length
length=${#SEQUENCE_LENGTHS[@]}
TIMES=("${TIMES[@]:0:$length}")
ACCELERATIONS=("${ACCELERATIONS[@]:0:$length}")
EFFICIENCIES=("${EFFICIENCIES[@]:0:$length}")

# Convert bash arrays to JSON arrays for Python
SEQUENCE_LENGTHS_JSON=$(printf '%s\n' "${SEQUENCE_LENGTHS[@]}" | jq -c -s '.')
TIMES_JSON=$(printf '%s\n' "${TIMES[@]}" | jq -c -s '.')
ACCELERATIONS_JSON=$(printf '%s\n' "${ACCELERATIONS[@]}" | jq -c -s '.')
EFFICIENCIES_JSON=$(printf '%s\n' "${EFFICIENCIES[@]}" | jq -c -s '.')

# Plot the performance metrics
python3 <<EOF
import matplotlib.pyplot as plt
import json

sequence_lengths = json.loads('$SEQUENCE_LENGTHS_JSON')
times = json.loads('$TIMES_JSON')
accelerations = json.loads('$ACCELERATIONS_JSON')
efficiencies = json.loads('$EFFICIENCIES_JSON')

plt.figure(figsize=(10, 10))

# Plot execution times
plt.subplot(1, 2, 1)
plt.plot(sequence_lengths, times, marker='o')
plt.xlabel("Longitud de Secuencia")
plt.ylabel("Tiempo de Ejecución (s)")
plt.title("Tiempo de Ejecución vs Longitud de Secuencia")

# Plot speedup and efficiency
plt.subplot(1, 2, 2)
plt.plot(sequence_lengths, accelerations, marker='o', label='Aceleración')
plt.plot(sequence_lengths, efficiencies, marker='x', label='Eficiencia')
plt.xlabel("Longitud de Secuencia")
plt.ylabel("Aceleración y Eficiencia")
plt.legend()
plt.title("Aceleración y Eficiencia vs Longitud de Secuencia")

plt.savefig("images/images_pycuda/metrics/graficas_pycuda.png")
plt.show()
EOF

# ejecutar programa con:
# dar permisos chmod +x ./run_cuda.sh
# bash ./run_cuda.sh
