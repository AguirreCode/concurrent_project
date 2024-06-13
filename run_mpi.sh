#!/bin/bash

# Define the file paths
FILE1=./data/E_coli.fna
FILE2=./data/Salmonella.fna
RESULTS_FILE=results/mpi/results_mpi.txt

# Ensure the results file is empty before starting
> $RESULTS_FILE

# Define the number of processes to test
NUM_PROCESSES=(1 2 3 4)

# Loop over the number of processes and run the Python script
for NUM in "${NUM_PROCESSES[@]}"; do
    echo "Running with $NUM processes..."
    mpiexec -n $NUM python3 index.py --file1=$FILE1 --file2=$FILE2 --mpi --num_processes=$NUM --results_file=$RESULTS_FILE
    echo "Finished running with $NUM processes."
done

# After the loop, gather the results and generate performance graphs
echo "Generating performance graphs..."

# Parse the results file to extract times, accelerations, and efficiencies
TIMES=($(grep 'Tiempo de ejecución mpi' $RESULTS_FILE | awk '{print $NF}'))
ACCELERATIONS=($(grep 'Aceleración con' $RESULTS_FILE | awk '{print $NF}'))
EFFICIENCIES=($(grep 'Eficiencia con' $RESULTS_FILE | awk '{print $NF}'))

# Add default values for 1 process if not present
if [ ${#ACCELERATIONS[@]} -lt ${#NUM_PROCESSES[@]} ]; then
    ACCELERATIONS=(1.0 "${ACCELERATIONS[@]}")
    EFFICIENCIES=(1.0 "${EFFICIENCIES[@]}")
fi

# Debugging: print the arrays to check their contents
echo "NUM_PROCESSES: ${NUM_PROCESSES[@]}"
echo "TIMES: ${TIMES[@]}"
echo "ACCELERATIONS: ${ACCELERATIONS[@]}"
echo "EFFICIENCIES: ${EFFICIENCIES[@]}"

# Ensure all arrays have the same length
length=${#NUM_PROCESSES[@]}
TIMES=("${TIMES[@]:0:$length}")
ACCELERATIONS=("${ACCELERATIONS[@]:0:$length}")
EFFICIENCIES=("${EFFICIENCIES[@]:0:$length}")

# Convert arrays to comma-separated strings for Python
NUM_PROCESSES_STR=$(printf ",%s" "${NUM_PROCESSES[@]}")
NUM_PROCESSES_STR=${NUM_PROCESSES_STR:1}
TIMES_STR=$(printf ",%s" "${TIMES[@]}")
TIMES_STR=${TIMES_STR:1}
ACCELERATIONS_STR=$(printf ",%s" "${ACCELERATIONS[@]}")
ACCELERATIONS_STR=${ACCELERATIONS_STR:1}
EFFICIENCIES_STR=$(printf ",%s" "${EFFICIENCIES[@]}")
EFFICIENCIES_STR=${EFFICIENCIES_STR:1}

# Plot the performance metrics
python3 <<EOF
import matplotlib.pyplot as plt

num_threads = [${NUM_PROCESSES_STR}]
times = [${TIMES_STR}]
accelerations = [${ACCELERATIONS_STR}]
efficiencies = [${EFFICIENCIES_STR}]

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(num_threads, times, marker='o')
plt.xlabel("Número de procesadores")
plt.ylabel("Tiempo")

plt.subplot(1, 2, 2)
plt.plot(num_threads, accelerations, label="Aceleración", marker='o')
plt.plot(num_threads, efficiencies, label="Eficiencia", marker='o')
plt.xlabel("Número de procesadores")
plt.ylabel("Aceleración y Eficiencia")
plt.legend()

plt.savefig("images/images_mpi/metrics/graficasMPI.png")
plt.show()
EOF

# ejecutar programa mpi con:
# dar permisos chmod +x ./run_mpi.sh
# bash ./run_mpi.sh
