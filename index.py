from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
from multiprocessing import Pool, cpu_count, shared_memory
import multiprocessing as mp
import cv2
from tqdm import tqdm
import os
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

os.environ['RDMAV_FORK_SAFE'] = '1'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/home/alexander/.local/lib/python3.10/site-packages/cv2/qt/plugins'

def read_fasta(file_name):
    sequences = []
    for record in SeqIO.parse(file_name, "fasta"):
        sequences.append(str(record.seq))
    return "".join(sequences)

def draw_dotplot(dotplot, fig_name='dotplot.svg'):
    plt.figure(figsize=(10, 10))
    plt.imshow(dotplot, cmap="Greys", aspect="auto")
    plt.xlabel("Secuencia 1")
    plt.ylabel("Secuencia 2")
    plt.savefig(fig_name)
    plt.show()

def dotplot_sequential(sequence1, sequence2):
    dotplot = np.empty((len(sequence1), len(sequence2)))    
    for i in tqdm(range(len(sequence1))):
        for j in range(len(sequence2)):
            if sequence1[i] == sequence2[j]:
                if i == j:
                    dotplot[i, j] = 1
                else:
                    dotplot[i, j] = 0.7
            else:
                dotplot[i, j] = 0
    return dotplot

"""def worker_multiprocessing(start_idx, end_idx, shm_name, shape, dtype, sequence1, sequence2):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    dotplot = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    
    for i in range(start_idx, end_idx):
        for j in range(len(sequence2)):
            if sequence1[i] == sequence2[j]:
                dotplot[i, j] = 1 if i == j else 0.7
            else:
                dotplot[i, j] = 0
    
    existing_shm.close()"""
def worker_multiprocessing(args):
    i, sequence1, sequence2 = args
    dotplot = []
    for j in range(len(sequence2)):
        if sequence1[i] == sequence2[j]:
            if i == j:
                dotplot.append(1)
            else:
                dotplot.append(0.7)
        else:
            dotplot.append(0)
    return dotplot

"""def parallel_multiprocessing_dotplot(sequence1, sequence2, threads=cpu_count(), chunk_size=500):
    shape = (len(sequence1), len(sequence2))
    dtype = np.float16

    shm = shared_memory.SharedMemory(create=True, size=np.prod(shape) * np.dtype(dtype).itemsize)
    dotplot = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    dotplot.fill(0)

    pool = Pool(processes=threads)

    indices = [(start_idx, min(start_idx + chunk_size, len(sequence1)), shm.name, shape, dtype, sequence1, sequence2)
               for start_idx in range(0, len(sequence1), chunk_size)]
    
    pool.starmap(worker_multiprocessing, indices)
    pool.close()
    pool.join()

    dotplot_copy = dotplot.copy()  # Copy the result to return
    shm.close()
    shm.unlink()

    return dotplot_copy"""
def parallel_multiprocessing_dotplot(sequence1, sequence2, threads=mp.cpu_count()):
    with mp.Pool(processes=threads) as pool:
        dotplot = pool.map(worker_multiprocessing, [
                           (i, sequence1, sequence2) for i in range(len(sequence1))])
    return dotplot

def save_results_to_file(results, file_name="images/results.txt"):
    with open(file_name, "a") as file:
        for result in results:
            file.write(str(result) + "\n")

def acceleration(times):
    return [times[0] / i for i in times]

def efficiency(accelerations, num_threads):
    return [accelerations[i] / num_threads[i] for i in range(len(num_threads))]

def draw_graphic(times, accelerations, efficiencies, num_threads, output_path):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(num_threads, times)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Tiempo")
    plt.subplot(1, 2, 2)
    plt.plot(num_threads, accelerations)
    plt.plot(num_threads, efficiencies)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Aceleración y Eficiencia")
    plt.legend(["Aceleración", "Eficiencia"])
    plt.savefig(output_path)

def draw_graphic_mpi(times, accelerations, efficiencies, num_threads):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.plot(num_threads, times)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Tiempo")
    plt.subplot(1, 2, 2)
    plt.plot(num_threads, accelerations)
    plt.plot(num_threads, efficiencies)
    plt.xlabel("Número de procesadores")
    plt.ylabel("Aceleración y Eficiencia")
    plt.legend(["Aceleración", "Eficiencia"])
    plt.savefig("images/graficasMPI.png")

def parallel_mpi_dotplot(sequence_1, sequence_2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    chunks = np.array_split(range(len(sequence_1)), size)

    dotplot = np.empty([len(chunks[rank]), len(sequence_2)], dtype=np.float16)

    for i in tqdm(range(len(chunks[rank]))):
        for j in range(len(sequence_2)):
            if sequence_1[chunks[rank][i]] == sequence_2[j]:
                if i == j:
                    dotplot[i, j] = np.float16(1.0)
                else:
                    dotplot[i, j] = np.float16(0.6)
            else:
                dotplot[i, j] = np.float16(0.0)

    dotplot = comm.gather(dotplot, root=0)

    if rank == 0:
        merged_data = np.vstack(dotplot)
        return merged_data
    
def parallel_pycuda_dotplot(sequence1, sequence2):
    len1, len2 = len(sequence1), len(sequence2)
    dotplot = np.zeros((len1, len2), dtype=np.float32)
    
    # Compile the CUDA kernel
    with open('dotplot_kernel.cu', 'r') as f:
        kernel_code = f.read()
    mod = SourceModule(kernel_code)
    dotplot_kernel = mod.get_function("dotplot_kernel")
    
    # Allocate GPU memory
    sequence1_gpu = cuda.mem_alloc(len1 * np.dtype(np.uint8).itemsize)
    sequence2_gpu = cuda.mem_alloc(len2 * np.dtype(np.uint8).itemsize)
    dotplot_gpu = cuda.mem_alloc(dotplot.nbytes)
    
    # Copy data to GPU
    cuda.memcpy_htod(sequence1_gpu, np.frombuffer(sequence1.encode(), dtype=np.uint8))
    cuda.memcpy_htod(sequence2_gpu, np.frombuffer(sequence2.encode(), dtype=np.uint8))
    cuda.memcpy_htod(dotplot_gpu, dotplot)
    
    # Define the block and grid dimensions
    block = (16, 16, 1)
    grid = (int((len1 + block[0] - 1) / block[0]), int((len2 + block[1] - 1) / block[1]), 1)
    
    # Execute the kernel
    dotplot_kernel(sequence1_gpu, sequence2_gpu, np.int32(len1), np.int32(len2), dotplot_gpu, block=block, grid=grid)
    
    # Copy the result back to the host
    cuda.memcpy_dtoh(dotplot, dotplot_gpu)
    return dotplot

def apply_filter(matrix, output_path):
    # Asegúrate de que la matriz es del tipo correcto (float32 o uint8)
    if matrix.dtype != np.float32 and matrix.dtype != np.uint8:
        matrix = matrix.astype(np.float32)
    
    # Kernel para detectar diagonales
    kernel_diagonales = np.array([[1, 0, -1],
                                  [0, 1, 0],
                                  [-1, 0, 1]], dtype=np.float32)

    # Aplica el filtro
    filtered_matrix = cv2.filter2D(matrix, -1, kernel_diagonales)
    
    # Normaliza los valores de la matriz para asegurarte de que están en el rango adecuado para la visualización
    filtered_matrix = cv2.normalize(filtered_matrix, None, 0, 255, cv2.NORM_MINMAX)
    
    # Guarda la imagen filtrada como uint8
    cv2.imwrite(output_path, filtered_matrix.astype(np.uint8))

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--file1', dest='file1', type=str,
                        default=None, help='Query sequence in FASTA format')
    parser.add_argument('--file2', dest='file2', type=str,
                        default=None, help='Subject sequence in FASTA format')

    parser.add_argument('--sequential', action='store_true',
                        help='Ejecutar en modo secuencial')
    parser.add_argument('--multiprocessing', action='store_true',
                        help='Ejecutar utilizando multiprocessing')
    parser.add_argument('--mpi', action='store_true',
                        help='Ejecutar utilizando mpi4py')
    parser.add_argument('--num_processes', dest='num_processes', type=int,
                        default=1, help='Número de procesos para la opción MPI')
    parser.add_argument('--results_file', dest='results_file', type=str,
                        default='results_mpi.txt', help='File to save results')
    parser.add_argument('--pycuda', action='store_true', help='Ejecutar utilizando PyCUDA')
    parser.add_argument('--seq_length', dest='seq_length', type=int, default=6000, help='Longitud de las secuencias a procesar')

    args = parser.parse_args()

    if rank == 0:
        chargeFilesStart = time.time()
        file_path_1 = args.file1
        file_path_2 = args.file2

        num_threads_array = args.num_processes
        seq_length = args.seq_length

        try:
            merged_sequence_1 = read_fasta(file_path_1)
            merged_sequence_2 = read_fasta(file_path_2)

        except FileNotFoundError as e:
            print("Archivo no encontrado, verifique la ruta")
            exit(1)

        Secuencia1 = merged_sequence_1[0:30000]
        Secuencia2 = merged_sequence_2[0:30000]
        chargeFilesFinish = time.time()

        save_results_to_file([f"Tiempo de carga de los archivos: {chargeFilesFinish - chargeFilesStart}"],
                             file_name="results/results_charge_files.txt")

        dotplot = np.empty([len(Secuencia1), len(Secuencia2)])
        results_print = []
        results_print_mpi = []
        times_multiprocessing = []
        times_mpi = []

    else:
        num_threads_array = None
        Secuencia1 = None
        Secuencia2 = None

    num_threads_array = comm.bcast(num_threads_array, root=0)
    Secuencia1 = comm.bcast(Secuencia1, root=0)
    Secuencia2 = comm.bcast(Secuencia2, root=0)

    if args.sequential and rank == 0:
        start_secuencial = time.time()
        dotplotSequential = dotplot_sequential(Secuencia1, Secuencia2)
        results_print.append(
            f"Tiempo de ejecución secuencial: {time.time() - start_secuencial}")
        draw_dotplot(dotplotSequential[:600, :600],
                     fig_name="images/images_sequential/dotplot/dotplot_secuencial.png")
        path_image = 'images/images_filter/dotplot_filter_sequential.png'
        apply_filter(dotplotSequential[:600, :600], path_image)
        save_results_to_file(results_print, file_name="results/sequential/results_sequential.txt")

    if args.multiprocessing and rank == 0:
        num_threads = [1, 2, 4, 8]
        for num_thread in num_threads:
            start_time = time.time()
            dotplotMultiprocessing = np.array(
                parallel_multiprocessing_dotplot(Secuencia1, Secuencia2, num_thread))
            times_multiprocessing.append(time.time() - start_time)
            results_print.append(
                f"Tiempo de ejecución multiprocessing con {num_thread} hilos: {time.time() - start_time}")
            
        # Aceleración
        accelerations = acceleration(times_multiprocessing)
        for i in range(len(accelerations)):
            results_print.append(
                f"Aceleración con {num_threads[i]} hilos: {accelerations[i]}")

        # Eficiencia
        efficiencies = efficiency(accelerations, num_threads)
        for i in range(len(efficiencies)):
            results_print.append(
                f"Eficiencia con {num_threads[i]} hilos: {efficiencies[i]}")

        save_results_to_file(results_print, file_name="results/multiprocessing/results_multiprocessing.txt")
        draw_graphic(times_multiprocessing, accelerations, efficiencies, num_threads, "images/images_multiprocessing/metrics/graficasMultiprocessing.png")
        draw_dotplot(dotplotMultiprocessing[:600, :600],
                     fig_name='images/images_multiprocessing/dotplot/dotplot_multiprocessing.png')
        
        path_image = 'images/images_filter/dotplot_filter_multiprocessing.png'  
        apply_filter(dotplotMultiprocessing[:600, :600], path_image)

    if args.mpi:
        if rank == 0:
            start_time = time.time()
        dotplot_mpi = parallel_mpi_dotplot(Secuencia1, Secuencia2)
        if rank == 0:
            elapsed_time = time.time() - start_time
            results_print_mpi = [f"Tiempo de ejecución mpi con {args.num_processes} procesos: {elapsed_time}"]

            # Calcular aceleraciones y eficiencias
            with open(args.results_file, 'r') as f:
                previous_times = [float(line.split()[-1]) for line in f if 'Tiempo de ejecución mpi' in line]
            
            if not previous_times:
                # Para 1 proceso, aceleración y eficiencia deben ser 1
                accelerations = [1.0]
                efficiencies = [1.0]
            else:
                base_time = previous_times[0]
                accelerations = [base_time / t for t in previous_times + [elapsed_time]]
                efficiencies = [accelerations[i] / (i + 1) for i in range(len(accelerations))]

            results_print_mpi.append(f"Aceleración con {args.num_processes} procesos: {accelerations[-1]}")
            results_print_mpi.append(f"Eficiencia con {args.num_processes} procesos: {efficiencies[-1]}")

            save_results_to_file(results_print_mpi, file_name=args.results_file)
            draw_dotplot(dotplot_mpi[:10000, :10000], fig_name=f'images/images_mpi/dotplot/dotplot_mpi_{args.num_processes}_processes.png')
            path_image = 'images/images_filter/dotplot_filter_mpi.png'
            apply_filter(dotplot_mpi[:10000, :10000], path_image)

    if args.pycuda:
        start_time = time.time()
        dotplot_pycuda = parallel_pycuda_dotplot(Secuencia1, Secuencia2)
        elapsed_time = time.time() - start_time
        results_print_pycuda = [f"Tiempo de ejecución PyCUDA: {elapsed_time}"]
        
        # Calcular aceleraciones y eficiencias
        results_file = "results/cuda/results_pycuda.txt"
        with open(results_file, 'r') as f:
            previous_times = [float(line.split()[-1]) for line in f if 'Tiempo de ejecución PyCUDA' in line]

        if not previous_times:
            previous_times = [elapsed_time]  # Add the initial elapsed time if the list is empty

        base_time = previous_times[0]
        accelerations = [base_time / t for t in previous_times + [elapsed_time]]
        efficiencies = [accelerations[i] / (i + 1) for i in range(len(accelerations))]

        results_print_pycuda.append(f"Aceleración con {args.seq_length} longitud de secuencia: {accelerations[-1]}")
        results_print_pycuda.append(f"Eficiencia con {args.seq_length} longitud de secuencia: {efficiencies[-1]}")

        save_results_to_file(results_print_pycuda, file_name=results_file)
        draw_dotplot(dotplot_pycuda[:10000, :10000], fig_name=f'images/images_pycuda/dotplot/dotplot_pycuda_{args.seq_length}.png')
        path_image = 'images/images_filter/dotplot_filter_cuda.png'
        apply_filter(dotplot_pycuda[:10000, :10000], path_image)

if __name__ == "__main__":
    main()
