import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2


def acceleration(times):
    sequential_time = times[0]
    return [sequential_time / t for t in times]

def efficiency(accelerations, num_threads):
    return [acc / n for acc, n in zip(accelerations, num_threads)]

def draw_graphic(times, accelerations, efficiencies, num_threads, fig_name):
    fig, ax = plt.subplots(3, 1, figsize=(10, 15))

    ax[0].plot(num_threads, times, marker='o')
    ax[0].set_title('Execution Time')
    ax[0].set_xlabel('Number of Threads')
    ax[0].set_ylabel('Time (s)')

    ax[1].plot(num_threads, accelerations, marker='o')
    ax[1].set_title('Acceleration')
    ax[1].set_xlabel('Number of Threads')
    ax[1].set_ylabel('Acceleration')

    ax[2].plot(num_threads, efficiencies, marker='o')
    ax[2].set_title('Efficiency')
    ax[2].set_xlabel('Number of Threads')
    ax[2].set_ylabel('Efficiency')

    plt.tight_layout()
    plt.savefig(fig_name)

def draw_dotplot(dotplot, fig_name):
    plt.imshow(dotplot, cmap='gray')
    plt.title('Dot Plot')
    plt.savefig(fig_name)

# def apply_filter(dotplot, path_image):
#     # Apply your filter here
#     # For example, a simple threshold filter
#     filtered_dotplot = dotplot > 0.5
#     plt.imshow(filtered_dotplot, cmap='gray')
#     plt.title('Filtered Dot Plot')
#     plt.savefig(path_image)

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
    
    # Guarda la imagen filtrada
    cv2.imwrite(output_path, filtered_matrix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphics after MPI execution.")
    parser.add_argument('--num_processes', type=int, required=True, help="Number of processes tested")
    parser.add_argument('--results_file', type=str, required=True, help="Path to the results file")
    args = parser.parse_args()

    times_mpi = []
    with open(args.results_file, "r") as f:
        for line in f:
            if "Tiempo de ejecución mpi" in line:
                time = float(line.strip().split(": ")[1])
                times_mpi.append(time)

    num_threads = list(range(1, args.num_processes + 1))
    accelerations = acceleration(times_mpi)
    efficiencies = efficiency(accelerations, num_threads)
    
    draw_graphic(times_mpi, accelerations, efficiencies, num_threads, "images/images_mpi/graficasMPI.png")

    for num in num_threads:
        dotplot_mpi = np.load(f'dotplot_mpi_{num}.npy')
        draw_dotplot(dotplot_mpi[:600, :600], fig_name=f'images/images_mpi/dotplot_mpi_{num}.png')
        path_image = f'images/images_filter/dotplot_filter_mpi_{num}.png'
        apply_filter(dotplot_mpi[:600, :600], path_image)
