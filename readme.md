# Proyecto Final: Análisis de Rendimiento de Dotplot
## Secuencial vs Paralelización
Este proyecto tiene como objetivo implementar y analizar el rendimiento de tres métodos diferentes para realizar un dotplot, una técnica comúnmente utilizada en bioinformática para comparar secuencias de ADN o proteínas.

## Prerrequisitos (usar alguna distribución unix)
El proyecto fue desarrollado utilizando Python 3.10.12 y soporta computación paralela mediante las bibliotecas multiprocessing, mpi4py y pycuda. Requiere como parámetros de entrada las secuencias de referencia y de consulta en formato .fna, que deben especificarse en la línea de comandos para calcular el dotplot.

## Instalación
Primero, asegúrate de tener Python instalado en tu sistema. Luego, instala las bibliotecas necesarias ejecutando los siguientes comandos:

pip install numpy
pip install matplotlib
pip install mpi4py
pip install biopython
pip install opencv-python
pip install tqdm==2.2.3
pip install pycuda==2024.1

## Finalmente, clona el repositorio:
### Clonar repositorio
git clone https://github.com/AguirreCode/concurrent_project.git

### Ejecución

#### Para ejecutar el programa en modo secuencial, utiliza el siguiente comando:
#### Copiar código
python3 index.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --sequential

#### Para ejecutar el programa utilizando multiprocessing, utiliza el siguiente comando:
#### Copiar código
python3 index.py --file1=./data/E_coli.fna --file2=./data/Salmonella.fna --multiprocessing

#### Para ejecutar el programa utilizando mpi4py, utiliza el siguiente comando:
#### Copiar código
bash ./run_mpi.sh

#### Para ejecutar el programa utilizando pycuda, utiliza el siguiente comando:
#### Copiar código
bash ./run_cuda.sh

## Resultados y Análisis
El proyecto incluye scripts para medir y comparar los tiempos de ejecución de los diferentes métodos. Los resultados se almacenan en archivos de texto y se generan gráficos de rendimiento para visualizar la aceleración y la eficiencia de cada método.

## Contribuciones
Este proyecto fue desarrollado por Jhair Alexander Peña Aguirre, Dany Orlando Guancha Tarapues y Jefferson Henao Cano como parte de la asignatura de Programación Concurrente y Distribuida.