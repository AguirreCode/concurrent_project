extern "C" __global__ void dotplot_kernel(char* sequence1, char* sequence2, int len1, int len2, float* dotplot) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < len1 && j < len2) {
        if (sequence1[i] == sequence2[j]) {
            dotplot[i * len2 + j] = (i == j) ? 1.0 : 0.7;
        } else {
            dotplot[i * len2 + j] = 0.0;
        }
    }
}
