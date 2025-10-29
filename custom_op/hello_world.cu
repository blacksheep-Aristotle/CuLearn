#include<stdio.h>

void CPUhelloworld(void) {
        printf("Hello from CPU.\n");
}
// kernel 函数
__global__ void GPUhelloworld(void) {
        printf("Hello from GPU.\n");
}

int main(void) {
        CPUhelloworld();
        GPUhelloworld<<<1, 1>>>();
        cudaDeviceSynchronize();
        return 0;
}
