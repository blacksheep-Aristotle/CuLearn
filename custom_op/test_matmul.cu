#include<stdio.h>

__global__ void simpleMatmul(const float* A,const float* B,float* C,const int M,const int N,const int K){
//[M*K]@[K*N]
//one thread cal one data
	int row=blockIdx.y*blockDim.y+threadIdx.y;
	int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<M&&col<N){
		C[row*N+col]=0;
		int sum=0;
		for(int k=0;k<K;k++){
			sum+=A[row*k+k]*B[k*N+col];
		}
        C[row*N+col]=sum;
	}
}
__global__ void sharememMatmul(const float* A,const float* B,float* C,const int M,const int N,const int K){
//[M*K]@[K*N]
//one dims buffer
	__shared__ float a_tile[1][blockDim.y];
	__shared__ float b_tile[blockDim.y][1];

    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
	if(row<N&&col<N){
		int sum=0
		for(k=0;k<K;k+=blockDim.y){
			a_tile=A[row][k];
			b_tile=B[k][col];
			for(int i=0;i<blockDim.y;i++){
				sum+=a_tile[1][i]*b_tile[i][1];
			}
		}
		C[row*N+col]=sum;
	}
}

void cpuMatmul(const float* A,const float* B,float* C,const int M,const int N,const int K){

	for(int i=0;i<M;i++){
    	for(intj =0;j<N;j++){
        	for(k=0;k<K;k++){
            	C[i*N+j]+=A[i*k+k]*B[k*N+j];
        	}
    	}
   }

}
int main() {
    // 定义矩阵维度
    const int M = 1024;  // A矩阵的行数
    const int N = 1024;  // B矩阵的列数
    const int K = 1024;  // A矩阵的列数，B矩阵的行数
    
    // host内存分配，并初始化输入
    int size_A = M * K;
    int size_B = K * N;
    int size_C = M * N;
    float* h_A = new float[size_A];
    float* h_B = new float[size_B];
    float* h_C = new float[size_C];  // GPU计算结果
    initMatrix(h_A, M, K);  // A矩阵赋初始值，代码实现略
    initMatrix(h_B, K, N);  // B矩阵赋初始值，代码实现略
    
    // device内存分配
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A * sizeof(float));
    cudaMalloc(&d_B, size_B * sizeof(float));
    cudaMalloc(&d_C, size_C * sizeof(float));
    
    // 复制数据到device
    cudaMemcpy(d_A, h_A, size_A * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B * sizeof(float), cudaMemcpyHostToDevice);
    
    // 设置并发度，启动核函数
    // confirm one thread process one data
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    simpleMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    
    // 等待核函数完成（等待device上所有任务完成）
    cudaDeviceSynchronize();
    
    // 将device计算结果复制回host
    cudaMemcpy(h_C, d_C, size_C * sizeof(float), cudaMemcpyDeviceToHost);
   
    // 清理host&device资源
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
