#include<iostream>
#include<malloc.h>
#include<sys/time.h>
#include<time.h>
#include<cstdlib>
#include<cstdio>

__global__
void obst(float *cost, int *root, int diag, int n, float *p){
    int low = 0, high = diag;
    int desl = 32*blockIdx.x + threadIdx.x;
    float rcost, bestcost, dcost;
    int r, j, bestroot;
    low += desl;
    high += desl;

    
    if((high > n) || (low > high)){
        return;
    }
    else if(low == high){
        cost[low*(n+1)+low]=0.0;
        root[low*(n+1)+low]=low;

    }
    else{
        bestcost=9999999;
        dcost = 0.0;
        for(j=low;j<high;j++) dcost+=p[j];
        for(r=low;r<high;r++) {
            rcost=cost[low*(n+1)+r]+cost[(r+1)*(n+1)+high];

            rcost+=dcost;
            if(rcost<bestcost) {
                bestcost=rcost;
                bestroot=r;
            }
        }
        cost[low*(n+1)+high]=bestcost;
        root[low*(n+1)+high]=bestroot;
        
    }

}


void print_matrix(int n, int *matriz){
    for(int i =0; i <= n;i++){
        for(int j = 0; j <= n;j++){
            if (i == j)
                printf("[%d] ", matriz[i*(n+1)+j]);
            else printf("%d ", matriz[i*(n+1)+j]);
        }
        printf("\n");
    }
    printf("\n");
}


using namespace std;
int main(int argc,char* argv[]) try {
    struct timeval timevalA;
    struct timeval timevalB;

    gettimeofday(&timevalA,NULL);
    int i;
    int n;
    int* root, *root_d;
    float* p, *p_d, *cost_d;
    void print_root(int**,int,int);
    fscanf(stdin,"%d",&n);
    p=(float*)malloc(n*sizeof(float));
    for(i=0;i<n;i++)
        fscanf(stdin,"%f",&p[i]);
    int* output=new int[n];
    printf("n = %d\n", n);
    
    cudaMalloc(&p_d, n*sizeof(float));
    cudaMalloc(&cost_d, (n+1)*(n+1)*sizeof(float));
    cudaMalloc(&root_d, (n+1)*(n+1)*sizeof(float));
    root = (int*)malloc((n+1)*(n+1)*sizeof(int));

    cudaMemcpy(p_d, p, n*sizeof(float), cudaMemcpyHostToDevice);

    int tam_diag, diag;
    unsigned int num_blocos;
    dim3 blocos, threads;

     cudaError_t err;

    for(diag = 0; diag <= n; diag++){
        tam_diag = (n+1) - diag;
        num_blocos = tam_diag/32;
        if (tam_diag % 32 > 0)
            num_blocos++;
        blocos = {num_blocos, 1, 1};
        threads = {32, 1, 1};

        obst<<<blocos, threads >>>(cost_d,root_d,diag,n,p_d);

        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err!= cudaSuccess) {
            printf("Erro: %s\n", cudaGetErrorString(err));
        }

    }
    

    cudaMemcpy(root, root_d, (n+1)*(n+1)*sizeof(float), cudaMemcpyDeviceToHost);

    gettimeofday(&timevalB,NULL);

    cout<<timevalB.tv_sec-timevalA.tv_sec+(timevalB.tv_usec-timevalA.tv_usec)/(double)1000000<<endl;
    cudaFree(root_d);
    cudaFree(cost_d);
    cudaFree(p_d);
    return EXIT_SUCCESS;
}
catch(...) {
    cerr<<"EXIT_FAILURE";
    return EXIT_FAILURE;
}

