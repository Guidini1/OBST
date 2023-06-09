#include<cstdlib>
#include<cstdio>
#include<limits>
using namespace std;
void alloc_matrix(void*** a,int m,int n,int size) {
    int i;
    void* storage;
    storage=(void*)malloc(m*n*size);
    *a=(void**)malloc(m*sizeof(void*));
    for(i=0;i<m;i++) {
        (*a)[i]=(char*)storage+i*n*size;
        //(*a)[i]=storage+i*n*size;
    }
}
void print_root(int& p,int* output,int** root,int low,int high) {
    output[p++]=root[low][high+1];
  printf("Root of the tree spanning %d-%d is %d\n",low,high,root[low][high+1]);
    if(low<=root[low][high+1]-1)
        print_root(p,output,root,low,root[low][high+1]-1);
    if(root[low][high+1]<=high-1)
        print_root(p,output,root,root[low][high+1]+1,high);
}
void obst(int* output,int n,float* p) {
    float bestcost;
    int bestroot;
    float** cost;
    int low;
    int j;
    int high;
    int r;
    float rcost;
    float dcost;
    int** root;
    alloc_matrix((void***)&root,n+1,n+1,sizeof(int));
    alloc_matrix((void***)&cost,n+1,n+1,sizeof(float));
    int diag, cell;
for(diag = 0; diag <= n; diag++){
    for(cell = 0; cell <= n-diag; cell++){
        low = cell;
        high = cell+diag;
        if(low == high){
            cost[low][low]=0.0;
            root[low][low]=low;
        }
        else{
            bestcost=numeric_limits<float>::max();
            for(r=low;r<high;r++) {
                rcost=cost[low][r]+cost[r+1][high];
                if(rcost<bestcost) {
                    bestcost=rcost;
                    bestroot=r;
                }
            }
            dcost = 0.0;
            for(j=low;j<high;j++) dcost+=p[j];
            cost[low][high]=bestcost+dcost;
            root[low][high]=bestroot;
        }
        low++;
    }      
}
int tmp=0;
print_root(tmp,output,root,0,n-1);
}
