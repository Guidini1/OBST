# OBST
# Código da maratona de programação sobre o problema de árvores binárias de busca ótimas
#Descrição do problema:
  Uma árvore de busca binária é uma maneira de organizar n chaves de um conjunto ordenado linearmente para garantir sua recuperação em tempo *Θ(log n)*. Dada a probabilidade de cada chave ser acessada, nossa tarefa consiste em criar uma árvore binária de busca ótima que minimize o tempo médio de busca.
#Estratégias de Paralelização:
  A paralelização do código foi feita utilizando duas estratégias diferentes, primeiro foi implementada uma paralelização utilizando OpenMP que aloca threads para dividir as execuções, foi implementada também uma paralelização utilizando a GPU com CUDA.
#Versão sequencial de referência:
  A parte do código original onde realizamos a paralelização é na função void obst(int* output,int n,float* p) no código obst-seq.cc, mas especificamente no trecho:
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
#Versão Paralela Comentada:
  
