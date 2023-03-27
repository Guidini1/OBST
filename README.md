# OBST
# Código da maratona de programação sobre o problema de árvores binárias de busca ótimas
Tive um pouco de dificuldade com o GitHub, mas acredito que deixei o código sequencial na pasta seq, a versão com openMP na pasta thr e a versão com CUDA na pasta Cuda. O gráfico dos resultados também está no diretório. Aqui está o link do Colab onde eu trabalhei que pode ver o que foi feito mais organizado https://colab.research.google.com/drive/1h0kjXxHeqbzl9hpVfRu0CdxtZS2U9EwR?usp=sharing
# Descrição do problema:
  Uma árvore de busca binária é uma maneira de organizar n chaves de um conjunto ordenado linearmente para garantir sua recuperação em tempo *Θ(log n)*. Dada a probabilidade de cada chave ser acessada, nossa tarefa consiste em criar uma árvore binária de busca ótima que minimize o tempo médio de busca.
# Estratégias de Paralelização:
  A paralelização do código foi feita utilizando duas estratégias diferentes, primeiro foi implementada uma paralelização utilizando OpenMP que aloca threads para dividir as execuções, foi implementada também uma paralelização utilizando a GPU com CUDA.
# Versão sequencial de referência:
  A parte do código original onde realizamos a paralelização é na função void obst(int* output,int n,float* p) no código obst-seq.cc, mas especificamente no trecho:  
```for(diag = 0; diag <= n; diag++){
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
```
# Versão Paralela com OpenMP Comentada:

```
# Esse for não pode ser paralelizado por que controla as diagonais e tem dependência de dados
for(diag = 0; diag <= n; diag++){
        // Esse for é onde temos a maior quantidade de operações e por isso eu escolhi paralelizá-lo, ele é o responsável por realizar os cálculos dentro das matrizes 
        // Por isso foi declarada as matrizes cost e root como shared para todas as threads poderem acessar, as variáveis declaradas como private são necessárias para 
        // cada thread realizarem os cálculos, e a cláusula schedule(dynamic) foi utilizada pois como temos tamanhos diferentes nas diagonais algumas threads estavam 
        // terminando antes e ficando ociosas, então usando essa cláusula alocamos ela dinamicamente para outra operação sem ficar ociosa
        #pragma omp parallel for shared(cost,root,n,p) private(cell,low,high,bestcost,dcost,r,j,rcost,bestroot) schedule(dynamic)
        for(cell = 0; cell <= n-diag; cell++){
          low = cell;
          high = cell+diag;
            if(low == high){
                cost[low][low]=0.0;
                root[low][low]=low;
            }else{
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
        }
    }
    int tmp=0;
    print_root(tmp,output,root,0,n-1);
}
```
# Versão Paralela com Cuda Comentada:
```
// Função kernel obst, responsável por calcular o custo e a raiz ótima para uma subárvore de uma árvore binária de busca otimizada. 
// Recebe como entrada um vetor p contendo as probabilidades dos nós da árvore e os ponteiros para a matriz de custo cost e matriz de raiz root. 
// As variáveis low e high delimitam o intervalo da subárvore, enquanto desl é utilizado para deslocar a subárvore de acordo com a posição dos blocos na 
// xecução paralela. A variável bestcost é responsável por armazenar o melhor custo obtido até o momento e bestroot armazena a melhor raiz encontrada.
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
```
# Análise da escalabilidade 
  A análise da escalabilidade do código deve levar em conta como o desempenho do programa muda com o aumento do tamanho do problema. Podemos ver que nesse caso, à medida que o tamanho do problema aumenta, o tempo de execução do programa aumente proporcionalmente. Com o OpenMP, a partir de um aumento para 100 nós já começamos a ver eficiência em aumento de tempo dividindo as tarefas. E com o CUDA vamos obter esse ganho lá com 400 nós onde ele supera por muito e deixa de crescer tanto.
  
# Discussão sobre a eficiência da solução e Conclusão
  A partir das duas implementações vemos que com o OpenMP divindo as tarefas em threads conseguimos ver que ela passa o código sequencial em efiência a partir mais ou menos dos problemas com 200 nós, e consegue se manter melhor para os valores maiores. Para os casos menores o OpenMP não é tão eficiente por conta dos overheads e sincronização das threads. A implementação com CUDA tem um tempo quase que constante e só começa a crescer mais perto dos 2000 nós até lá tem um valor médio que consegue ser melhor que o sequencial e o paralelo a partir de aproximadamente 400 threads.
  Podemos concluir então que para problemas muito grandes o CUDA será a melhor opção em questão de tempo superando bastante as outras duas implementações, e para baixos valores a implementação com OpenMP e sequencial tem valores parecidos mas com o OpenMP sendo um pouco melhor.
