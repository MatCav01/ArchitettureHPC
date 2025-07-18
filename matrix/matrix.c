#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <omp.h> // #pragma omp library
#define NRA 8
#define NCA 8
#define NCB 8

// void print_matrix(int nr, int nc, int m[nr][nc]);

int main(int argc, char *argv[])
{
    int tid, chunk, i, j, k, *a, *b, *c;
    // int a[NRA][NCA], b[NCA][NCB], c[NRA][NCB];
    posix_memalign((void **) &a, 4096, NRA * NCA * sizeof(int));
    posix_memalign((void **) &b, 4096, NCA * NCB * sizeof(int));
    posix_memalign((void **) &c, 4096, NRA * NCB * sizeof(int));
    
    for(i = 0; i < NRA; i++)
        for(j = 0; j < NCA; j++)
            // a[i][j] = i + j;
            a[i * NCA + j] = i + j;

    for(i = 0; i < NCA; i++)
        for(j = 0; j < NCB; j++)
            // b[i][j] = (i == j ? 1 : 0);
            b[i * NCB + j] = (i == j ? 1 : 0);

    chunk = 2; // OpenMP schedula 1 chunk come una riga della matrice a

    #pragma omp parallel private(tid, j, k)
    {
        tid = omp_get_thread_num();

        printf("Thread %d starting matrix multiply... \n", tid);

        #pragma omp for schedule(static, chunk)
        for (i = 0; i < NRA; i++)
        {
            printf("Thread %d did row %d\n", tid, i);

            for(j = 0; j < NCB; j++)
                for(k = 0; k < NCA; k++)
                    // c[i][j] += a[i][k] * b[k][j];
                    c[i * NCB + j] += a[i * NCA + k] * b[k * NCB + j];
        }
    }

    /* printf("\na =\n");
    print_matrix(NRA, NCA, a);

    printf("\nb =\n");
    print_matrix(NCA, NCB, b);

    printf("\nc =\n");
    print_matrix(NRA, NCB, c); */

    return 0;
}

/* void print_matrix(int nr, int nc, int m[nr][nc])
{
    for(int i = 0; i < nr; i++)
    {
        for(int j = 0; j < nc; j++)
            printf("%d\t", m[i][j]);
        
        printf("\n");
    }
} */