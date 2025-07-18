#include <stdio.h> // I/O library
#include <omp.h> // #pragma omp library

int main(int argc, char *argv[])
{
    int nthreads, tid;
    
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("Hello World from thread %d\n", tid);

        if (tid == 0)
        {
            nthreads = omp_get_num_threads();
            printf("Number of threads = %d\n", nthreads);
        }
    }

    return 0;
}