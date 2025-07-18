#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main(void) {
    // printf("%ld", sysconf(_SC_VERSION));
    // #pragma omp for
    printf("%lu\n", sizeof(void *));
    
    return 0;
}
