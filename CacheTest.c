#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
// #include <unistd.h>

#define NS_PER_SECOND 1e9

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td) {
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec = t2.tv_sec - t1.tv_sec;
    
    if (td->tv_sec > 0 && td->tv_nsec < 0) {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0) {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}

double simple_sub_timespec(struct timespec t1 , struct timespec t2) {
    double td1;
    double td2;
    td1 = t1.tv_sec + (t1.tv_nsec / (double) NS_PER_SECOND);
    td2 = t2.tv_sec + (t2.tv_nsec / (double) NS_PER_SECOND);
    return td2 - td1;
}

int main(void) {
    struct timespec start, finish;
    size_t size, s, i, j, rep;
    double delta, result;
    double* array;

    rep = 100;
    
    size = 32 * 1024 * 1024;
    array = (double*) malloc(size * sizeof(double));

    for (i = 0; i < size; i++) {
        array[i] = i * 0.42;
    }

    printf("KByte \t\t Sec \t\t GByte/s \t\t Result \n");

    for (s = 1024; s < size; s *= 2) {
        clock_gettime(CLOCK_REALTIME, &start);
        
        for (j = 0; j < rep; j++) {
            for (i = 0; i < s; i++) {
                array[i] = 1.2 * array[i] + 0.32;
            }
        }

        clock_gettime(CLOCK_REALTIME, &finish);

        delta = simple_sub_timespec(start, finish) / rep;
        double bytes = s * sizeof(double);
        double gbsec = bytes / (delta * NS_PER_SECOND);

        printf ("%8.5g \t %5.5g \t %5.5g \t\t %5g \n", bytes / 1024, delta, gbsec, array[42]);
    }

    return 0;
}