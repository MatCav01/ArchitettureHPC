#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define NS_PER_SECOND 1e9

double simple_sub_timespec(struct timespec t1, struct timespec t2) {

    double td1;
    double td2;

    td1 = t1.tv_sec + (t1.tv_nsec/(double)NS_PER_SECOND);
    td2 = t2.tv_sec + (t2.tv_nsec/(double)NS_PER_SECOND);

    return (td2-td1);

}


int main(void) {

    double* array;
    size_t size, s, i, j, rep;
    double result;

    struct timespec start;
    struct timespec finish;
    double delta;

//    printf("%d.%.9ld\n", (int)delta.tv_sec, delta.tv_nsec);
//    printf("%.10g\n", delta2);
 
    rep = 100;

    size = 32*1024*1024;

    array = (double*)malloc(size*sizeof(double));
    //array = (double*) aligned_alloc( 4096, size*sizeof(*array));

    for (i=0; i<size; i++) {
      array[i] = i*0.42;
    }

    printf("KByte \t\t Sec \t\t GByte/s \t\t Result \n");

size_loop:
    for (s=1024; s<size; s*=2) {

      // Start
      clock_gettime(CLOCK_REALTIME, &start);

compute_loop:
     for (j=0; j<rep; j++) {
      #pragma omp parallel for
      for (i=0; i<s; i++) {
        //result += 1.2*array[i]+0.32;
        array[i] = 1.2*array[i]+0.32;
      }
      }

      // Stop
      clock_gettime(CLOCK_REALTIME, &finish);

      delta = simple_sub_timespec(start, finish)/rep;
   
      double bytes = s*sizeof(double);
      double gbsec = bytes/(delta*NS_PER_SECOND);

      printf("%8.5g \t %5.5g \t %5.5g \t\t %5g \n", bytes/1024, delta, gbsec, array[42]);

    }

    return 0;
}

