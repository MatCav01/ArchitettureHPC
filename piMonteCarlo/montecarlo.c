#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

int main(int argc, char *argv[]){

  double x_value, y_value;
  unsigned long long int hit, myhit, i;
  unsigned long long int n_darts;

  n_darts=(unsigned long long int)1024;
  
  srand(42);

  hit = 0;

  for (i=0;i<n_darts;i++) {

    x_value = (double)(rand())/RAND_MAX;
    y_value = (double)(rand())/RAND_MAX;

    if (((x_value*x_value)+(y_value*y_value))<=1) hit++;

  }

  printf("There were %d hits in the circle \n", hit);
  printf("The estimated value of pi is: " );
  printf("%.10f \n", (double)(hit*4)/(double)(n_darts));
  printf("The actual value is:          ");
  printf("%.10f... \n", M_PI );

  return 0;

}

