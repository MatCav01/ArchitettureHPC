#define _POSIX_C_SOURCE 200809L
#include <stdio.h> // I/O library
#include <stdlib.h> // posix_memalign library
#include <math.h> // log2f library
#include <omp.h> // #pragma omp library
#define IMGX 512
#define IMGY 512
#define GAIN 1.0
#define MAXNUMTHREAD 16
#define REPEAT 1000

int main()
{
    FILE *fpin, *fpout;
    float *imgin, *imgout;
    double t0, dt;

    // imgin, imgout allocation
    posix_memalign((void **) &imgin, 4096, IMGX * IMGY * sizeof(float));
    posix_memalign((void **) &imgout, 4096, IMGX * IMGY * sizeof(float));

    // moon.in read
    fpin = fopen("moon.in", "r");
    for (int i = 0; i < IMGX; i++)
    {
        size_t nr = fread(imgin + i * IMGY, sizeof(float), IMGY, fpin);
        if (nr != IMGY)
        {
            perror("fread() failed");
            exit(-1);
        }
    }
    fclose(fpin);

    // img process
    for (int t = 1; t <= MAXNUMTHREAD; t++)
    {
        t0 = omp_get_wtime();

        #pragma omp parallel for num_threads(t)
        for (int i = 0; i < IMGX; i++)
        {
            for (int j = 0; j < IMGY; j++)
            {
                // imgout[i * IMGY + j] = GAIN * log2f(1.0 + imgin[i * IMGY + j]);
                for (int k = 0; k < REPEAT; k++)
                {
                    imgout[i * IMGY + j] += GAIN * log2f(1.0 + imgin[i * IMGY + j]);
                }
            }
        }
        dt = omp_get_wtime() - t0;

        printf("t: %d  dt: %f msec  (%f ns/point)\n", t, dt * 1e3, dt * 1e9 / (double) (IMGX * IMGY));

        // moon.out write
        fpout = fopen("moon.out", "w");
        for (int i = 0; i < IMGX; i++)
        {
            size_t nr = fwrite(imgout + i * IMGY, sizeof(float), IMGY, fpout);
            if (nr != IMGY)
            {
                perror("fwrite() failed");
                exit(-1);
            }
        }
        fclose(fpout);
    }

    return 0;
}