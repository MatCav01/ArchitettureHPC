#ifndef GLX
#warning setting GLX=512
#define GLX 512
#endif

#ifndef GLY
#warning setting GLY=512
#define GLY 512
#endif

#ifndef MAXITER
#warning setting MAXITER=10000
#define MAXITER 10000
#endif

#if DUMP == 1
#ifndef DUMPSTEP
#warning setting DUMPSTEP=500
#define DUMPSTEP 500
#endif
#endif

#define TB 72.0
#define TH 212.0
#define TC 32.0

#define HX 1
#define HY 1

#define GX (HX + GLX + HX)
#define GY (HY + GLY + HY)

#define K 64

void init(double *grid)
{
    int GLXH2 = (int)(GLX / 2);
    int GLYH2 = (int)(GLY / 2);

    for (int i = 0; i < GY; i++)
    {
        for (int j = 0; j < GX; j++)
        {
            grid[i * GX + j] = TC;
        }
    }

    // init vertical gradient
    // for (int i = HY; i < HY + GLYH2; i++)
    // {
    //     for (int j = GX - HX - 1 - K; j < GX - HX; j++)
    //     {
    //         grid[i * GX + j] = TC + (double)(TH * 2 * (i - HY)) / (double)(GLY);
    //     }
    // }
    // for (int i = HY + GLYH2; i < HY + GLY; i++)
    // {
    //     for (int j = GX - HX - 1 - K; j < GX - HX; j++)
    //     {
    //         grid[i * GX + j] = TC + (double)(TH * 2 * (GY - (i - HY))) / (double)(GLY);
    //     }
    // }

    // init circle gradient
    for (int i = HY + GLYH2 - 1 - K; i < HY + GLYH2; i++)
    {
        for (int j = HX + GLXH2 - 1 - K; j < HX + GLXH2; j++)
        {
            grid[i * GX + j] = TC + (double)(TH * (i - (HY + GLYH2 - 1 - K)) * (j - (HX + GLXH2 - 1 - K))) / (double)(K * K);
        }
        for (int j = HX + GLXH2; j < HX + GLXH2 + K; j++)
        {
            grid[i * GX + j] = TC + (double)(TH * (i - (HY + GLYH2 - 1 - K)) * ((HX + GLXH2 + K) - j)) / (double)(K * K);
        }
    }
    for (int i = HY + GLYH2; i < HY + GLYH2 + K; i++)
    {
        for (int j = HX + GLXH2 - 1 - K; j < HX + GLXH2; j++)
        {
            grid[i * GX + j] = TC + (double)(TH * ((HY + GLYH2 + K) - i) * (j - (HX + GLXH2 - 1 - K))) / (double)(K * K);
        }
        for (int j = HX + GLXH2; j < HX + GLXH2 + K; j++)
        {
            grid[i * GX + j] = TC + (double)(TH * ((HY + GLYH2 + K) - i) * ((HX + GLXH2 + K) - j)) / (double)(K * K);
        }
    }
}

void dump(double *grid, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening dump file\n");
        exit(EXIT_FAILURE);
    }

    for (int i = HY; i < GY - HY; i++)
    {
        fwrite(grid + i * GX + HX, sizeof(double), GLX, file);
    }
    
    fclose(file);
}

double checksum(double *grid)
{
    double chk = 0.0;
    for (int i = HY; i < GY - HY; i++)
    {
        for (int j = HX; j < GX - HX; j++)
        {
            chk += grid[i * GX + j];
        }
    }
    return chk;
}