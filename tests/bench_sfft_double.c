/*

 This file is part of SFFT.

 Copyright (c) 2014, Peter Kümmel
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
    * Neither the name of the organization nor the
      names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL ANTHONY M. BLAKE BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <fftw3.h>
#include <sfft/sfft.h>

#include "../config.h"

#ifdef HAVE_SSE
#include <xmmintrin.h>
#endif


typedef struct Data
{
    int N;
    double* in;
    double* out;
    void* plan;
} Data;


long nsecClock()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e+9* ts.tv_sec + ts.tv_nsec;
}


void initData(Data* d)
{
    for(int i = 0; i < d->N; i++) {
        d->in[2*i]   = 0.0f;
        d->in[2*i+1] = 0.0f;
    }
    d->in[2] = 1.0f;
}


long fftw_transform(Data* d)
{
    if (d->in == 0) {
        fftw_complex* in =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * d->N);
        fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * d->N);
        d->in = (double*)in;
        d->out = (double*)out;
        d->plan = (void*)fftw_plan_dft_1d(d->N, in, out, FFTW_FORWARD, FFTW_PATIENT);
        initData(d);
    }

    long t0 = nsecClock();
    fftw_execute((fftw_plan)d->plan);
    return nsecClock() - t0;
}


void fftw_clean(Data* d)
{
    fftw_free((fftw_complex*)d->in);
    fftw_free((fftw_complex*)d->out);
    fftw_destroy_plan((fftw_plan)d->plan);
}


long sfft_transform (Data* d)
{
    if (d->in == 0) {
#ifdef HAVE_SSE
        double __attribute__ ((aligned(32))) *in = _mm_malloc(2*d->N * sizeof(double), 32);
        double __attribute__ ((aligned(32))) *out = _mm_malloc(2*d->N * sizeof(double), 32);
#endif
        d->in = in;
        d->out = out;
        d->plan = (void*)sfft_init(d->N, SFFT_FORWARD | SFFT_DOUBLE);
        if(!d->plan)
            printf("Invalid sfft plan\n");
        initData(d);
    }

    long t0 = nsecClock();
    if (d->plan)
        sfft_execute((sfft_plan_t*)d->plan, d->in, d->out);
    return nsecClock() - t0;
}


void sfft_clean(Data* d)
{
#ifdef HAVE_SSE
    _mm_free(d->in);
    _mm_free(d->out);
#else
    free(d->in);
    free(d->out);
#endif
    if (d->plan)
        sfft_free((sfft_plan_t*)d->plan);
}

double gigaComplexCooleyTukeyFLOPS(int N, long nsec)
{
    if (nsec == 0) {
        return 0;
    }
    return 5.0*N*log2(N) / nsec;
}


double bench(int N, const char* msg, int runs, long(*transform)(Data*), void(*cleanup)(Data*))
{
    Data d;
    d.N = N;
    d.in = 0;
    d.out = 0;

    double tmin = 1e20;
    for (int i = 0; i < runs; i++) {
        double dt = transform(&d);
        if (dt != 0)
            tmin = dt < tmin ? dt : tmin;
    }
    cleanup(&d);

    double gigaflops = gigaComplexCooleyTukeyFLOPS(N, tmin);
    //printf("N=2^%.0f=%i: %s %.1f Gigaflops\n", log2(N), N, msg, gigaflops);
    return gigaflops;
}


int main(int argc, char *argv[])
{
    int Nmax = 1 << 18;
    int minRuns = 10;
    
    if(argc == 2) {
        minRuns = atoi(argv[1]);
    }

    
    printf("\nRunning benchmarks until N = 2^%.0f = %i = %ikB\n\n", log2(Nmax), Nmax, 2*Nmax/1024*(int)sizeof(double));
    for (int N = 4; N <= Nmax; N = N<<1) {
        int runs = 1e6 / gigaComplexCooleyTukeyFLOPS(N, 1);
        runs = runs < minRuns ? minRuns : runs;
        double gw = bench(N, "FFTW ", runs, &fftw_transform, &fftw_clean);
        double gs = bench(N, "SFFT ", runs, &sfft_transform, &sfft_clean);
        printf("N = 2^%2.0f = %7i: Speed %4.1f GFLOPS  SFFT/FFTW = %4.2f\n", log2(N), N, gs, gs/gw);
    }

    return 0;
}



// vim: set autoindent noexpandtab tabstop=3 shiftwidth=3:
