
noinst_PROGRAMS = test bench_fftw bench_sfft_single

test_SOURCES = test.c
test_LDADD = $(top_builddir)/src/libffts.la

bench_fftw_SOURCES = bench_fftw.c
bench_fftw_LDADD = $(top_builddir)/src/libffts.la -lrt -lfftw3f

bench_sfft_single_SOURCES = bench_sfft_single.c
bench_sfft_single_LDADD = -lrt -lfftw3f -lsfft
