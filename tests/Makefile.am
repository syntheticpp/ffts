
noinst_PROGRAMS = test bench_fftw bench_sfft_single bench_sfft_double

test_SOURCES = test.c
test_LDADD = $(top_builddir)/src/libffts.la

bench_fftw_SOURCES = bench_fftw.c
bench_fftw_LDADD = $(top_builddir)/src/libffts.la -lrt -lfftw3f

bench_sfft_single_SOURCES = bench_sfft_single.c
bench_sfft_single_LDADD = -lrt -lfftw3f -lsfftf

bench_sfft_double_SOURCES = bench_sfft_double.c
bench_sfft_double_LDADD = -lrt -lfftw3 -lsfft

