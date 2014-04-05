
noinst_PROGRAMS = test bench_fftw

test_SOURCES = test.c
test_LDADD = $(top_builddir)/src/libffts.la

bench_fftw_SOURCES = bench_fftw.c
bench_fftw_LDADD = $(top_builddir)/src/libffts.la -lrt -lfftw3f
