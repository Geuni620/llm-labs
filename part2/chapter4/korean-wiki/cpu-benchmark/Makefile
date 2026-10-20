CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -fno-math-errno
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
CPPFLAGS += -DACCELERATE_NEW_LAPACK
BLAS_LIBS := -framework Accelerate
else
OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS ?= $(shell pkg-config --libs openblas 2>/dev/null)
BLAS_LIBS := $(if $(strip $(OPENBLAS_LIBS)),$(OPENBLAS_LIBS),-lopenblas)
CPPFLAGS += $(OPENBLAS_CFLAGS)
endif

CPPFLAGS += -DBENCH_BUILD_FLAGS='"$(CFLAGS)"'

.PHONY: all clean
all: cpu_benchmark

cpu_benchmark: cpu_benchmark.c Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDFLAGS) $(BLAS_LIBS) -lm

clean:
	rm -f cpu_benchmark
