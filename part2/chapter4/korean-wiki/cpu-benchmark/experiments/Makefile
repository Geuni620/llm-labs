CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -ffp-contract=off -fno-math-errno
OUTPUT ?= benchmark
LIBOMP_PREFIX ?= /opt/homebrew/opt/libomp
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
CPPFLAGS += -DACCELERATE_NEW_LAPACK -isystem $(LIBOMP_PREFIX)/include
OMP_FLAGS := -Xpreprocessor -fopenmp
BACKEND_LIBS := -framework Accelerate -L$(LIBOMP_PREFIX)/lib -lomp -Wl,-rpath,$(LIBOMP_PREFIX)/lib
else
OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS ?= $(shell pkg-config --libs openblas 2>/dev/null)
CPPFLAGS += $(OPENBLAS_CFLAGS)
OMP_FLAGS := -fopenmp
BACKEND_LIBS := $(if $(strip $(OPENBLAS_LIBS)),$(OPENBLAS_LIBS),-lopenblas)
endif

CPPFLAGS += -DBENCH_BUILD_FLAGS='"$(CFLAGS) $(OMP_FLAGS)"'
SOURCES := benchmark.c gemm_kernels.c vector_kernels.c
HEADERS := experiment_kernels.h fast_exp.h

.PHONY: all clean
all: $(OUTPUT)

$(OUTPUT): $(SOURCES) $(HEADERS) Makefile
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMP_FLAGS) $(SOURCES) -o $@ $(LDFLAGS) $(BACKEND_LIBS) -lm

clean:
	rm -f $(OUTPUT)
