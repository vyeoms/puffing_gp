# Makefile -- CPU GP and CUDA GP
#
# Targets:
#   make           -- build everything (libgp.a, libgpcu.a, gp_test)
#   make libgp.a   -- CPU-only static library
#   make libgpcu.a -- CUDA static library
#   make gp_test   -- cross-validation binary
#   make clean

# compilers

CC   := gcc
NVCC := nvcc

# CUDA paths (adjust if your CUDA is installed elsewhere)

CUDA_ROOT ?= /usr/local/cuda

# CPU flags

CFLAGS := -O3 -march=native -ffast-math -std=c11 \
          $(shell pkg-config --cflags blas lapacke 2>/dev/null)

BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null || \
             pkg-config --libs blas 2>/dev/null || \
             echo "-lopenblas") \
             $(shell pkg-config --libs lapacke 2>/dev/null || echo "-llapacke")

CPU_LDFLAGS := $(BLAS_LIBS) -lm

# CUDA flags

NVCCFLAGS := -O3 -arch=native --use_fast_math -std=c++14 \
             -Xcompiler -O3,-march=native

CUDA_INC  := -I$(CUDA_ROOT)/include
CUDA_LIBS := -L$(CUDA_ROOT)/lib64 -lcublas -lcusolver -lcudart

INC := -I.

# object files

CPU_OBJS  := gp_kernel.o gp.o
CUDA_OBJS := gp_cuda_kernel.o gp_cuda.o
TEST_OBJ  := gp_test.o

.PHONY: all clean

all: libgp.a libgpcu.a gp_test

# CPU static library

libgp.a: $(CPU_OBJS)
	ar rcs $@ $^

gp_kernel.o: gp_kernel.c gp_kernel.h
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

gp.o: gp.c gp.h gp_kernel.h
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

# CUDA static library

libgpcu.a: $(CUDA_OBJS)
	ar rcs $@ $^

gp_cuda_kernel.o: gp_cuda_kernel.cu gp_cuda_kernel.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) $(INC) -c $< -o $@

gp_cuda.o: gp_cuda.cu gp_cuda.h gp_cuda_kernel.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) $(INC) -c $< -o $@

# Test binary (links both)

gp_test: $(TEST_OBJ) libgp.a libgpcu.a
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) $(INC) -o $@ \
	    $(TEST_OBJ) libgp.a libgpcu.a \
	    $(CUDA_LIBS) $(CPU_LDFLAGS)

$(TEST_OBJ): gp_test.cu gp.h gp_kernel.h gp_cuda.h gp_cuda_kernel.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) $(INC) -c $< -o $@

clean:
	rm -f $(CPU_OBJS) $(CUDA_OBJS) $(TEST_OBJ) \
	      libgp.a libgpcu.a gp_test
