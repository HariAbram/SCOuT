#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = acpp
LAUNCHER=

OPTIMIZE  = yes
DEBUG     = no
OUTPUT    = no

GPU       = no
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp ./util/num/num.cpp ./util/timer/timer.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := --acpp-platform=cpu --acpp-targets=generic $(EXTRA_CFLAGS) -std=c++17 -Wall 

# Linker Flags
LDFLAGS = 

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -Ofast
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif

ifeq ($(OUTPUT),yes)
  CFLAGS += -DOUTPUT
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o:	main.cpp main.h kernel.sycl

./util/num/num.o: ./util/num/num.cpp ./util/num/num.h 

./util/timer/timer.o: ./util/timer/timer.cpp ./util/timer/timer.h 

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -boxes1d 30
