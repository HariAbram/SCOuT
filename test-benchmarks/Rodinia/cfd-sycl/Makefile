#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = acpp
LAUNCHER=

OPTIMIZE  = yes
DEBUG     = no
DUMP      = no

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

source = euler3d.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := --acpp-platform=cpu --acpp-targets=generic $(EXTRA_CFLAGS) -std=c++17 -Wall  

# Linker Flags
LDFLAGS = 

ifeq ($(VENDOR), codeplay)
LDFLAGS += -lComputeCpp
endif

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

ifeq ($(DUMP),yes)
  CFLAGS += -DOUTPUT
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp kernel_compute_flux.sycl  \
           kernel_compute_step_factor.sycl  \
	   kernel_initialize_variables.sycl \
	   kernel_memset.sycl \
	   kernel_time_step.sycl
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) result* density momentum density_energy

run: $(program)
	$(LAUNCHER) ./$(program) ../data/cfd/fvcorr.domn.097K 
	#./$(program) ../data/cfd/fvcorr.domn.193K   
	#./$(program) ../data/cfd/missile.domn.0.2M  
