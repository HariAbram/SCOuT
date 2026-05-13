#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        ?= acpp
CXX       ?= $(CC)
OPTIMIZE  = no
DEBUG     = no
LAUNCHER  =

GPU       = no
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

SOURCE ?= multimat.cpp
REPLACE_SOURCE ?= multimat.cpp
EXEC ?= multimat

program = $(EXEC)

project_source = multimat.cpp compact.cpp full_matrix.cpp

source = $(SOURCE) $(filter-out $(REPLACE_SOURCE),$(project_source))

obj = $(patsubst %.cpp,%.o,$(filter %.cpp,$(source)))

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall 

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
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(source)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) $(source) -o $@ $(LDFLAGS)
clean:
	rm -rf $(program)

run: $(program)
	$(LAUNCHER) ./multimat_FL 2000 2000
	$(LAUNCHER) ./multimat_F 2000 2000
	$(LAUNCHER) ./multimat 2000 2000
	$(LAUNCHER) ./multimat_FL 2000 2000 0.3 0.05 0.05
	$(LAUNCHER) ./multimat_F 2000 2000 0.3 0.05 0.05
	$(LAUNCHER) ./multimat 2000 2000 0.3 0.05 0.05
