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

SOURCE ?= main.cpp
REPLACE_SOURCE ?= main.cpp
EXEC ?= vj-gpu

program = $(EXEC)

project_source = main.cpp haar.cpp image.cpp stdio-wrapper.cpp \
         rectangles.cpp

main_source = $(if $(filter main.cpp,$(REPLACE_SOURCE)),$(SOURCE),main.cpp)
haar_source = $(if $(filter haar.cpp,$(REPLACE_SOURCE)),$(SOURCE),haar.cpp)
image_source = $(if $(filter image.cpp,$(REPLACE_SOURCE)),$(SOURCE),image.cpp)
stdio_source = $(if $(filter stdio-wrapper.cpp,$(REPLACE_SOURCE)),$(SOURCE),stdio-wrapper.cpp)
rectangles_source = $(if $(filter rectangles.cpp,$(REPLACE_SOURCE)),$(SOURCE),rectangles.cpp)

source = $(main_source) $(haar_source) $(image_source) $(stdio_source) $(rectangles_source)

main_obj = build/$(notdir $(program))-main.o
haar_obj = build/$(notdir $(program))-haar-gpu.o
image_obj = build/$(notdir $(program))-image.o
stdio_obj = build/$(notdir $(program))-stdio-wrapper.o
rectangles_obj = build/$(notdir $(program))-rectangles.o
obj = $(main_obj) $(haar_obj) $(image_obj) $(stdio_obj) $(rectangles_obj)

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

all: build

build: $(program)

HEADERS := image.h haar.h stdio-wrapper.h

$(main_obj): $(main_source) $(HEADERS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -x c++ -o $@ -c $<

$(image_obj): $(image_source) $(HEADERS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -x c++ -o $@ -c $<

$(stdio_obj): $(stdio_source) $(HEADERS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -x c++ -o $@ -c $<

$(haar_obj): $(haar_source) $(HEADERS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -DSYCL -x c++ -o $@ -c $<

$(rectangles_obj): $(rectangles_source) $(HEADERS)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -x c++ -o $@ -c $<

$(program): $(obj)
	mkdir -p $(dir $@)
	$(CXX) $(CFLAGS) -o $@ $+ $(LDFLAGS)
clean:
	rm -rf $(program) $(obj) *.o Output-cpu.pgm Output-gpu.pgm

run: build
	$(LAUNCHER) ./vj-cpu ../face-cuda/Face.pgm ../face-cuda/info.txt ../face-cuda/class.txt Output-cpu.pgm
	$(LAUNCHER) ./vj-gpu ../face-cuda/Face.pgm ../face-cuda/info.txt ../face-cuda/class.txt Output-gpu.pgm
