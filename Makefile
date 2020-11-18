include ../support/Makefile.inc

CXXFLAGS += -g -Wall

.PHONY: clean

$(BIN)/main_cuda: main_cuda.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) main_cuda.cpp $(LIB_HALIDE) -o $@ $(IMAGE_IO_FLAGS) $(LDFLAGS) $(LIBHALIDE_LDFLAGS) $(HALIDE_SYSTEM_LIBS)

clean:
	rm -rf $(BIN)

test: $(BIN)/main_cuda
	@mkdir -p $(@D)
	$(BIN)/main_cuda
