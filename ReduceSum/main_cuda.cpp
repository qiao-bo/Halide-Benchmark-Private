#include <iostream>
#include <limits>

#include "Halide.h"
#include "halide_benchmark.h"

#define WIDTH 65536
#define USE_AUTO

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output;
  Buffer<int> input;

  PipelineClass(Buffer<int> in) : input(in) {
    // Parallel reduction: summation
    output() = 0;
    RDom r(0, WIDTH);
    output() = output() + input(r.x);
  }

  bool test_performance() {
    target = get_host_target();
    target.set_feature(Target::CUDA);
    if (!target.has_gpu_feature()) {
      return false;
    }

#ifdef USE_AUTO
    Pipeline p(output);
    p.auto_schedule(target);
    output.compile_jit(target);
    printf("Using auto-scheduler...\n");
#else
    output.compute_root();
    output.compile_jit(target);
    printf("Computing from root...\n");
#endif

    // The equivalent C is:
    int c_ref = 0;
    for (int y = 0; y < WIDTH; y++) {
      c_ref += input(y);
    }

    input.copy_to_device(target);
    double best_time = benchmark(10, 5, [&]() {
      Buffer<int> out = output.realize();
      out.copy_to_host();
      out.device_sync();
    });
    printf("Halide time (best): %gms\n", best_time * 1e3);

    return true;
  }

private:
  Var x;
  Target target;
};

int main(int argc, char **argv) {
  const int width = WIDTH;

  // Initialize with random data
  Buffer<int> input(width);
  for (int x = 0; x < input.width(); x++) {
    input(x) = rand() & 0xfff;
  }

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }

  return 0;
}
