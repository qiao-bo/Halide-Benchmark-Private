#include <iostream>
#include <limits>

#include "Halide.h"
#include "halide_benchmark.h"

#define WIDTH 512
#define HEIGHT 512

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output;
  Func gaus, sharp, ratio;
  const int norm = 16;
  Buffer<float> input;
  Buffer<int> mask;

  PipelineClass(Buffer<float> in, Buffer<int> mask) : input(in), mask(mask) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);

    gaus(x, y) = Gauss(gray)(x, y);
    sharp(x, y) = 2 * gray(x, y) - gaus(x, y);
    ratio(x, y) = sharp(x, y) / gray(x, y);
    output(x, y) = ratio(x, y) * gray(x, y);
  }

  bool test_performance() {
    target = get_host_target();
    target.set_feature(Target::CUDA);
    if (!target.has_gpu_feature()) {
      return false;
    }

    // Enable debug info
    // target.set_feature(Target::Profile);

    output.estimate(x, 0, WIDTH).estimate(y, 0, HEIGHT);
    Pipeline p(output);
    p.auto_schedule(target);
    output.compile_jit(target);

    // Test the performance of the scheduled pipeline.
    Buffer<float> out(input.width(), input.height());

    mask.copy_to_device(target);
    double best_auto = benchmark(10, 3, [&]() {
      input.copy_to_device(target);
      p.realize(out);
      out.copy_to_host();
      out.device_sync();
    });
    printf("Auto-tuned time: %gms\n", best_auto * 1e3);

    return true;
  }

private:
  Var x, y;
  Target target;
  Func Gauss(Func f) {
    using Halide::_;
    Func blur;
    Func out;
    RDom dom(mask); // a reduction domain of 3x3
    Expr conv = f(x + dom.x, y + dom.y) * mask(dom.x, dom.y);
    blur(x, y) += conv;
    out(x, y) = blur(x, y) / norm;
    return out;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = 3;
  const int size_y = 3;

  // Gaussian mask
  const int coef[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};

  // Initialize with random image
  Buffer<float> input(width, height);
  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = rand() & 0xfff;
    }
  }

  Buffer<int> mask(size_x, size_y);
  for (int y = 0; y < mask.height(); y++) {
    for (int x = 0; x < mask.width(); x++) {
      mask(x, y) = coef[x][y];
    }
  }

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input, mask);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
