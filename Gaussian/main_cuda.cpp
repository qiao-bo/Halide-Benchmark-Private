#include "Halide.h"
#include "halide_benchmark.h"
#include <iostream>
#include <limits>

#define WIDTH 256
#define HEIGHT 256

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output;
  Func blur_x;
  Buffer<float> input;
  Buffer<float> maskGaus;

  PipelineClass(Buffer<float> in, Buffer<float> mask)
      : input(in), maskGaus(mask) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);
    // Gaussian
    output(x, y) = GaussBlur(gray)(x, y);
  }

  bool test_performance() {
    target = get_host_target();
    target.set_feature(Target::CUDA);
    if (!target.has_gpu_feature()) {
      return false;
    }

    // target.set_feature(Target::Profile); // Enable debug info

    // Auto schedule the pipeline
    output.estimate(x, 0, WIDTH).estimate(y, 0, HEIGHT);
    Pipeline p(output);
    p.auto_schedule(target);
    output.compile_jit(target);

    // Test the performance of the scheduled pipeline.
    Buffer<float> out(input.width(), input.height());

    double best_auto = benchmark(10, 3, [&]() {
      maskGaus.copy_to_device(target); // include H2D copying time
      input.copy_to_device(target);
      p.realize(out);
      out.copy_to_host(); // include D2H copying time
      out.device_sync();
    });
    printf("Auto-tuned time: %gms\n", best_auto * 1e3);

    return true;
  }

private:
  Var x, y;
  Target target;

  // 3x3 Gaussian filter
  Func GaussBlur(Func f) {
    using Halide::_;
    Func blur;
    RDom dom(maskGaus);
    Expr conv = f(x + dom.x, y + dom.y) * maskGaus(dom.x, dom.y);
    blur(x, y) += conv;
    return blur;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = 3;
  const int size_y = 3;

  // Gaussian mask
  const float coef[3][3] = {0.057118f, 0.124758f, 0.057118f, 0.124758f, 0.272496f,
                            0.124758f, 0.057118f, 0.124758f, 0.057118f};

  // Initialize with random image
  Buffer<float> input(width, height);
  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = rand() & 0xfff;
    }
  }

  Buffer<float> mask(size_x, size_y);
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
