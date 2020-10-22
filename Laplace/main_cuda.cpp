#include "Halide.h"
#include "halide_benchmark.h"
#include <iostream>
#include <limits>

#define WIDTH 1024
#define HEIGHT 1024

#define DTYPE unsigned char

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func intermBuf;
  Func output;
  Buffer<DTYPE> input;
  Buffer<float> maskDoG;

  PipelineClass(Buffer<DTYPE> in, Buffer<float> mask) : input(in), maskDoG(mask) {
    Func gray = BoundaryConditions::repeat_edge(input);
    intermBuf(x, y) = Laplace(gray)(x, y);
    intermBuf(x, y) = intermBuf(x, y) + 128.0f;
    intermBuf(x, y) =
        Halide::select(intermBuf(x, y) > 255.0f, 255.0f, intermBuf(x, y));
    intermBuf(x, y) =
        Halide::select(intermBuf(x, y) < 0.0f, 0.0f, intermBuf(x, y));
    output(x, y) = cast<DTYPE>(intermBuf(x, y));
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

    // Auto schedule the pipeline
    Pipeline p(output);
    p.auto_schedule(target);
    output.compile_jit(target);

    // Test the performance of the scheduled pipeline.
    Buffer<DTYPE> out(input.width(), input.height());

    maskDoG.copy_to_device(target);
    input.copy_to_device(target);
    double best_auto = benchmark(10, 3, [&]() {
      p.realize(out);
      // out.copy_to_host();
      out.device_sync();
    });
    printf("Auto-tuned time: %gms\n", best_auto * 1e3);

    return true;
  }

private:
  Var x, y;
  Target target;

  Func Laplace(Func f) {
    using Halide::_;
    Func blur;
    RDom dom(maskDoG); // a reduction domain
    Expr conv = f(x + dom.x, y + dom.y) * maskDoG(dom.x, dom.y);
    blur(x, y) += conv;
    return blur;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = 5;
  const int size_y = 5;

  // Laplace mask
  const float coef[size_x][size_y] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -24,
                                      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Initialize with random image
  Buffer<DTYPE> input(width, height);
  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = (DTYPE)((rand()) % 256);
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
