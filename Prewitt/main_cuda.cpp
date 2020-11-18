#include <iostream>
#include <limits>

#include "Halide.h"
#include "halide_benchmark.h"

#define WIDTH 384
#define HEIGHT 256

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output;
  Func dx, dy, dxn, dyn, outs;
  float norm = 3.0f;
  Buffer<float> input;
  Buffer<int> masksx;
  Buffer<int> masksy;

  PipelineClass(Buffer<float> in, Buffer<int> msksx, Buffer<int> msksy)
      : input(in), masksx(msksx), masksy(msksy) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);

    dx(x, y) = Dx(gray)(x, y);
    dy(x, y) = Dy(gray)(x, y);

    dxn(x, y) = dx(x, y) / norm;
    dyn(x, y) = dy(x, y) / norm;

    outs(x, y) = sqrt(dxn(x, y) * dxn(x, y) + dyn(x, y) * dyn(x, y));
    outs(x, y) = Halide::select(outs(x, y) > 255.0f, 255.0f, outs(x, y));
    output(x, y) = Halide::select(outs(x, y) < 0.0f, 0.0f, outs(x, y));
  }

  bool test_performance() {
    target = get_host_target();
    target.set_feature(Target::CUDA);
    if (!target.has_gpu_feature()) {
      return false;
    }

    // Enable debug info
    // target.set_feature(Target::Profile);

    // Auto schedule the pipeline
    output.set_estimate(x, 0, WIDTH).set_estimate(y, 0, HEIGHT);
    Pipeline p(output);
    p.auto_schedule(target);
    output.compile_jit(target);

    // Test the performance of the scheduled pipeline.
    Buffer<float> out(input.width(), input.height());

    masksx.copy_to_device(target);
    masksy.copy_to_device(target);
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
  Func Dy(Func f) {
    using Halide::_;
    Func sobelY;
    Func outy;
    RDom dom(masksy); // a reduction domain of 3x3
    Expr conv = f(x + dom.x, y + dom.y) * masksy(dom.x, dom.y);
    sobelY(x, y) += conv;
    outy(x, y) = sobelY(x, y) / 6;
    return outy;
  }
  Func Dx(Func f) {
    using Halide::_;
    Func sobelX;
    Func outx;
    RDom dom(masksx); // a reduction domain of 3x3
    Expr conv = f(x + dom.x, y + dom.y) * masksx(dom.x, dom.y);
    sobelX(x, y) += conv;
    outx(x, y) = sobelX(x, y) / 6;
    return outx;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = 3;
  const int size_y = 3;

  // Prewitt mask
  const int coef_sx[size_x][size_y] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
  const int coef_sy[size_x][size_y] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};

  // Initialize with random image
  Buffer<float> input(width, height);
  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = rand() & 0xfff;
    }
  }

  Buffer<int> masksx(size_x, size_y);
  Buffer<int> masksy(size_x, size_y);
  for (int y = 0; y < masksx.height(); y++) {
    for (int x = 0; x < masksx.width(); x++) {
      masksx(x, y) = coef_sx[x][y];
    }
  }
  for (int y = 0; y < masksy.height(); y++) {
    for (int x = 0; x < masksy.width(); x++) {
      masksy(x, y) = coef_sy[x][y];
    }
  }

  printf("Running pipeline on GPU:\n");
  PipelineClass pipe(input, masksx, masksy);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
