#include <iostream>
#include <limits>

#include "Halide.h"
#include "halide_benchmark.h"

#define WIDTH 1024
#define HEIGHT 1024

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output;
  Func dx, dy, sx, sy, sxy, gx, gy, gxy, interm, lambda, lambda1, lambda2;
  float threshold = 200.0f;
  const int norm = 16;
  Buffer<int> input;
  Buffer<int> maskg;
  Buffer<int> masksx;
  Buffer<int> masksy;

  PipelineClass(Buffer<int> in, Buffer<int> mskg, Buffer<int> msksx,
                Buffer<int> msksy)
      : input(in), maskg(mskg), masksx(msksx), masksy(msksy) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);

    // compute x- and y-derivative
    dx(x, y) = Dx(gray)(x, y);
    dy(x, y) = Dy(gray)(x, y);

    // compute Hessian matrix
    sx(x, y) = dx(x, y) * dx(x, y);
    sy(x, y) = dy(x, y) * dy(x, y);
    sxy(x, y) = dx(x, y) * dy(x, y);

    gx(x, y) = Gauss(sx)(x, y);
    gy(x, y) = Gauss(sy)(x, y);
    gxy(x, y) = Gauss(sxy)(x, y);

    // compute shi-tomasi features
    interm(x, y) = sqrt((gx(x, y) - gy(x, y)) * (gx(x, y) - gy(x, y)) +
                        4.0f * gxy(x, y) * gxy(x, y));
    lambda1(x, y) = 0.5f * (gx(x, y) + gy(x, y) + interm(x, y));
    lambda2(x, y) = 0.5f * (gx(x, y) + gy(x, y) - interm(x, y));
    lambda(x, y) = min(lambda1(x, y), lambda2(x, y));
    output(x, y) = Halide::select(lambda(x, y) > threshold, 1, 0);
  }

  bool test_performance() {
    // Auto schedule the pipeline
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
    Buffer<int> out(input.width(), input.height());

    // Exclude the H2D copying time
    maskg.copy_to_device(target);
    masksx.copy_to_device(target);
    masksy.copy_to_device(target);

    double best_auto = benchmark(10, 3, [&]() {
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
  Func Gauss(Func f) {
    using Halide::_;
    Func blur;
    Func out;
    RDom dom(maskg); // a reduction domain of 3x3
    Expr conv = f(x + dom.x, y + dom.y) * maskg(dom.x, dom.y);
    blur(x, y) += conv;
    out(x, y) = blur(x, y) / norm;
    return out;
  }
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

  // Gaussian mask
  const int coef_g[size_x][size_y] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  // Sobel mask
  const int coef_sx[size_x][size_y] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
  const int coef_sy[size_x][size_y] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};

  // Initialization with random image
  Buffer<int> input(width, height);
  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = rand() & 0xfff;
    }
  }

  Buffer<int> maskg(size_x, size_y);
  Buffer<int> masksx(size_x, size_y);
  Buffer<int> masksy(size_x, size_y);
  for (int y = 0; y < maskg.height(); y++) {
    for (int x = 0; x < maskg.width(); x++) {
      maskg(x, y) = coef_g[x][y];
    }
  }
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

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input, maskg, masksx, masksy);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
