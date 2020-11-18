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
  Buffer<float> input;
  Buffer<float> mask;
  float sigma_s;

  PipelineClass(Buffer<float> in, Buffer<float> mask, float sigma_s)
      : input(in), mask(mask), sigma_s(sigma_s) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);
    // Bilateral
    output(x, y) = Bilateral(gray)(x, y);
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

    double best_auto = benchmark(10, 1, [&]() {
      mask.copy_to_device(target); // include H2D copying time
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

  // Bilateral filter
  Func Bilateral(Func f) {
    using Halide::_;
    Func d, p, out;
    float c_r = 0.5f / (sigma_s * sigma_s);
    RDom dom(mask); // a reduction domain of 13x13

    Expr diff = f(x + dom.x, y + dom.y) - f(x, y);
    Expr sp = diff * diff * -c_r;
    Expr s = exp(sp) * mask(dom.x, dom.y);
    d(x, y) += s;
    p(x, y) += s * f(x + dom.x, y + dom.y);
    out(x, y) = p(x, y) / d(x, y) + 0.5f;
    return out;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int sigma_s = 13;

  // Bilateral mask 13x13
  const float coef[sigma_s][sigma_s] = {
      {0.018316f, 0.033746f, 0.055638f, 0.082085f, 0.108368f, 0.128022f,
       0.135335f, 0.128022f, 0.108368f, 0.082085f, 0.055638f, 0.033746f,
       0.018316f},
      {0.033746f, 0.062177f, 0.102512f, 0.151240f, 0.199666f, 0.235877f,
       0.249352f, 0.235877f, 0.199666f, 0.151240f, 0.102512f, 0.062177f,
       0.033746f},
      {0.055638f, 0.102512f, 0.169013f, 0.249352f, 0.329193f, 0.388896f,
       0.411112f, 0.388896f, 0.329193f, 0.249352f, 0.169013f, 0.102512f,
       0.055638f},
      {0.082085f, 0.151240f, 0.249352f, 0.367879f, 0.485672f, 0.573753f,
       0.606531f, 0.573753f, 0.485672f, 0.367879f, 0.249352f, 0.151240f,
       0.082085f},
      {0.108368f, 0.199666f, 0.329193f, 0.485672f, 0.641180f, 0.757465f,
       0.800737f, 0.757465f, 0.641180f, 0.485672f, 0.329193f, 0.199666f,
       0.108368f},
      {0.128022f, 0.235877f, 0.388896f, 0.573753f, 0.757465f, 0.894839f,
       0.945959f, 0.894839f, 0.757465f, 0.573753f, 0.388896f, 0.235877f,
       0.128022f},
      {0.135335f, 0.249352f, 0.411112f, 0.606531f, 0.800737f, 0.945959f,
       1.000000f, 0.945959f, 0.800737f, 0.606531f, 0.411112f, 0.249352f,
       0.135335f},
      {0.128022f, 0.235877f, 0.388896f, 0.573753f, 0.757465f, 0.894839f,
       0.945959f, 0.894839f, 0.757465f, 0.573753f, 0.388896f, 0.235877f,
       0.128022f},
      {0.108368f, 0.199666f, 0.329193f, 0.485672f, 0.641180f, 0.757465f,
       0.800737f, 0.757465f, 0.641180f, 0.485672f, 0.329193f, 0.199666f,
       0.108368f},
      {0.082085f, 0.151240f, 0.249352f, 0.367879f, 0.485672f, 0.573753f,
       0.606531f, 0.573753f, 0.485672f, 0.367879f, 0.249352f, 0.151240f,
       0.082085f},
      {0.055638f, 0.102512f, 0.169013f, 0.249352f, 0.329193f, 0.388896f,
       0.411112f, 0.388896f, 0.329193f, 0.249352f, 0.169013f, 0.102512f,
       0.055638f},
      {0.033746f, 0.062177f, 0.102512f, 0.151240f, 0.199666f, 0.235877f,
       0.249352f, 0.235877f, 0.199666f, 0.151240f, 0.102512f, 0.062177f,
       0.033746f},
      {0.018316f, 0.033746f, 0.055638f, 0.082085f, 0.108368f, 0.128022f,
       0.135335f, 0.128022f, 0.108368f, 0.082085f, 0.055638f, 0.033746f,
       0.018316f}};

  // Initialize with random image
  Buffer<float> input(width, height);
  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = rand() & 0xfff;
    }
  }

  Buffer<float> mask(sigma_s, sigma_s);
  for (int y = 0; y < mask.height(); y++) {
    for (int x = 0; x < mask.width(); x++) {
      mask(x, y) = coef[x][y];
    }
  }

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input, mask, sigma_s);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
