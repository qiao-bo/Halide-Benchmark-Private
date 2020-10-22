#include "Halide.h"
#include "halide_benchmark.h"
#include <iostream>
#include <limits>

#define WIDTH 256
#define HEIGHT 368
#define PARN 10

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output[PARN];
  Func avgImg[PARN];
  Buffer<float> input;
  Buffer<float> maskAvg;
  int gain = 2;
  float gamma = 0.6;

  PipelineClass(Buffer<float> in, Buffer<float> mask) : input(in), maskAvg(mask) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);

    // Average Filter
    for (int n = 0; n < PARN; n++) {
      avgImg[n](x, y) = AverageFilter(gray)(x, y);
    }

    // Global Gain and Gamma Correction
    for (int n = 0; n < PARN; n++) {
      output[n](x, y) = pow(avgImg[n](x, y) * gain, gamma);
    }
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
    for (int n = 0; n < PARN; n++) {
      output[n].estimate(x, 0, WIDTH).estimate(y, 0, HEIGHT);
    }
    Pipeline p({output[0], output[1], output[2], output[3], output[4], output[5],
                output[6], output[7], output[8], output[9]});
    p.auto_schedule(target);
    for (int n = 0; n < PARN; n++) {
      output[n].compile_jit(target);
    }

    // Test the performance of the scheduled pipeline.
    Buffer<float> out0(input.width(), input.height());
    Buffer<float> out1(input.width(), input.height());
    Buffer<float> out2(input.width(), input.height());
    Buffer<float> out3(input.width(), input.height());
    Buffer<float> out4(input.width(), input.height());
    Buffer<float> out5(input.width(), input.height());
    Buffer<float> out6(input.width(), input.height());
    Buffer<float> out7(input.width(), input.height());
    Buffer<float> out8(input.width(), input.height());
    Buffer<float> out9(input.width(), input.height());

    // Timing code
    double best_auto = benchmark(10, 3, [&]() {
      maskAvg.copy_to_device(target);
      input.copy_to_device(target);
      p.realize({out0, out1, out2, out3, out4, out5, out6, out7, out8, out9});
      out0.copy_to_host();
      out1.copy_to_host();
      out2.copy_to_host();
      out3.copy_to_host();
      out4.copy_to_host();
      out5.copy_to_host();
      out6.copy_to_host();
      out7.copy_to_host();
      out8.copy_to_host();
      out9.copy_to_host();

      out0.device_sync();
      out1.device_sync();
      out2.device_sync();
      out3.device_sync();
      out4.device_sync();
      out5.device_sync();
      out6.device_sync();
      out7.device_sync();
      out8.device_sync();
      out9.device_sync();
    });
    printf("Auto-tuned time: %gms\n", best_auto * 1e3);

    return true;
  }

private:
  Var x, y;
  Target target;

  // 3x3 Gaussian filter
  Func AverageFilter(Func f) {
    using Halide::_;
    Func avg;
    RDom dom(maskAvg); // a reduction domain of 3x3
    Expr conv = f(x + dom.x, y + dom.y) * maskAvg(dom.x, dom.y);
    avg(x, y) += conv;
    return avg;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = 3;
  const int size_y = 3;

  // Average filter mask
  const float coefAvg[size_x][size_y] = {0.111111f, 0.111111f, 0.111111f,
                                         0.111111f, 0.111111f, 0.111111f,
                                         0.111111f, 0.111111f, 0.111111f};

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
      mask(x, y) = coefAvg[x][y];
    }
  }

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input, mask);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
