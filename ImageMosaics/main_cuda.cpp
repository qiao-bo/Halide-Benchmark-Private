#include <iostream>
#include <limits>

#include "Halide.h"
#include "halide_benchmark.h"

#define WIDTH 512
#define HEIGHT 512
#define LEVEL 8

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output;
  Func gPyramid1[LEVEL];
  Func gPyramid2[LEVEL];
  Func lPyramid1[LEVEL];
  Func lPyramid2[LEVEL];
  Func lPyramid[LEVEL];
  Func outLPyramid[LEVEL];
  Buffer<float> input1;
  Buffer<float> input2;
  Buffer<float> mask;

  PipelineClass(Buffer<float> in1, Buffer<float> in2, Buffer<float> mask)
      : input1(in1), input2(in2), mask(mask) {
    // Set a boundary condition
    Func gray1 = BoundaryConditions::repeat_edge(input1);
    Func gray2 = BoundaryConditions::repeat_edge(input2);

    // Make the Gaussian pyramid of the input 1
    gPyramid1[0](x, y) = gray1(x, y);
    for (int j = 1; j < LEVEL; j++) {
      gPyramid1[j](x, y) = downsample(gPyramid1[j - 1])(x, y);
    }
    // Get its laplacian pyramid
    lPyramid1[LEVEL - 1](x, y) = gPyramid1[LEVEL - 1](x, y);
    for (int j = LEVEL - 2; j >= 0; j--) {
      lPyramid1[j](x, y) = gPyramid1[j](x, y) - upsample(gPyramid1[j + 1])(x, y);
    }

    // Make the Gaussian pyramid of the input 2
    gPyramid2[0](x, y) = gray2(x, y);
    for (int j = 1; j < LEVEL; j++) {
      gPyramid2[j](x, y) = downsample(gPyramid2[j - 1])(x, y);
    }
    // Get its laplacian pyramid
    lPyramid2[LEVEL - 1](x, y) = gPyramid2[LEVEL - 1](x, y);
    for (int j = LEVEL - 2; j >= 0; j--) {
      lPyramid2[j](x, y) = gPyramid2[j](x, y) - upsample(gPyramid2[j + 1])(x, y);
    }

    // Level Processing
    for (int j = 0; j < LEVEL; j++) {
      lPyramid[j](x, y) = levelMerge(lPyramid1[j], lPyramid2[j])(x, y);
    }

    // Make the Gaussian pyramid of the output
    outLPyramid[LEVEL - 1](x, y) = lPyramid[LEVEL - 1](x, y);
    for (int j = LEVEL - 2; j >= 0; j--) {
      Expr outL = lPyramid[j](x, y) * cast<float>(0.5f);
      outLPyramid[j](x, y) = upsample(lPyramid[j + 1])(x, y) + outL;
    }
    output(x, y) = outLPyramid[0](x, y);
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
    Buffer<float> out(input1.width(), input1.height());
    double best_auto = benchmark(10, 3, [&]() {
      mask.copy_to_device(target); // include H2D copying time
      input1.copy_to_device(target);
      input2.copy_to_device(target);
      p.realize(out);
      out.copy_to_host(); // include D2H copying time
      out.device_sync();
    });
    printf("Auto-tuned time: %gms\n", best_auto * 1e3);

    return true;
  }

private:
  Var x, y, c, k;
  Target target;

  // Downsample
  Func downsample(Func f) {
    using Halide::_;
    Func blur, subsample;
    RDom dom(mask); // a reduction domain of 3x3
    Expr conv = f(x + dom.x, y + dom.y) * mask(dom.x, dom.y);
    blur(x, y) += conv;
    subsample(x, y) = blur(2 * x, 2 * y);
    return subsample;
  }

  // Upsample using bilinear interpolation
  Func upsample(Func f) {
    using Halide::_;
    Func upx, upy;
    upx(x, y) = 0.25f * f((x / 2) - 1 + 2 * (x % 2), y) + 0.75f * f(x / 2, y);
    upy(x, y) = 0.25f * upx(x, (y / 2) - 1 + 2 * (y % 2)) + 0.75f * upx(x, y / 2);
    return upy;
  }

  // mosaics level merging
  Func levelMerge(Func f1, Func f2) {
    using Halide::_;
    Func out;
    out(x, y) = select(x < (input1.width() / 2), f1(x, y), f2(x, y));
    return out;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = 3;
  const int size_y = 3;

  // Gaussian filter mask
  const float coef[size_x][size_y] = {0.057118f, 0.124758f, 0.057118f,
                                      0.124758f, 0.272496f, 0.124758f,
                                      0.057118f, 0.124758f, 0.057118f};

  // Initialize with random image
  Buffer<float> input1(width, height);
  Buffer<float> input2(width, height);
  for (int y = 0; y < input1.height(); y++) {
    for (int x = 0; x < input1.width(); x++) {
      input1(x, y) = rand() & 0xfff;
      input2(x, y) = rand() & 0xfff;
    }
  }

  Buffer<float> mask(size_x, size_y);
  for (int y = 0; y < mask.height(); y++) {
    for (int x = 0; x < mask.width(); x++) {
      mask(x, y) = coef[x][y];
    }
  }

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input1, input2, mask);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
