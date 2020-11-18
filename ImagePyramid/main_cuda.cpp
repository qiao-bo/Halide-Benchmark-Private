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
  Func gPyramid[LEVEL];
  Func lPyramid[LEVEL];
  Func outLPyramid[LEVEL];
  Func BLPyramid[LEVEL];
  Buffer<float> input;
  Buffer<float> mask;
  Buffer<float> maskGaus;
  float sigma_s;

  PipelineClass(Buffer<float> in, Buffer<float> msk, Buffer<float> mskg,
                float sigma_s)
      : input(in), mask(msk), maskGaus(mskg), sigma_s(sigma_s) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);

    // Make the Gaussian pyramid of the input
    gPyramid[0](x, y) = gray(x, y);
    for (int j = 1; j < LEVEL; j++) {
      gPyramid[j](x, y) = downsample(gPyramid[j - 1])(x, y);
    }

    // Get its laplacian pyramid
    lPyramid[LEVEL - 1](x, y) = gPyramid[LEVEL - 1](x, y);
    for (int j = LEVEL - 2; j >= 0; j--) {
      lPyramid[j](x, y) = gPyramid[j](x, y) - upsample(gPyramid[j + 1])(x, y);
    }

    BLPyramid[0](x, y) = lPyramid[0](x, y);
    // Level Processing
    for (int j = 1; j < LEVEL; j++) {
      BLPyramid[j](x, y) = bilateral(lPyramid[j])(x, y);
    }

    // Make the Gaussian pyramid of the output
    outLPyramid[LEVEL - 1](x, y) = BLPyramid[LEVEL - 1](x, y);
    for (int j = LEVEL - 2; j >= 0; j--) {
      Expr outL = BLPyramid[j](x, y) * cast<float>(0.5f);
      outLPyramid[j](x, y) = upsample(BLPyramid[j + 1])(x, y) + outL;
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

    output.set_estimate(x, 0, WIDTH).set_estimate(y, 0, HEIGHT);

    // Auto schedule the pipeline
    Pipeline p(output);
    p.auto_schedule(target);
    output.compile_jit(target);

    // Test the performance of the scheduled pipeline.
    Buffer<float> out(input.width(), input.height());
    double best_auto = benchmark(10, 3, [&]() {
      maskGaus.copy_to_device(target); // include H2D copying time
      mask.copy_to_device(target);
      input.copy_to_device(target);
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

  // Downsample with a 3x3 Gaussian filter
  Func downsample(Func f) {
    using Halide::_;
    Func blur, subsample;
    RDom dom(maskGaus); // a reduction domain of 3x3
    Expr conv = f(x + dom.x, y + dom.y) * maskGaus(dom.x, dom.y);
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

  // Bilateral filter for level processing
  Func bilateral(Func f) {
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
  const int size_x = 3;
  const int size_y = 3;
  const int sigma_s = 13;

  // Gaussian filter mask
  const float coefGaus[size_x][size_y] = {0.057118f, 0.124758f, 0.057118f,
                                          0.124758f, 0.272496f, 0.124758f,
                                          0.057118f, 0.124758f, 0.057118f};
  // Bilateral mask
  const float coefBil[sigma_s][sigma_s] = {
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

  Buffer<float> maskg(size_x, size_y);
  for (int y = 0; y < maskg.height(); y++) {
    for (int x = 0; x < maskg.width(); x++) {
      maskg(x, y) = coefGaus[x][y];
    }
  }
  Buffer<float> maskb(sigma_s, sigma_s);
  for (int y = 0; y < maskb.height(); y++) {
    for (int x = 0; x < maskb.width(); x++) {
      maskb(x, y) = coefBil[x][y];
    }
  }

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input, maskb, maskg, sigma_s);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
