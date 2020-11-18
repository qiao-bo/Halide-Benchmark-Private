#include "Halide.h"
#include "halide_benchmark.h"
#include <iostream>
#include <limits>

#define WIDTH 128
#define HEIGHT 184
#define NPIPE 20

using namespace Halide;
using namespace Halide::Tools;
using std::vector;

class PipelineClass {
public:
  std::vector<Func> output;
  Func bufImg[NPIPE];
  Func bufOut[NPIPE];
  Buffer<uint> input;
  Buffer<float> mask;

  PipelineClass(Buffer<uint> in, Buffer<float> mask) : input(in), mask(mask) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);

    // Atrous Filter
    for (int n = 0; n < NPIPE; n++) {
      bufImg[n](x, y) = AtrousFilter(gray)(x, y);
    }
    // Scoto tone mapping
    for (int n = 0; n < NPIPE; n++) {
      bufOut[n](x, y) = Scoto(bufImg[n])(x, y);
    }

    for (int n = 0; n < NPIPE; n++) {
      output.push_back(bufOut[n]);
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
    for (int n = 0; n < NPIPE; n++) {
      output[n].set_estimate(x, 0, WIDTH).set_estimate(y, 0, HEIGHT);
    }
    Pipeline p(output);
    p.auto_schedule(target);
    for (int n = 0; n < NPIPE; n++) {
      output[n].compile_jit(target);
    }

    // Test the performance of the scheduled pipeline.
    std::vector<Buffer<>> outputBufs;
    for (int n = 0; n < NPIPE; n++) {
      Buffer<uint> out(input.width(), input.height());
      outputBufs.push_back(out);
    }

    Realization r(outputBufs);
    double best_auto = benchmark(10, 3, [&]() {
      mask.copy_to_device(target); // include H2D copying time
      input.copy_to_device(target);
      p.realize(r);
      for (int n = 0; n < NPIPE; n++) { // include D2H copying time
        outputBufs[n].copy_to_host();
      }
      for (int n = 0; n < NPIPE; n++) {
        outputBufs[n].device_sync();
      }
    });
    printf("Auto-tuned time: %gms\n", best_auto * 1e3);

    return true;
  }

private:
  Var x, y;
  Target target;

  // 9x9 Atrous filter
  Func AtrousFilter(Func f) {
    using Halide::_;
    Func out;
    Func sum_weight;
    Func sum_r;
    Func sum_g;
    Func sum_b;
    RDom dom(mask); // a reduction domain of 9x9

    // unpack center
    Expr val = f(x, y);
    Expr rin = val & 0xff;
    Expr gin = (val >> 8) & 0xff;
    Expr bin = (val >> 16) & 0xff;
    rin /= 255.0f;
    gin /= 255.0f;
    bin /= 255.0f;

    Expr pixel = f(x + dom.x, y + dom.y);
    Expr rpixel = pixel & 0xff;
    Expr gpixel = (pixel >> 8) & 0xff;
    Expr bpixel = (pixel >> 16) & 0xff;
    rpixel /= 255.0f;
    gpixel /= 255.0f;
    bpixel /= 255.0f;

    Expr rd = rpixel - rin;
    Expr gd = gpixel - gin;
    Expr bd = bpixel - bin;

    Expr weight = rd * rd + gd * gd + bd * bd;
    // Expf 256
    Expr xx = 1.0f + weight / 256.0f;
    xx *= xx;
    xx *= xx;
    xx *= xx;
    xx *= xx;
    xx *= xx;
    xx *= xx;
    xx *= xx;
    xx *= xx;
    weight = Halide::select(xx > 1.0f, 1.0f, xx);

    sum_weight(x, y) += weight * mask(dom.x, dom.y);
    sum_r(x, y) += rpixel * weight;
    sum_g(x, y) += gpixel * weight;
    sum_b(x, y) += bpixel * weight;

    Expr rout = sum_r(x, y) * 255.0f / sum_weight(x, y);
    Expr gout = sum_g(x, y) * 255.0f / sum_weight(x, y);
    Expr bout = sum_b(x, y) * 255.0f / sum_weight(x, y);

    Expr crout = cast(UInt(32), rout);
    Expr cgout = cast(UInt(32), gout) << 8;
    Expr cbout = cast(UInt(32), bout) << 16;
    Expr cwout = cast(UInt(32), 255) << 24;

    Expr ucout = crout | cgout | cbout | cwout;
    Expr output = cast(UInt(32), ucout);
    out(x, y) = output;
    return out;
  }

  Func Scoto(Func f) {
    using Halide::_;
    Func out;

    // unpack center
    Expr val = f(x, y);
    Expr rin = val & 0xff;
    Expr gin = (val >> 8) & 0xff;
    Expr bin = (val >> 16) & 0xff;

    Expr X = 0.5149f * rin + 0.3244f * gin + 0.1607f * bin;
    Expr Y = (0.2654f * rin + 0.6704f * gin + 0.0642f * bin) / 3.0f;
    Expr Z = 0.0248f * rin + 0.1248f * gin + 0.8504f * bin;
    Expr V = Y * (((((Y + Z) / X) + 1.0f) * 1.33f) - 1.68f);
    Expr W = X + Y + Z;
    Expr luma = 0.2126f * rin + 0.7152f * gin + 0.0722f * bin;
    Expr s = 0.0f; // luma / 2.0f;
    Expr x1 = X / W;
    Expr y1 = Y / W;

    x1 = ((1.0f - s) * 0.25f) + (s * x1);
    y1 = ((1.0f - s) * 0.25f) + (s * y1);
    Y = (V * 0.4468f * (1.0f - s)) + (s * Y);
    X = (x1 * Y) / y1;
    Z = (X / y1) - X - Y;

    Expr r = 2.562263f * X + -1.166107f * Y + -0.396157f * Z;
    Expr g = -1.021558f * X + 1.977828f * Y + 0.043730f * Z;
    Expr b = 0.075196f * X + -0.256248f * Y + 1.181053f * Z;

    r = Halide::select(r > 255.0f, 255.0f, r);
    r = Halide::select(r < 0.0f, 0.0f, r);
    g = Halide::select(g > 255.0f, 255.0f, r);
    g = Halide::select(g < 0.0f, 0.0f, r);
    b = Halide::select(b > 255.0f, 255.0f, r);
    b = Halide::select(b < 0.0f, 0.0f, r);

    Expr crout = cast(UInt(32), r);
    Expr cgout = cast(UInt(32), g) << 8;
    Expr cbout = cast(UInt(32), b) << 16;
    Expr cwout = cast(UInt(32), 255) << 24;

    Expr ucout = crout | cgout | cbout | cwout;
    Expr output = cast(UInt(32), ucout);
    out(x, y) = output;
    return out;
  }
};

int main(int argc, char **argv) {
  const int width = WIDTH;
  const int height = HEIGHT;
  const int size_x = 9;
  const int size_y = 9;

  // night filter mask
  const float coef[size_x][size_y] = {
      0.057118f, 0.0f, 0.0f, 0.0f, 0.124758f, 0.0f, 0.0f, 0.0f, 0.057118f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.124758f, 0.0f, 0.0f, 0.0f, 0.272496f, 0.0f, 0.0f, 0.0f, 0.124758f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.057118f, 0.0f, 0.0f, 0.0f, 0.124758f, 0.0f, 0.0f, 0.0f, 0.057118f};

  // Initialize with random image
  Buffer<uint> input(width, height);
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
