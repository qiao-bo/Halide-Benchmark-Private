#include "Halide.h"
#include "halide_benchmark.h"
#include <iostream>
#include <limits>

#define WIDTH 1024
#define HEIGHT 1024

using namespace Halide;
using namespace Halide::Tools;

class PipelineClass {
public:
  Func output;
  Func intermBuf3;
  Func intermBuf5;
  Func intermBuf9;
  Func intermBuf17;
  Buffer<uint> input;
  Buffer<float> mask3;
  Buffer<float> mask5;
  Buffer<float> mask9;
  Buffer<float> mask17;

  PipelineClass(Buffer<uint> in, Buffer<float> msk3, Buffer<float> msk5,
                Buffer<float> msk9, Buffer<float> msk17)
      : input(in), mask3(msk3), mask5(msk5), mask9(msk9), mask17(msk17) {
    // Set a boundary condition
    Func gray = BoundaryConditions::repeat_edge(input);

    // Astrous Filter (Iteratively)
    intermBuf3(x, y) = AtrousFilter(gray, mask3)(x, y);
    intermBuf5(x, y) = AtrousFilter(intermBuf3, mask5)(x, y);
    intermBuf9(x, y) = AtrousFilter(intermBuf5, mask9)(x, y);
    intermBuf17(x, y) = AtrousFilter(intermBuf9, mask17)(x, y);

    // Tone mapping
    output(x, y) = Scoto(intermBuf17)(x, y);
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
    Buffer<uint> out(input.width(), input.height());

    mask3.copy_to_device(target);
    mask5.copy_to_device(target);
    mask9.copy_to_device(target);
    mask17.copy_to_device(target);
    input.copy_to_device(target);
    double best_auto = benchmark(10, 3, [&]() {
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

  // Atrous filter
  Func AtrousFilter(Func f, Buffer<float> &mask) {
    using Halide::_;
    Func out;
    Func sum_weight;
    Func sum_r;
    Func sum_g;
    Func sum_b;
    RDom dom(mask);

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

  // Atrous masks with holes
  const float coef3[3][3] = {0.057118f, 0.124758f, 0.057118f,
                             0.124758f, 0.272496f, 0.124758f,
                             0.057118f, 0.124758f, 0.057118f};
  const float coef5[5][5] = {
      0.057118f, 0.0f, 0.124758f, 0.0f, 0.057118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.124758f, 0.0f, 0.272496f, 0.0f, 0.124758f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.057118f, 0.0f, 0.124758f, 0.0f, 0.057118f};
  const float coef9[9][9] = {
      0.057118f, 0.0f, 0.0f, 0.0f, 0.124758f, 0.0f, 0.0f, 0.0f, 0.057118f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.124758f, 0.0f, 0.0f, 0.0f, 0.272496f, 0.0f, 0.0f, 0.0f, 0.124758f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f,      0.0f, 0.0f, 0.0f, 0.0f,
      0.057118f, 0.0f, 0.0f, 0.0f, 0.124758f, 0.0f, 0.0f, 0.0f, 0.057118f};
  const float coef17[17][17] = {
      0.057118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.124758f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.057118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.124758f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.272496f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.124758f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.0f,      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.057118f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.124758f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      0.057118f};

  // Initialize with random image
  Buffer<uint> input(width, height);
  for (int y = 0; y < input.height(); y++) {
    for (int x = 0; x < input.width(); x++) {
      input(x, y) = rand() & 0xfff;
    }
  }

  // masks
  Buffer<float> mask3(3, 3);
  for (int y = 0; y < mask3.height(); y++) {
    for (int x = 0; x < mask3.width(); x++) {
      mask3(x, y) = coef3[x][y];
    }
  }
  Buffer<float> mask5(5, 5);
  for (int y = 0; y < mask5.height(); y++) {
    for (int x = 0; x < mask5.width(); x++) {
      mask5(x, y) = coef5[x][y];
    }
  }
  Buffer<float> mask9(9, 9);
  for (int y = 0; y < mask9.height(); y++) {
    for (int x = 0; x < mask9.width(); x++) {
      mask9(x, y) = coef9[x][y];
    }
  }
  Buffer<float> mask17(17, 17);
  for (int y = 0; y < mask17.height(); y++) {
    for (int x = 0; x < mask17.width(); x++) {
      mask17(x, y) = coef17[x][y];
    }
  }

  printf("Running Halide pipeline...\n");
  PipelineClass pipe(input, mask3, mask5, mask9, mask17);
  if (!pipe.test_performance()) {
    printf("Scheduling failed\n");
  }
  return 0;
}
