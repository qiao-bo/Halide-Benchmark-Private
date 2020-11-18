# halide-app-private
This repository contains non-expert [Halide](https://halide-lang.org/) implementations of some commonly seen image processing applications using Halide auto-schedulers described in [this paper](https://dl.acm.org/doi/10.1145/2897824.2925952) for GPU backend.

Some experiments on RTX2080, Ubuntu 18.04, CUDA 11.1, Clang/LLVM 10.0, Halide master branch @v10.0.0

| app                 | time (ms)   |
|:------------------- | -----------:|
| Bilateral           |  39.6088    |
| Gaussian            |   0.2277    |
| HarrisCorner        |   39.7723   |
| ImageEnhance        |   2.6234    |
| ImageMosaics        |   0.6578    |
| ImagePyramid        |   2.9069    |
| Laplace             |   0.4219    |
| NightFilter         | 331.9770    |
| NightFilterPipeline |  29.9191    |
| Prewitt             |   0.4985    |
| ReduceSum           |   0.0036    |
| ShiTomasiFeature    |   3.1689    |
| Sobel               |   0.5059    |
| Unsharp             |   0.1136    |
