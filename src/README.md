# signum-plotter

## Features
- windows, linux, unix & macOS
- x86 32&64bit 
- direct and async i/o
- SIMD support: sse2, avx, avx2, avx512f
- gpu support
- fastest plotter there is

## Binary files and source code releases

https://github.com/signum-network/signum-plotter/releases

## Running the binaries

This is a command line tool, download the executable for your system and run it in a command prompt:

```shell
./signum-plotter --help
```

## Build from Sources

 - First you need to install a Rust stable toolchain, check https://www.rust-lang.org/tools/install.
 - Binaries are in **target/debug** or **target/release** depending on optimization.

``` shell
# build release (optimized) with GPU support:
cargo build --release [--features=opencl]

# build debug (unoptimized)
cargo build [--features=opencl]
```

## Forked from

This is a code fork from https://github.com/PoC-Consortium/engraver

