#!/usr/bin/env bash
# Build whisper.cpp whisper-server with the Vulkan backend into vendor/whisper.cpp/build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor"
mkdir -p "$VENDOR"

if [[ ! -d "$VENDOR/whisper.cpp" ]]; then
  git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git "$VENDOR/whisper.cpp"
fi

# ggml-vulkan needs SPIRV-Headers; install a private copy when the distro package is absent.
SPIRV_PREFIX="$VENDOR/spirv-prefix"
if [[ ! -f "$SPIRV_PREFIX/include/spirv/unified1/spirv.hpp" ]]; then
  if [[ ! -d "$VENDOR/SPIRV-Headers" ]]; then
    git clone --depth 1 https://github.com/KhronosGroup/SPIRV-Headers.git "$VENDOR/SPIRV-Headers"
  fi
  cmake -S "$VENDOR/SPIRV-Headers" -B "$VENDOR/SPIRV-Headers/build" \
    -DCMAKE_INSTALL_PREFIX="$SPIRV_PREFIX" -DSPIRV_HEADERS_ENABLE_TESTS=OFF
  cmake --install "$VENDOR/SPIRV-Headers/build"
fi

cd "$VENDOR/whisper.cpp"
cmake -B build -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH="$SPIRV_PREFIX" \
  -DCMAKE_CXX_FLAGS="-I$SPIRV_PREFIX/include" -DCMAKE_C_FLAGS="-I$SPIRV_PREFIX/include"
cmake --build build -j "$(nproc)" --target whisper-server whisper-cli
ls -la build/bin/whisper-server
