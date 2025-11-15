#!/bin/bash
# Pre-build verification script for mGBA Python bindings on M2 Mac

echo "=== Pre-Build Verification Script ==="
echo ""

# Check if we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ ERROR: Virtual environment is NOT activated!"
    echo "   Run: source ~/Documents/pokeagent-speedrun/.venv/bin/activate"
    exit 1
else
    echo "✅ Virtual environment: $VIRTUAL_ENV"
fi

# Check Python path
PYTHON_PATH=$(which python)
echo "✅ Python executable: $PYTHON_PATH"

if [[ "$PYTHON_PATH" != *".venv"* ]]; then
    echo "⚠️  WARNING: Python is not from .venv!"
    echo "   Expected path to contain '.venv'"
    echo "   Run: source ~/Documents/pokeagent-speedrun/.venv/bin/activate"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version)
echo "✅ Python version: $PYTHON_VERSION"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ ERROR: CMake not found!"
    echo "   Run: brew install cmake"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1)
echo "✅ CMake: $CMAKE_VERSION"

# Check required brew packages
echo ""
echo "Checking Homebrew packages..."
REQUIRED_PACKAGES=(cmake pkg-config qt@5 sdl2 mgba)
for package in "${REQUIRED_PACKAGES[@]}"; do
    if brew list "$package" &> /dev/null; then
        echo "✅ $package installed"
    else
        echo "❌ $package NOT installed"
        echo "   Run: brew install $package"
    fi
done

# Check cffi
echo ""
echo "Checking Python dependencies..."
if python -c "import cffi" 2>/dev/null; then
    echo "✅ cffi installed"
else
    echo "❌ cffi NOT installed"
    echo "   Run: pip install cffi"
fi

# Check mGBA library
echo ""
echo "Checking mGBA library..."
if [ -f "/opt/homebrew/lib/libmgba.0.10.dylib" ]; then
    echo "✅ mGBA library found: /opt/homebrew/lib/libmgba.0.10.dylib"
else
    echo "⚠️  mGBA 0.10 library not found, checking for other versions..."
    ls -la /opt/homebrew/lib/libmgba* 2>/dev/null || echo "❌ No mGBA library found!"
fi

# Print Python config info
echo ""
echo "=== Python Configuration ==="
python -c "
import sys
from sysconfig import get_paths

paths = get_paths()
print(f'Python prefix: {sys.prefix}')
print(f'Include dir: {paths[\"include\"]}')
print(f'Lib dir: {sys.prefix}/lib')
"

echo ""
echo "=== Verification Complete ==="
echo ""
echo "If all checks passed, you're ready to build mGBA!"
echo "Run these commands:"
echo ""
echo "  cd /tmp"
echo "  git clone https://github.com/mgba-emu/mgba.git"
echo "  cd mgba"
echo "  git checkout 0.10.5"
echo "  mkdir build && cd build"
echo "  cmake .. -DCMAKE_PREFIX_PATH=\$(brew --prefix qt@5) -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=\$(which python)"
echo "  make -j\$(sysctl -n hw.ncpu)"
echo ""