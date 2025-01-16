git submodule update --init --recursive

# llvm 只需要编译一次

# mkdir third_party/llvm-project/build

# pushd third_party/llvm-project/build
# cmake -G "Unix Makefiles" \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DLLVM_ENABLE_PROJECTS="mlir" \
#   -DCMAKE_INSTALL_PREFIX=/path/to/install/llvm \
#   ../llvm

# make -j$(nproc)

popd

LLVM_REPO=third_party/llvm-project
BUILD_DIR=build

rm -rf $BUILD_DIR
mkdir $BUILD_DIR

pushd $BUILD_DIR
cmake -G Ninja .. \
  -DLLVM_DIR=$LLVM_REPO/build/lib/cmake/llvm  \
   -DMLIR_DIR=$LLVM_REPO/build/lib/cmake/mlir


cmake --build .  --target ToyCh2OpsIncGen
popd