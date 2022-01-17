LLVM_REPO=3rdparty/llvm-project
BUILD_DIR=build

rm -rf $BUILD_DIR
mkdir $BUILD_DIR

pushd $BUILD_DIR
cmake -G Ninja .. \
  -DLLVM_DIR=$LLVM_REPO/build/lib/cmake/llvm  \
   -DMLIR_DIR=$LLVM_REPO/build/lib/cmake/mlir


cmake --build .  --target ToyCh2OpsIncGen
popd