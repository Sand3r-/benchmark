#!/bin/sh

#rm -r paddle
#cp /data/mgallus/Sander/Paddle/paddle/ . -r
#cp /data/mgallus/Sander/Paddle/build_rel/paddle/fluid/platform/profiler* paddle/fluid/platform/

cd build && rm -r *
cmake -DUSE_GPU=OFF -DPADDLE_ROOT=/home/mgallus/src/Sander/Paddle/build_rel/fluid_install_dir -DCMAKE_BUILD_TYPE=Release -DUSE_PROFILER=OFF ..
make -j30

cd -
