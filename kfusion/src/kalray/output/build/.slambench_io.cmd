cmd_output/build/slambench_io := /home/accesscore/files/2.0.0-eng2-rc7/install/usr/local/k1tools/bin/k1-g++ -o output/build/slambench_io   output/build/slambench_io_build/kernels.cpp.o output/build/slambench_io_build/benchmark.cpp.o output/build/slambench_io_build/ktime.cpp.o output/build/slambench_io_build/reader.cpp.o   -mcore=k1io -mboard=developer -lm  -lmppaipc   -mos=rtems 
