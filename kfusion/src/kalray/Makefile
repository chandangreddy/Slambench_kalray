platforms := simulator hardware

HOME_DIR := /home/accesscore/Slambench/SLAMBench_1_1

mppa-bin := slambench_multibin

cflags := -I. -I${HOME_DIR}/kfusion/include
lflags := -lm 

slambench_multibin-objs := slambench_io slambench_cluster 

cluster-bin := slambench_cluster
slambench_cluster-srcs := cc_main.cpp kernels_cc.cpp

io-bin := slambench_io
slambench_io-srcs := kernels.cpp benchmark.cpp  ktime.cpp reader.cpp

slambench_io-cflags := -DCLUSTER_BIN_NAME=\"slambench_cluster\" 

#pre-build-hooks := process_header

include $(K1_TOOLCHAIN_DIR)/share/make/Makefile.mppaipc

#slambench_header.h:
#   sed  "s/cmakedefine/define/g" slambench.h.cmake > slambench_header.h
#   #   sed -i "s/@IO_EXECUTABLE@/slambench_io/g" slambench_header.h
#   sed -i "s/@CLUSTER1_EXECUTABLE@_0/slambench_cluster/g" slambench_header.h
#
#   #process_header: slambench_header.h

run_sim:
	k1-mppa -- ./output/bin/slambench_multibin.mpk -s 4.8 -p 0.34,0.5,0.24 -z 4 -c 2 -r 1 -k 481.2,480,320,240 -i ../../../living_room_traj2_loop.raw -o benchmark.2.kalray.log 2>kernels.2.kalray.log

run_mppa:
	k1-jtag-runner --multibinary output/bin/slambench_multibin.mpk  --exec-multibin=IODDR0:slambench_io 

clean1:
	rm -fr output/  *~

