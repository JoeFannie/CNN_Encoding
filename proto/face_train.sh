#!/bin/bash

/home/caojiajiong/workspace/caffe-windows/build/tools/caffe train \
	-model "/home/caojiajiong/workspace/CNN_Encoding/proto/intra_mode_v1.prototxt" \
	-solver "/home/caojiajiong/workspace/CNN_Encoding/proto/face_solver.prototxt" \
	-gpu 3 
