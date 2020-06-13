#! /bin/bash

source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5

PLATFORM=$1
MODE=$2

if [[ $PLATFORM == "mac" ]]; then
    if [[ $MODE == 1 ]]; then
            python3 main.py -i CAM -m models/tf_ssd/frozen_inference_graph.xml -l /opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    elif [[ $MODE == 2 ]]; then
        python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tf_ssd/frozen_inference_graph.xml -l /opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    elif [[ $MODE == 3 ]]; then
        python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tf_ssd/frozen_inference_graph.xml -l /opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
      elif [[ $MODE == 4 ]]; then
        python3 main.py -i resources/testfile.txt -m model/FP16/frozen_inference_graph.xml -l /opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.3
    fi

elif [[ $PLATFORM == 'linux' ]]; then
    if [[ $MODE == 1 ]]; then
        python main.py -i CAM -m models/tf_ssd/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    elif [[ $MODE == 2 ]]; then
        python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/tf_ssd/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
    elif [[ $MODE == 3 ]]; then
        python main.py -i resources/test-image1.jpg -m models/tf_ssd/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
      elif [[ $MODE == 4 ]]; then
        python3 main.py -i resources/testfile.txt -m model/FP16/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.3
    fi
fi

