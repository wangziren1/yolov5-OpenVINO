# Introduction
OpenVINO inference for yolov5 in cpp.

# Dependencies
1. openvino 2021.4
2. opencv

You can follow https://docs.openvino.ai/2021.4/openvino_docs_install_guides_installing_openvino_linux.html to install the above two libraries.
# OpenVINO export 
```
cd yolov5
python3 export.py --weights yolov5s.pt --img 384 640 --train --simplify --include onnx openvino
option:
--img: openvino network inference image size(height, width)
```
You can adjust `--img` as small as possible and keep it a multiple of 16, so as to decrease computation and increase inference speed. For example, the height and width of origin image are 720 and 1280. If `--img` is set to 384 640, the image will be resized to 360\*640 and then enlarged to 384\*640. If `--img` is set to 640 640, the image will be resized to 360\*640 and then enlarged to 640\*640. The first one is prefered because of its fast inference speed.
# Build for linux
first set up environment
```
source /opt/intel/openvino_2021/bin/setupvars.sh
source /opt/intel/openvino_2021/opencv/setupvars.sh
```
```
git clone https://github.com/wangziren1/yolov5-OpenVINO.git
cd yolov5-OpenVINO
mkdir build && cd build
cmake ..
make
```
# Run
```
./demo
```
The result.jpg will be saved in build directory.
