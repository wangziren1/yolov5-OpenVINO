cmake_minimum_required(VERSION 3.5)
project(yolov5_openvino)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#查找必要的依赖包
find_package(OpenCV REQUIRED)
find_package(InferenceEngine)
find_package(ngraph REQUIRED)


# 编译detector.so
add_library (yolov5 SHARED yolov5.cc)
target_include_directories(yolov5
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${InferenceEngine_INCLUDE_DIRS}
)
target_link_libraries(yolov5
        ${OpenCV_LIBS}
        ${InferenceEngine_LIBRARIES}
        ${ngraph_LIBRARIES}
)

add_executable(demo demo.cc)
target_link_libraries(demo yolov5)