#include "yolov5.h"

int main() {
  yolov5::Yolov5 yolov5("../yolov5s_openvino_model_384_640/yolov5s.xml", 0.25,
                        0.45, 384, 640);
  cv::Mat img = cv::imread("../images/zidane.jpg");
  auto start = yolov5::Now();
  std::vector<yolov5::Object> objects = yolov5.Detect(img);
  auto end = std::chrono::steady_clock::now();
  std::cout << "total: "<< yolov5::Duration(yolov5::Now(), start) <<"s" 
            << std::endl;
  cv::imwrite("result.jpg", img);
  std::cout << "save to result.jpg" << std::endl;
  cv::imshow("result", img);
  cv::waitKey(0);
}
