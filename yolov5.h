#ifndef YOLOV5_H_
#define YOLOV5_H_

#include <chrono>
#include <cmath>
#include <iostream>

#include <inference_engine.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace yolov5 {

struct Object {
  cv::Rect_<float> rect;
  int label;
  float prob;
};

class Yolov5 {
 public:
  Yolov5(std::string xml_path, float conf_thresh, float nms_thresh, 
         int height, int width);
  std::vector<Object> Detect(cv::Mat& img);

 private:
  cv::Mat Resize(cv::Mat& img);

  std::vector<Object> GenerateProposals(std::vector<float>& anchors, 
                                       int stride, 
                                       const InferenceEngine::Blob::Ptr &blob);
  
  int Draw(cv::Mat& rgb, const std::vector<Object>& objects);     
  
  const std::string xml_path_;
  const float conf_thresh_;
  const float obj_score_thresh_;
  const float nms_thresh_;
  std::vector<std::vector<float>> anchors_;
  const int height_;
  const int width_;
  float scale_;

  std::string input_name_;
  InferenceEngine::ExecutableNetwork network_;
  InferenceEngine::OutputsDataMap outputinfo_;
};

inline std::chrono::steady_clock::time_point Now() {
  return std::chrono::steady_clock::now();
}

inline std::chrono::duration<double>::rep Duration(
    const std::chrono::steady_clock::time_point& t2, 
    const std::chrono::steady_clock::time_point& t1) {
  std::chrono::duration<double> interval = t2 - t1;
  return interval.count();
}

}  // namespace yolov5

#endif  // YOLOV5_H_