#include "yolov5.h"

#include <iomanip>

namespace yolov5 {

static inline float intersection_area(const Object& a, const Object& b) {
  cv::Rect_<float> inter = a.rect & b.rect;
  return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left,
                                  int right) {
  int i = left;
  int j = right;
  float p = faceobjects[(left + right) / 2].prob;

  while (i <= j) {
    while (faceobjects[i].prob > p) i++;

    while (faceobjects[j].prob < p) j--;

    if (i <= j) {
      // swap
      std::swap(faceobjects[i], faceobjects[j]);

      i++;
      j--;
    }
  }

  if (left < j) qsort_descent_inplace(faceobjects, left, j);
  if (i < right) qsort_descent_inplace(faceobjects, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects) {
  if (faceobjects.empty()) return;

  qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects,
                              std::vector<int>& picked, float nms_threshold) {
  picked.clear();

  const int n = faceobjects.size();

  std::vector<float> areas(n);
  for (int i = 0; i < n; i++) {
    areas[i] = faceobjects[i].rect.width * faceobjects[i].rect.height;
  }

  for (int i = 0; i < n; i++) {
    const Object& a = faceobjects[i];

    int keep = 1;
    for (int j = 0; j < (int)picked.size(); j++) {
      const Object& b = faceobjects[picked[j]];

      // intersection over union
      float inter_area = intersection_area(a, b);
      float union_area = areas[i] + areas[picked[j]] - inter_area;
      // float IoU = inter_area / union_area
      if (inter_area / union_area > nms_threshold) keep = 0;
    }

    if (keep) picked.push_back(i);
  }
}

static inline float sigmoid(float x) {
  return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float inverse_sigmoid(float x) {
  return static_cast<float>(-(log(1.f / x - 1.f)));
}

std::vector<Object> Yolov5::GenerateProposals(
    std::vector<float>& anchors, int stride,
    const InferenceEngine::Blob::Ptr& blob) {
  InferenceEngine::SizeVector size_vector = blob->getTensorDesc().getDims();
  int height = size_vector[2];
  int width = size_vector[3];
  int item_size = size_vector[4];
  InferenceEngine::LockedMemory<const void> blobMapped =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(blob)->rmap();
  const float* output_blob = blobMapped.as<float*>();

  const int num_class = item_size - 5;
  const int num_anchors = anchors.size() / 2;

  std::vector<Object> objects;
  for (int n = 0; n < num_anchors; ++n)
    for (int i = 0; i < height; ++i)
      for (int j = 0; j < width; ++j) {
        float obj_score =
            output_blob[n * height * width * item_size + i * width * item_size +
                        j * item_size + 4];
        if (obj_score < obj_score_thresh_) continue;

        // find class index with max class score
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
          float score =
              output_blob[n * height * width * item_size +
                          i * width * item_size + j * item_size + k + 5];
          if (score > class_score) {
            class_index = k;
            class_score = score;
          }
        }
        float conf = sigmoid(obj_score) * sigmoid(class_score);
        if (conf < conf_thresh_) continue;

        float x = output_blob[n * height * width * item_size +
                              i * width * item_size + j * item_size + 0];
        float y = output_blob[n * height * width * item_size +
                              i * width * item_size + j * item_size + 1];
        float w = output_blob[n * height * width * item_size +
                              i * width * item_size + j * item_size + 2];
        float h = output_blob[n * height * width * item_size +
                              i * width * item_size + j * item_size + 3];
        // float w = 0;
        // float h = 0;
        x = (sigmoid(x) * 2 - 0.5 + j) * stride;
        y = (sigmoid(y) * 2 - 0.5 + i) * stride;
        w = pow(sigmoid(w) * 2, 2) * anchors[n * 2];
        h = pow(sigmoid(h) * 2, 2) * anchors[n * 2 + 1];

        float x0 = x - w * 0.5f;
        float y0 = y - h * 0.5f;
        float x1 = x + w * 0.5f;
        float y1 = y + h * 0.5f;

        Object obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
        obj.label = class_index;
        obj.prob = conf;

        objects.push_back(obj);
      }
  return objects;
}

Yolov5::Yolov5(std::string xml_path, float conf_thresh, float nms_thresh,
               int height, int width)
    : xml_path_(xml_path),
      conf_thresh_(conf_thresh),
      obj_score_thresh_(inverse_sigmoid(conf_thresh_)),
      nms_thresh_(nms_thresh),
      height_(height),
      width_(width) {
  anchors_ = std::vector<std::vector<float>>{
      std::vector<float>{10, 13, 16, 30, 33, 23},
      std::vector<float>{30, 61, 62, 45, 59, 119},
      std::vector<float>{116, 90, 156, 198, 373, 326}};

  InferenceEngine::Core ie;
  auto cnnNetwork = ie.ReadNetwork(xml_path_);
  //输入设置
  InferenceEngine::InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
  InferenceEngine::InputInfo::Ptr& input = inputInfo.begin()->second;
  input_name_ = inputInfo.begin()->first;
  input->setPrecision(InferenceEngine::Precision::FP32);
  input->getInputData()->setLayout(InferenceEngine::Layout::NCHW);
  InferenceEngine::ICNNNetwork::InputShapes inputShapes =
      cnnNetwork.getInputShapes();
  InferenceEngine::SizeVector& inSizeVector = inputShapes.begin()->second;
  cnnNetwork.reshape(inputShapes);
  //输出设置
  outputinfo_ = InferenceEngine::OutputsDataMap(cnnNetwork.getOutputsInfo());
  for (auto& output : outputinfo_) {
    output.second->setPrecision(InferenceEngine::Precision::FP32);
  }
  //获取可执行网络
  //_network =  ie.LoadNetwork(cnnNetwork, "GPU");
  network_ = ie.LoadNetwork(cnnNetwork, "CPU");
}

std::vector<Object> Yolov5::Detect(cv::Mat& img) {
  auto t1 = Now();
  float h = static_cast<float>(img.rows);
  float w = static_cast<float>(img.cols);
  int resize_h = 0;
  int resize_w = 0;
  if (w >= h) {
    scale_ = float(width_) / w;
    resize_w = width_;
    resize_h = int(scale_ * h);
  } else {
    scale_ = float(height_) / h;
    resize_h = height_;
    resize_w = int(scale_ * w);
  }
  cv::Mat img_resize;
  cv::resize(img, img_resize, cv::Size(resize_w, resize_h));
  std::cout << "resize to h w(" << resize_h << "," << resize_w << ")"
            << std::endl;
  cv::Mat input(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
  int top = (height_ - resize_h) / 2;
  int bottom = height_ - top - resize_h;
  int left = (width_ - resize_w) / 2;
  int right = width_ - left - resize_w;
  cv::copyMakeBorder(img_resize, input, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  std::cout << "resize: " << std::fixed << std::setprecision(5)
            << Duration(Now(), t1) << std::endl;

  t1 = Now();
  cv::Mat input_rgb;
  cv::cvtColor(input, input_rgb, cv::COLOR_BGR2RGB);
  std::cout << "convert color: " << Duration(Now(), t1) << std::endl;
  size_t img_size = height_ * width_;
  InferenceEngine::InferRequest infer_request = network_.CreateInferRequest();
  InferenceEngine::Blob::Ptr frameBlob = infer_request.GetBlob(input_name_);
  InferenceEngine::LockedMemory<void> blobMapped =
      InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
  float* blob_data = blobMapped.as<float*>();

  t1 = Now();
  uchar* p;
  for (size_t i = 0; i < height_; ++i) {
    p = input_rgb.ptr<uchar>(i);
    for (size_t j = 0; j < width_; ++j) {
      for (size_t c = 0; c < 3; ++c) {
        blob_data[c * img_size + i * width_ + j] = float(p[3 * j + c]) / 255.0f;
      }
    }
  }
  std::cout << "copy data: " << Duration(Now(), t1) << std::endl;

  t1 = Now();
  infer_request.Infer();
  std::cout << "infer: " << Duration(Now(), t1) << std::endl;

  t1 = Now();
  std::vector<Object> proposals;
  InferenceEngine::Blob::Ptr blob = infer_request.GetBlob("output");
  std::vector<Object> objects8 = GenerateProposals(anchors_[0], 8, blob);
  proposals.insert(proposals.end(), objects8.begin(), objects8.end());

  blob = infer_request.GetBlob("365");
  std::vector<Object> objects16 = GenerateProposals(anchors_[1], 16, blob);
  proposals.insert(proposals.end(), objects16.begin(), objects16.end());

  blob = infer_request.GetBlob("385");
  std::vector<Object> objects32 = GenerateProposals(anchors_[2], 32, blob);
  proposals.insert(proposals.end(), objects32.begin(), objects32.end());

  qsort_descent_inplace(proposals);

  std::vector<int> picked;
  nms_sorted_bboxes(proposals, picked, nms_thresh_);

  int count = picked.size();
  std::vector<Object> objects;
  objects.resize(count);
  for (int i = 0; i < count; i++) {
    objects[i] = proposals[picked[i]];

    // adjust offset to original unpadded
    float x0 = (objects[i].rect.x - left) / scale_;
    float y0 = (objects[i].rect.y - top) / scale_;
    float x1 = (objects[i].rect.x + objects[i].rect.width - left) / scale_;
    float y1 = (objects[i].rect.y + objects[i].rect.height - top) / scale_;

    // clip
    x0 = std::max(std::min(x0, (float)(w - 1)), 0.f);
    y0 = std::max(std::min(y0, (float)(h - 1)), 0.f);
    x1 = std::max(std::min(x1, (float)(w - 1)), 0.f);
    y1 = std::max(std::min(y1, (float)(h - 1)), 0.f);

    objects[i].rect.x = x0;
    objects[i].rect.y = y0;
    objects[i].rect.width = x1 - x0;
    objects[i].rect.height = y1 - y0;
  }
  std::cout << "postprocess: " << Duration(Now(), t1) << std::endl;

  t1 = Now();
  Draw(img, objects);
  std::cout << "draw: " << Duration(Now(), t1) << std::endl;

  return objects;
}

cv::Mat Yolov5::Resize(cv::Mat& img) {
  float h = static_cast<float>(img.rows);
  float w = static_cast<float>(img.cols);
  int resize_h = 0;
  int resize_w = 0;
  if (w >= h) {
    scale_ = float(width_) / w;
    resize_w = width_;
    resize_h = int(scale_ * h);
  } else {
    scale_ = float(height_) / h;
    resize_h = height_;
    resize_w = int(scale_ * w);
  }
  cv::resize(img, img, cv::Size(resize_w, resize_h));
  cv::Mat input(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
  int top = (height_ - resize_h) / 2;
  int bottom = height_ - top - resize_h;
  int left = (width_ - resize_w) / 2;
  int right = width_ - left - resize_w;
  cv::copyMakeBorder(img, input, top, bottom, left, right, cv::BORDER_CONSTANT,
                     cv::Scalar(0, 0, 0));
  return input;
}

int Yolov5::Draw(cv::Mat& rgb, const std::vector<Object>& objects) {
  static const char* class_names[] = {
      "person",        "bicycle",      "car",
      "motorcycle",    "airplane",     "bus",
      "train",         "truck",        "boat",
      "traffic light", "fire hydrant", "stop sign",
      "parking meter", "bench",        "bird",
      "cat",           "dog",          "horse",
      "sheep",         "cow",          "elephant",
      "bear",          "zebra",        "giraffe",
      "backpack",      "umbrella",     "handbag",
      "tie",           "suitcase",     "frisbee",
      "skis",          "snowboard",    "sports ball",
      "kite",          "baseball bat", "baseball glove",
      "skateboard",    "surfboard",    "tennis racket",
      "bottle",        "wine glass",   "cup",
      "fork",          "knife",        "spoon",
      "bowl",          "banana",       "apple",
      "sandwich",      "orange",       "broccoli",
      "carrot",        "hot dog",      "pizza",
      "donut",         "cake",         "chair",
      "couch",         "potted plant", "bed",
      "dining table",  "toilet",       "tv",
      "laptop",        "mouse",        "remote",
      "keyboard",      "cell phone",   "microwave",
      "oven",          "toaster",      "sink",
      "refrigerator",  "book",         "clock",
      "vase",          "scissors",     "teddy bear",
      "hair drier",    "toothbrush"};

  static const unsigned char colors[19][3] = {
      {54, 67, 244},  {99, 30, 233},   {176, 39, 156}, {183, 58, 103},
      {181, 81, 63},  {243, 150, 33},  {244, 169, 3},  {212, 188, 0},
      {136, 150, 0},  {80, 175, 76},   {74, 195, 139}, {57, 220, 205},
      {59, 235, 255}, {7, 193, 255},   {0, 152, 255},  {34, 87, 255},
      {72, 85, 121},  {158, 158, 158}, {139, 125, 96}};

  int color_index = 0;

  for (size_t i = 0; i < objects.size(); i++) {
    const Object& obj = objects[i];

    //         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n",
    //         obj.label, obj.prob,
    //                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

    const unsigned char* color = colors[color_index % 19];
    color_index++;

    cv::Scalar cc(color[0], color[1], color[2]);

    cv::rectangle(rgb, obj.rect, cc, 2);

    char text[256];
    sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = obj.rect.x;
    int y = obj.rect.y - label_size.height - baseLine;
    if (y < 0) y = 0;
    if (x + label_size.width > rgb.cols) x = rgb.cols - label_size.width;

    cv::rectangle(
        rgb,
        cv::Rect(cv::Point(x, y),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        cc, -1);

    cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381)
                            ? cv::Scalar(0, 0, 0)
                            : cv::Scalar(255, 255, 255);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
  }

  return 0;
}

}  // namespace yolov5