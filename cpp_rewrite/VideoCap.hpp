#pragma once
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <filesystem>
#include <vector>

class VideoCap {
public:
  VideoCap(void);
  VideoCap(bool);
  VideoCap(cv::VideoCapture &, std::string, const unsigned, const unsigned);
  VideoCap(cv::VideoCapture &, std::string, const unsigned, const unsigned, bool);
  VideoCap(const VideoCap &);
  ~VideoCap(void);
  VideoCap &operator=(const VideoCap &);

  void readSample(void);
  void runTrial(void);
  void saveOutput(void);

private:
  const unsigned num_trials;
  const unsigned num_samples;
  const std::size_t size;

  bool focus;
  bool quiet;

  std::filesystem::path subject; // output directory
  cv::VideoCapture cap;
  std::chrono::time_point<std::chrono::system_clock> begin;

  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta;
  std::vector<cv::Mat> frames;
};
