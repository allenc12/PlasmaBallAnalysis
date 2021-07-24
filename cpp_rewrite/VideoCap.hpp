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
  VideoCap(cv::VideoCapture &, const char*, unsigned, unsigned);
  VideoCap(cv::VideoCapture &, const char*, unsigned, unsigned, bool);
  VideoCap(const VideoCap &);
  ~VideoCap(void);
  VideoCap &operator=(const VideoCap &);

  void readSample(void);
  void runTrial(void);
  void saveOutput(void);

private:
  unsigned num_trials;
  unsigned num_samples;
  std::size_t size;

  bool focus;
  bool quiet;

  std::filesystem::path subject; // output directory
  cv::VideoCapture cap;
  std::chrono::time_point<std::chrono::system_clock> begin;

  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta;
  std::vector<cv::Mat> frames;
};
