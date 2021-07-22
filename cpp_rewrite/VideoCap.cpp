#include "VideoCap.hpp"

#include <limits>
#include <system_error>
#include <stdexcept>

#include <cstdlib>
#include <cstddef>
#include <cstring>

namespace fs = std::filesystem;

static fs::path find_valid_subject_path(const char* str)
{
  const char* fmt = "%s_%010X";
  std::size_t length = std::snprintf(nullptr, 0, fmt, str, 0);
  std::vector<char> buf(length + 1);
  fs::path tmp, cur(".");
  std::error_code existc, createc;
  for (unsigned int i=0; i < std::numeric_limits<unsigned int>::max(); ++i) {
    std::snprintf(&buf[0], buf.size, fmt, str, i);
    tmp = cur / buf;
    if (!fs::exists(fs::status(tmp), existc)) {
        if (!fs::create_directory(tmp, createc))
            throw std::exception(ec.message());
        break;
    }
  }
  return tmp;
}

VideoCap::VideoCap( void ) : quiet(false), focus(false), num_trials(5), num_samples(100), size(2*num_trials*num_samples)
{
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta(size);
  std::vector<cv::Mat> frames(size);

  cv::VideoCapture cap(0);

  subject = find_valid_subject_path(std::getenv("USER"));
}

VideoCap::VideoCap(bool quiet) : quiet(quiet), focus(false), num_trials(5), num_samples(100), size(2*num_trials*num_samples)
{
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta(size);
  std::vector<cv::Mat> frames(size);

  cv::VideoCapture cap(0);

  subject = find_valid_subject_path(std::getenv("USER"));
}

VideoCap::VideoCap(cv::VideoCapture& cap, std::string pth, const unsigned ntrials, const unsigned nsamples): quiet(false), focus(false), num_trials(ntrials), num_samples(nsamples), size(2*num_trials*num_samples), cap(cap)
{
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta(size);
  std::vector<cv::Mat> frames(size);

  subject = find_valid_subject_path(pth);
}

VideoCap::VideoCap(cv::VideoCapture& cap, std::string pth, const unsigned ntrials, const unsigned nsamples, bool quiet): quiet(quiet), focus(false), num_trials(ntrials), num_samples(nsamples), size(2*num_trials*num_samples), cap(cap)
{
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta(size);
  std::vector<cv::Mat> frames(size);

  subject = find_valid_subject_path(pth);
}

VideoCap::VideoCap(VideoCap::VideoCap& that) { *this = that; }

VideoCap::~VideoCap(void) {}

VideoCap& VideoCap::operator=(const VideoCap& rhs) {
    if (this != &rhs) {
        this->quiet = rhs.quiet;
        this->focus = rhs.focus;
        this->num_trials = rhs.num_trials;
        this->num_samples = rhs.num_samples;
        this->size = rhs.size;
        this->subject = rhs.subject;
        this->cap = rhs.cap;
        this->begin = rhs.begin;
        this->time_delta = rhs.time_delta;
        this->frames = rhs.frames;
    }
    return *this;
}

void VideoCap::readSample(void)
{
    Mat frame;

    cap >> frame;
    time_delta.append(std::chrono::steady_clock::now());
    frames.append(frame);
}

void VideoCap::runTrial(void) {}

void VideoCap::saveOutput(void) {}
