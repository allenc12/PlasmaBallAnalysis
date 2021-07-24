#include "VideoCap.hpp"

#include <chrono>
#include <fstream>
#include <jsoncpp/json/writer.h>
#include <limits>
#include <system_error>
#include <stdexcept>
#include <string>

#include <cstdlib>
#include <cstddef>
#include <cstring>

#include <jsoncpp/json/json.h>

namespace fs = std::filesystem;

static fs::path find_valid_subject_path(const char* str)
{
  const char* fmt = "%s_%010X";
  std::size_t length = std::snprintf(nullptr, 0, fmt, str, 0);
  std::string buf;
  buf.reserve(length + 1);
  fs::path tmp, cur(".");
  for (unsigned int i=0; i < std::numeric_limits<unsigned int>::max(); ++i) {
    std::snprintf(&buf[0], buf.size(), fmt, str, i);
    tmp = cur / buf;
    if (!fs::exists(fs::status(tmp))) {
        fs::create_directory(tmp);
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

VideoCap::VideoCap(cv::VideoCapture& cap, const char* pth, unsigned ntrials, unsigned nsamples): quiet(false), focus(false), num_trials(ntrials), num_samples(nsamples), size(2*num_trials*num_samples), cap(cap)
{
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta(size);
  std::vector<cv::Mat> frames(size);

  subject = find_valid_subject_path(pth);
}

VideoCap::VideoCap(cv::VideoCapture& cap, const char* pth, unsigned ntrials, unsigned nsamples, bool quiet): quiet(quiet), focus(false), num_trials(ntrials), num_samples(nsamples), size(2*num_trials*num_samples), cap(cap)
{
  std::vector<std::chrono::time_point<std::chrono::steady_clock>> time_delta(size);
  std::vector<cv::Mat> frames(size);

  subject = find_valid_subject_path(pth);
}

VideoCap::VideoCap(const VideoCap& that) { *this = that; }

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
    cv::Mat frame;

    cap >> frame;
    time_delta.push_back(std::chrono::steady_clock::now());
    frames.push_back(frame);
}

void VideoCap::runTrial(void)
{
  this->begin = std::chrono::system_clock::now();
  for (int i=0; i < this->num_trials; ++i) {
    if (!this->focus) {
      this->focus = true;
      if (!this->quiet)
        std::puts("Focus");
    } else {
      this->focus = false;
      if (!this->quiet)
        std::puts("Relax");
    }
    for (int j=0; j < this->num_samples; ++j)
      readSample();
  }
}

void VideoCap::saveOutput(void)
{
  // dubject,trials,samples,elapsed,time_delta,begin
  std::fstream outf;
  Json::Value subject = this->subject.c_str();
  Json::Value trials = this->num_trials;
  Json::Value samples = this->num_samples;
  Json::StreamWriterBuilder builder;

  outf.open(this->subject / "data.json");
  builder["commentStyle"] = "None";
  builder["indentation"] = "";
  std::unique_ptr<Json::StreamWriter> writer(
    builder.newStreamWriter());
  writer->write(subject, &outf);
  writer->write(trials, &outf);
  writer->write(samples, &outf);
  outf.flush();
  outf.close();
  // TODO: Video output
}
