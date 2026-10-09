// Native ORB-SLAM3 stereo entry point. Tracking/mapping are upstream code.
// All inputs are original RGB images/calibration/timestamps; no truth interface.
#include "System.h"
#include "Tracking.h"
#include "MapPoint.h"
#include "Map.h"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "bhl-orb-native-v1 ORB_SLAM3 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4 headless stereo\n";
    return 0;
  }
  if (argc != 4) {
    std::cerr << "Usage: orb_native VOCABULARY SETTINGS RESPONSE_JSONL < FRAME_REQUESTS_TSV\n";
    return 2;
  }
  try {
    // A regular output file or FIFO; diagnostics stay on inherited stdout.
    std::ofstream replies(argv[3]);
    if (!replies) throw std::runtime_error("Cannot open response stream");
    replies << std::setprecision(17);
    cv::setNumThreads(1);
    ORB_SLAM3::System slam(argv[1], argv[2], ORB_SLAM3::System::STEREO, false);
    replies << "{\"schema\":\"bhl-orb-native-ready-v1\",\"upstream_commit\":\"4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4\"}\n" << std::flush;
    std::string request;
    double previous = -std::numeric_limits<double>::infinity();
    while (std::getline(std::cin, request)) {
      std::istringstream fields(request);
      std::string stamp_text, left_path, right_path, extra;
      if (!std::getline(fields, stamp_text, '\t') || !std::getline(fields, left_path, '\t') ||
          !std::getline(fields, right_path, '\t') || std::getline(fields, extra, '\t'))
        throw std::runtime_error("Request must contain exactly timestamp, left PNG, right PNG");
      std::size_t consumed = 0;
      const double stamp = std::stod(stamp_text, &consumed);
      if (consumed != stamp_text.size() || !std::isfinite(stamp) || stamp <= previous)
        throw std::runtime_error("Timestamps must be finite and strictly increasing");
      auto start = std::chrono::steady_clock::now();
      cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
      cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
      if (left.empty() || right.empty() || left.size() != right.size())
        throw std::runtime_error("Missing or mismatched original stereo images");
      // imread gives BGR; exported settings explicitly declare RGB input.
      cv::cvtColor(left, left, cv::COLOR_BGR2RGB);
      cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
      const Sophus::SE3f T_C_W = slam.TrackStereo(left, right, stamp);
      const int state = slam.GetTrackingState();
      const bool tracked = state == ORB_SLAM3::Tracking::OK;
      const Eigen::Matrix4f pose = T_C_W.inverse().matrix();
      if (tracked && !pose.allFinite()) throw std::runtime_error("Native tracked pose is not finite");
      long map_id = -1;
      // Read-only native instrumentation; no estimator modification is needed.
      for (auto* point : slam.GetTrackedMapPoints()) {
        if (point && !point->isBad() && point->GetMap()) {
          map_id = static_cast<long>(point->GetMap()->GetId()); break;
        }
      }
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      replies << "{\"schema\":\"bhl-orb-native-frame-v1\",\"timestamp_s\":" << stamp
              << ",\"tracking_state\":" << state << ",\"tracked\":" << (tracked ? "true" : "false")
              << ",\"compute_seconds\":" << seconds << ",\"map_id\":";
      if (map_id < 0) replies << "null"; else replies << map_id;
      replies << ",\"T_W_C\":";
      if (!tracked) replies << "null";
      else {
        replies << '[';
        for (int row = 0; row < 4; ++row) {
          if (row) replies << ','; replies << '[';
          for (int col = 0; col < 4; ++col) { if (col) replies << ','; replies << pose(row, col); }
          replies << ']';
        }
        replies << ']';
      }
      replies << "}\n" << std::flush;
      if (!replies) throw std::runtime_error("Response stream failed");
      previous = stamp;
    }
    slam.Shutdown();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Native ORB replay failed: " << error.what() << '\n';
    return 1;
  }
}
