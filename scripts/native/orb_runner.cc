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
#include <vector>
#include <array>

#ifdef BHL_ORB_ATLAS_TELEMETRY
#define BHL_ORB_VERSION_SUFFIX " atlas-telemetry-v1"
#else
#define BHL_ORB_VERSION_SUFFIX ""
#endif

int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "bhl-orb-native-v1 ORB_SLAM3 4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4 headless stereo stereo-inertial-v1"
              << BHL_ORB_VERSION_SUFFIX << "\n";
    return 0;
  }
  const bool inertial = argc == 5 && std::string(argv[4]) == "--stereo-inertial";
  if (argc != 4 && !inertial) {
    std::cerr << "Usage: orb_native VOCABULARY SETTINGS RESPONSE_JSONL [--stereo-inertial] < FRAME_REQUESTS_TSV\n";
    return 2;
  }
  try {
    // A regular output file or FIFO; diagnostics stay on inherited stdout.
    std::ofstream replies(argv[3]);
    if (!replies) throw std::runtime_error("Cannot open response stream");
    replies << std::setprecision(17);
    cv::setNumThreads(1);
    ORB_SLAM3::System slam(argv[1], argv[2], inertial ? ORB_SLAM3::System::IMU_STEREO : ORB_SLAM3::System::STEREO, false);
    replies << "{\"schema\":\"bhl-orb-native-ready-v1\",\"upstream_commit\":\"4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4\",\"sensor_mode\":\"" << (inertial ? "stereo_inertial" : "stereo") << "\"";
#ifdef BHL_ORB_ATLAS_TELEMETRY
    replies << ",\"atlas_telemetry\":\"bhl-orb-atlas-telemetry-v1\"";
#endif
    replies << "}\n" << std::flush;
    std::string request;
    double previous = -std::numeric_limits<double>::infinity();
    double previous_imu = -std::numeric_limits<double>::infinity();
#ifdef BHL_ORB_ATLAS_TELEMETRY
    // Wrapper-side memory of the previous Atlas snapshot. Only used to flag an
    // active-map reset that reuses the same native map id (Map::clear keeps
    // mnId; the next first keyframe replaces the map's init keyframe id).
    long previous_atlas_map = -1;
    unsigned long previous_init_kf = 0;
    long previous_keyframes = -1;
    long active_map_resets = 0;
#endif
    while (std::getline(std::cin, request)) {
      std::istringstream fields(request);
      std::string stamp_text, left_path, right_path, imu_path, extra;
      if (!std::getline(fields, stamp_text, '\t') || !std::getline(fields, left_path, '\t') ||
          !std::getline(fields, right_path, '\t') ||
          (inertial && !std::getline(fields, imu_path, '\t')) || std::getline(fields, extra, '\t'))
        throw std::runtime_error("Request requires timestamp, left PNG, right PNG and, only in inertial mode, IMU TSV");
      std::size_t consumed = 0;
      const double stamp = std::stod(stamp_text, &consumed);
      if (consumed != stamp_text.size() || !std::isfinite(stamp) || stamp <= previous)
        throw std::runtime_error("Timestamps must be finite and strictly increasing");
      auto start = std::chrono::steady_clock::now();
      std::vector<ORB_SLAM3::IMU::Point> imu;
      if (inertial) {
        std::ifstream input(imu_path);
        if (!input) throw std::runtime_error("Cannot open original IMU batch");
        std::string line;
        while (std::getline(input, line)) {
          std::istringstream values(line);
          std::array<double, 7> sample;
          for (double& value : sample) {
            if (!(values >> value) || !std::isfinite(value))
              throw std::runtime_error("IMU rows require finite time, gyro xyz and specific force xyz in SI units");
          }
          if (values >> extra || sample[0] <= previous_imu || sample[0] > stamp + 1e-12 ||
              (std::isfinite(previous_imu) && sample[0] - previous_imu > .020000000001) || imu.size() >= 10000)
            throw std::runtime_error("IMU must be bounded, causal, increasing and contain no gaps above 20 ms");
          // Upstream Point takes acceleration before angular velocity.
          imu.emplace_back(sample[4], sample[5], sample[6], sample[1], sample[2], sample[3], sample[0]);
          previous_imu = sample[0];
        }
        if (imu.empty() || stamp - imu.back().t > .020000000001 ||
            (std::isfinite(previous) && imu.front().t - previous > .020000000001))
          throw std::runtime_error("Original IMU samples must cover every camera interval causally");
      }
      cv::Mat left = cv::imread(left_path, cv::IMREAD_COLOR);
      cv::Mat right = cv::imread(right_path, cv::IMREAD_COLOR);
      if (left.empty() || right.empty() || left.size() != right.size())
        throw std::runtime_error("Missing or mismatched original stereo images");
      // imread gives BGR; exported settings explicitly declare RGB input.
      cv::cvtColor(left, left, cv::COLOR_BGR2RGB);
      cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
      const Sophus::SE3f T_C_W = slam.TrackStereo(left, right, stamp, imu);
      const int state = slam.GetTrackingState();
      const bool tracked = state == ORB_SLAM3::Tracking::OK;
      const Eigen::Matrix4f pose = T_C_W.inverse().matrix();
      if (tracked && !pose.allFinite()) throw std::runtime_error("Native tracked pose is not finite");
      long map_id = -1;
      bool imu_initialized = false;
      // Read-only native instrumentation; no estimator modification is needed.
      for (auto* point : slam.GetTrackedMapPoints()) {
        if (point && !point->isBad() && point->GetMap()) {
          map_id = static_cast<long>(point->GetMap()->GetId());
          imu_initialized = inertial && point->GetMap()->isImuInitialized(); break;
        }
      }
#ifdef BHL_ORB_DIAGNOSTICS
      // Wrapper-only, read-only access after synchronous TrackStereo returns.
      // The linked original libORB_SLAM3 and initializer threshold are unchanged.
      const auto* tracker = slam.BhlReadOnlyTracker();
      const auto& frame = tracker->mCurrentFrame;
      const std::size_t keypoints_left = frame.mvKeys.size();
      const std::size_t keypoints_right = frame.mvKeysRight.size();
      std::size_t stereo_depth_matches = 0;
      for (float depth : frame.mvDepth) if (std::isfinite(depth) && depth > 0) ++stereo_depth_matches;
      const char* initialization = state != ORB_SLAM3::Tracking::NOT_INITIALIZED ? "NOT_IN_INITIALIZATION" :
          frame.N <= 500 ? "FEATURE_COUNT_NOT_ABOVE_500" : "OTHER_NATIVE_INITIALIZATION_CONDITION";
#endif
#ifdef BHL_ORB_ATLAS_TELEMETRY
      // Wrapper-only Atlas snapshot after synchronous TrackStereo returns. Only
      // upstream getters are called (each takes the upstream mutex); upstream
      // Tracking::Track itself calls Atlas::GetCurrentMap on every frame. No
      // estimator state, threshold or library code is modified.
      // System::MapChanged keeps a function-static counter compared against the
      // CURRENT map's big-change index, so it is recorded but not relied upon.
      const bool system_map_changed = slam.MapChanged();
      ORB_SLAM3::Atlas* atlas = slam.BhlTelemetryAtlas();
      ORB_SLAM3::Map* active = atlas->GetCurrentMap();
      const int atlas_map_count = atlas->CountMaps();
      const int atlas_big_change = atlas->GetLastBigChangeIdx();
      const long atlas_map_id = static_cast<long>(active->GetId());
      const unsigned long init_kf_id = active->GetInitKFid();
      const unsigned long max_kf_id = active->GetMaxKFid();
      const long keyframes = static_cast<long>(active->KeyFramesInMap());
      const int map_big_change = active->GetLastBigChangeIdx();
      const bool map_imu_initialized = active->isImuInitialized();
      const bool map_inertial_ba1 = active->GetIniertialBA1();
      const bool map_inertial_ba2 = active->GetIniertialBA2();
      // Concurrent LoopClosing could swap the current map during the snapshot.
      const bool snapshot_consistent = atlas->GetCurrentMap() == active && map_big_change == atlas_big_change;
      // Unambiguous same-id reset: Tracking::ResetActiveMap -> Atlas::clearMap ->
      // Map::clear keeps the id but empties the keyframes; Map::AddKeyFrame then
      // replaces the init keyframe id. KeyFrameCulling/SetBadFlag never remove
      // the init keyframe, and KeyFrame::nNextId is not reset by ResetActiveMap.
      const bool active_map_reset = atlas_map_id == previous_atlas_map && previous_keyframes > 0 &&
          (keyframes == 0 || init_kf_id != previous_init_kf);
      if (active_map_reset) ++active_map_resets;
      previous_atlas_map = atlas_map_id;
      previous_init_kf = init_kf_id;
      previous_keyframes = keyframes;
#endif
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      replies << "{\"schema\":\"bhl-orb-native-frame-v1\",\"timestamp_s\":" << stamp
              << ",\"tracking_state\":" << state << ",\"tracked\":" << (tracked ? "true" : "false")
              << ",\"compute_seconds\":" << seconds << ",\"map_id\":";
      if (map_id < 0) replies << "null"; else replies << map_id;
#ifdef BHL_ORB_DIAGNOSTICS
      replies << ",\"diagnostics\":{\"schema\":\"bhl-orb-frame-diagnostics-v1\",\"features_total\":" << frame.N
              << ",\"features_left\":" << keypoints_left << ",\"features_right\":" << keypoints_right
              << ",\"positive_stereo_depth_matches\":" << stereo_depth_matches
              << ",\"initialization_feature_threshold_exclusive\":500,\"initialization_reason\":\"" << initialization
              << "\",\"read_only_native_frame\":true}";
#endif
#ifdef BHL_ORB_ATLAS_TELEMETRY
      replies << ",\"atlas\":{\"schema\":\"bhl-orb-atlas-telemetry-v1\",\"system_map_changed\":"
              << (system_map_changed ? "true" : "false") << ",\"atlas_map_count\":" << atlas_map_count
              << ",\"atlas_last_big_change_idx\":" << atlas_big_change
              << ",\"current_map_id\":" << atlas_map_id << ",\"current_map_init_kf_id\":" << init_kf_id
              << ",\"current_map_max_kf_id\":" << max_kf_id << ",\"current_map_keyframes\":" << keyframes
              << ",\"current_map_big_change_idx\":" << map_big_change
              << ",\"current_map_imu_initialized\":" << (map_imu_initialized ? "true" : "false")
              << ",\"current_map_inertial_ba1\":" << (map_inertial_ba1 ? "true" : "false")
              << ",\"current_map_inertial_ba2\":" << (map_inertial_ba2 ? "true" : "false")
              << ",\"snapshot_consistent\":" << (snapshot_consistent ? "true" : "false")
              << ",\"active_map_reset\":" << (active_map_reset ? "true" : "false")
              << ",\"active_map_resets_total\":" << active_map_resets
              << ",\"read_only_getters_after_synchronous_track\":true}";
#endif
      if (inertial) {
        replies << ",\"inertial\":{\"schema\":\"bhl-orb-inertial-frame-v1\",\"imu_samples\":" << imu.size()
                << ",\"first_imu_timestamp_s\":" << imu.front().t << ",\"last_imu_timestamp_s\":" << imu.back().t
                << ",\"map_imu_initialized\":" << (imu_initialized ? "true" : "false") << '}';
      }
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
