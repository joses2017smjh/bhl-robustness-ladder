// Visualization-only replacements for ORB-SLAM3's Viewer.cc/MapDrawer.cc.
// The pinned tracking, local mapping, loop closure and optimization sources
// are compiled unchanged. A viewer request is rejected rather than simulated.
// ORB-SLAM3 is GPL-3.0; upstream attribution/license accompany runtime packs.
#include "Viewer.h"
#include "MapDrawer.h"
#include <stdexcept>

namespace ORB_SLAM3 {
MapDrawer::MapDrawer(Atlas* atlas, const std::string&, Settings*) : mpAtlas(atlas) {}
void MapDrawer::newParameterLoader(Settings*) {}
void MapDrawer::DrawMapPoints() {}
void MapDrawer::DrawKeyFrames(bool, bool, bool, bool) {}
void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix&) {}
void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f& pose) {
    std::unique_lock<std::mutex> lock(mMutexCamera); mCameraPose = pose;
}
void MapDrawer::SetReferenceKeyFrame(KeyFrame*) {}
void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix&, pangolin::OpenGlMatrix&) {}
bool MapDrawer::ParseViewerParamFile(cv::FileStorage&) { return true; }
Viewer::Viewer(System*, FrameDrawer*, MapDrawer*, Tracking*, const std::string&, Settings*) {
    throw std::runtime_error("This native build is headless; bUseViewer must be false");
}
void Viewer::newParameterLoader(Settings*) {}
void Viewer::Run() { throw std::runtime_error("Headless build has no viewer"); }
void Viewer::RequestFinish() {}
void Viewer::RequestStop() {}
bool Viewer::isFinished() { return true; }
bool Viewer::isStopped() { return true; }
bool Viewer::isStepByStep() { return false; }
void Viewer::Release() {}
bool Viewer::ParseViewerParamFile(cv::FileStorage&) { return true; }
bool Viewer::Stop() { return true; }
bool Viewer::CheckFinish() { return true; }
void Viewer::SetFinish() {}
}
