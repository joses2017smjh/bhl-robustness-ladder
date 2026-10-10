// FAST-LIVO2 headless recorded-sensor transport. Upstream IMU, voxel-map and
// direct-VIO translation units are compiled unchanged. No reference pose input.
#include "IMU_Processing.h"
#include "vio.h"
#include <cstdint>
#include <iomanip>
#include <stdexcept>
#include <fstream>

namespace {
template<class T> T read_value(std::istream& input) {
  T value; if (!input.read(reinterpret_cast<char*>(&value),sizeof(value))) throw std::runtime_error("truncated native transport"); return value;
}
void transform_cloud(const M3D& rotation,const V3D& position,const M3D& extrinsic_r,const V3D& extrinsic_t,
                     const PointCloudXYZI::Ptr& source,const PointCloudXYZI::Ptr& world) {
  world->resize(source->size());
  for (size_t i=0;i<source->size();++i) {
    const auto& p=source->points[i]; V3D v=rotation*(extrinsic_r*V3D(p.x,p.y,p.z)+extrinsic_t)+position;
    world->points[i]=p;world->points[i].x=v.x();world->points[i].y=v.y();world->points[i].z=v.z();
  }
}
}
int main(int argc,char**argv) {
  if(argc!=3) throw std::runtime_error("usage: fastlivo_headless input.bin frames.jsonl");
  std::ifstream input(argv[1],std::ios::binary);std::ofstream output(argv[2]);
  if(!input||!output) throw std::runtime_error("cannot open native input/output");
  char magic[8];input.read(magic,8);if(std::string(magic,8)!="BHLLIVO1")throw std::runtime_error("invalid native transport magic");
  M3D extR,Rcl;V3D extT,Pcl;
  for(int r=0;r<3;++r)for(int c=0;c<3;++c)extR(r,c)=read_value<double>(input);
  for(int i=0;i<3;++i)extT(i)=read_value<double>(input);
  for(int r=0;r<3;++r)for(int c=0;c<3;++c)Rcl(r,c)=read_value<double>(input);
  for(int i=0;i<3;++i)Pcl(i)=read_value<double>(input);
  const auto width=read_value<uint32_t>(input),height=read_value<uint32_t>(input);
  const double fx=read_value<double>(input),fy=read_value<double>(input),cx=read_value<double>(input),cy=read_value<double>(input);
  const double surface_voxel=read_value<double>(input),blind=read_value<double>(input);
  const auto count=read_value<uint32_t>(input), enable_visual=read_value<uint32_t>(input);
  if(width<128||height<128||width>4096||height>4096||!(fx>0&&fy>0&&surface_voxel>0&&blind>=0)||count>1000000||enable_visual>1)
    throw std::runtime_error("invalid native calibration/configuration");
  cv::setNumThreads(1);omp_set_num_threads(1);
  StatesGroup state,propagated;
  static ImuProcess imu;imu.lidar_type=VELO16;imu.set_extrinsic(extT,extR);
  imu.set_gyr_cov_scale(V3D(.3,.3,.3));imu.set_acc_cov_scale(V3D(.5,.5,.5));
  imu.set_gyr_bias_cov(V3D(.0001,.0001,.0001));imu.set_acc_bias_cov(V3D(.0001,.0001,.0001));
  imu.set_inv_expo_cov(.1);imu.set_imu_init_frame_num(30);
  VoxelMapConfig cfg;cfg.max_voxel_size_=.5;cfg.max_layer_=2;cfg.max_iterations_=5;
  cfg.layer_init_num_={5,5,5,5,5};cfg.max_points_num_=50;cfg.planner_threshold_=.0025;
  cfg.beam_err_=.05;cfg.dept_err_=.02;cfg.sigma_num_=3;cfg.is_pub_plane_map_=false;
  cfg.sliding_thresh=8;cfg.map_sliding_en=false;cfg.half_map_size=100;
  std::unordered_map<VOXEL_LOCATION,VoxelOctoTree*> map;
  VoxelMapManager lidar(cfg,map);lidar.extR_=extR;lidar.extT_=extT;
  vk::PinholeCamera camera(width,height,1.0,fx,fy,cx,cy);
  VIOManager vio;vio.cam=&camera;vio.state=&state;vio.state_propagat=&propagated;
  vio.grid_size=5;vio.grid_n_width=0;vio.grid_n_height=17;vio.patch_size=8;vio.patch_pyrimid_level=4;
  vio.outlier_threshold=1000;vio.max_iterations=5;vio.img_point_cov=100;
  vio.normal_en=true;vio.inverse_composition_en=false;vio.raycast_en=false;
  vio.exposure_estimate_en=true;vio.colmap_output_en=false;vio.plot_flag=false;vio.total_points=0;
  vio.setImuToLidarExtrinsic(extT,extR);
  std::vector<double> rcl(9),pcl(3);for(int r=0;r<3;++r)for(int c=0;c<3;++c)rcl[r*3+c]=Rcl(r,c);
  for(int i=0;i<3;++i)pcl[i]=Pcl(i);vio.setLidarToCameraExtrinsic(rcl,pcl);vio.initializeVIO();
  pcl::VoxelGrid<PointType> down;down.setLeafSize(surface_voxel,surface_voxel,surface_voxel);
  PointCloudXYZI::Ptr undistorted(new PointCloudXYZI),filtered(new PointCloudXYZI),world(new PointCloudXYZI);
  LidarMeasureGroup measures;bool map_initialized=false;double first_time=0,last_timestamp=-1;
  output<<std::setprecision(17);
  for(uint32_t index=0;index<count;++index) {
    const double packet_begin=read_value<double>(input),timestamp=read_value<double>(input);
    const auto point_count=read_value<uint32_t>(input),imu_count=read_value<uint32_t>(input),image_size=read_value<uint32_t>(input);
    if(timestamp<=last_timestamp||!std::isfinite(timestamp)||packet_begin>timestamp||point_count>100000||imu_count>10000||image_size>32*1024*1024)
      throw std::runtime_error("noncausal or oversized native packet");
    last_timestamp=timestamp;
    measures.lidar.reset(new PointCloudXYZI);measures.pcl_proc_cur.reset(new PointCloudXYZI);
    measures.lidar_frame_beg_time=packet_begin;measures.lidar_frame_end_time=timestamp;
    // Upstream LIVO synchronization expresses every point offset relative to
    // the preceding camera/LIO update time, preserving the original point time.
    measures.last_lio_update_time=packet_begin;
    for(uint32_t i=0;i<point_count;++i) {
      PointType p{};p.x=read_value<float>(input);p.y=read_value<float>(input);p.z=read_value<float>(input);
      p.intensity=read_value<float>(input);const float offset=read_value<float>(input);read_value<uint16_t>(input);
      if(offset<0||packet_begin+offset>timestamp+1e-6)throw std::runtime_error("future lidar return");
      p.curvature=offset*1000.f;
      if(p.x*p.x+p.y*p.y+p.z*p.z>blind*blind)measures.pcl_proc_cur->push_back(p);
    }
    MeasureGroup measurement;measurement.lio_time=timestamp;
    for(uint32_t i=0;i<imu_count;++i) {
      sensor_msgs::Imu::Ptr sample(new sensor_msgs::Imu);
      sample->header.stamp.fromSec(read_value<double>(input));
      sample->angular_velocity.x=read_value<double>(input);sample->angular_velocity.y=read_value<double>(input);sample->angular_velocity.z=read_value<double>(input);
      sample->linear_acceleration.x=read_value<double>(input);sample->linear_acceleration.y=read_value<double>(input);sample->linear_acceleration.z=read_value<double>(input);
      if(sample->header.stamp.toSec()>timestamp+1e-9)throw std::runtime_error("future IMU sample");
      measurement.imu.push_back(sample);
    }
    std::vector<unsigned char> png(image_size);if(!input.read(reinterpret_cast<char*>(png.data()),image_size))throw std::runtime_error("truncated camera image");
    const double decode_started=omp_get_wtime();cv::Mat image=cv::imdecode(png,cv::IMREAD_COLOR);
    const double decode_seconds=omp_get_wtime()-decode_started;
    if(image.empty()||image.cols!=int(width)||image.rows!=int(height))throw std::runtime_error("camera image calibration mismatch");
    measures.measures.clear();measures.measures.push_back(measurement);measures.lio_vio_flg=LIO;
    if(index==0){first_time=packet_begin;imu.first_lidar_time=packet_begin;}
    const double started=omp_get_wtime();std::string status="INITIALIZING";bool tracked=false;
    int effective=0,visual_points=0;bool visual_attempted=false;
    undistorted->clear();
    if(measurement.imu.empty())status=measures.pcl_proc_cur->empty()?"NO_CAUSAL_LIDAR":"NO_CAUSAL_IMU";
    else {
      imu.Process2(measures,state,undistorted);
      propagated=state;lidar.state_=state;lidar.feats_undistort_=undistorted;
      if(undistorted->empty())status=imu.imu_need_init?"IMU_INITIALIZING":"NO_CAUSAL_LIDAR";
      else {
        down.setInputCloud(undistorted);down.filter(*filtered);
        lidar.feats_down_body_=filtered;lidar.feats_down_size_=filtered->size();
        transform_cloud(state.rot_end,state.pos_end,extR,extT,filtered,world);lidar.feats_down_world_=world;
        if(filtered->size()<5)status="INSUFFICIENT_SCAN_POINTS";
        else {
          if(!map_initialized){lidar.BuildVoxelMap();map_initialized=true;}
          lidar.StateEstimation(propagated);state=lidar.state_;effective=lidar.effct_feat_num_;
          transform_cloud(state.rot_end,state.pos_end,extR,extT,filtered,world);
          // Exact upstream LIVMapper covariance propagation before voxel update.
          for(size_t i=0;i<world->size();++i) {
            lidar.pv_list_[i].point_w<<world->points[i].x,world->points[i].y,world->points[i].z;
            M3D cross=lidar.cross_mat_list_[i],var=lidar.body_cov_list_[i];
            var=(state.rot_end*extR)*var*(state.rot_end*extR).transpose()+(-cross)*state.cov.block<3,3>(0,0)*(-cross).transpose()+state.cov.block<3,3>(3,3);
            lidar.pv_list_[i].var=var;
          }
          lidar.UpdateVoxelMap(lidar.pv_list_);
          tracked=(effective>0||visual_points>0)&&state.pos_end.allFinite()&&state.rot_end.allFinite();
          status=tracked?"TRACKED":"NO_EFFECTIVE_MEASUREMENTS";
        }
      }
    }
          if(enable_visual && map_initialized && !measurement.imu.empty()) {
            // Upstream synchronizer switches LIO -> VIO at the same timestamp;
            // its second IMU call has no new samples and zero propagation time.
            MeasureGroup visual;visual.lio_time=timestamp;visual.vio_time=timestamp;visual.img=image;
            measures.measures.clear();measures.measures.push_back(visual);measures.lio_vio_flg=VIO;
            imu.Process2(measures,state,undistorted);propagated=state;
            vio.processFrame(image,lidar.pv_list_,lidar.voxel_map_,timestamp-first_time);
            visual_attempted=true;visual_points=vio.total_points;
          }
    tracked=(effective>0||visual_points>0)&&state.pos_end.allFinite()&&state.rot_end.allFinite();
    if(tracked)status="TRACKED";
    output<<"{\"frame_index\":"<<index<<",\"timestamp_s\":"<<timestamp<<",\"scan_begin_s\":"<<packet_begin
          <<",\"tracked\":"<<(tracked?"true":"false")<<",\"state\":\""<<status<<"\",\"compute_seconds\":"<<(omp_get_wtime()-started)
          <<",\"image_decode_seconds\":"<<decode_seconds<<",\"map_reset_id\":0,\"effective_points\":"<<effective
          <<",\"visual_update_attempted\":"<<(visual_attempted?"true":"false")<<",\"visual_points\":"<<visual_points<<",\"T_W_I\":";
    if(!tracked)output<<"null";
    else {
      output<<'[';for(int r=0;r<4;++r){if(r)output<<',';output<<'[';for(int c=0;c<4;++c){if(c)output<<',';
      output<<(r==3?(c==3?1.:0.):(c==3?state.pos_end(r):state.rot_end(r,c)));}output<<']';}output<<']';
    }
    output<<"}\n";output.flush();
  }
  if(input.peek()!=std::char_traits<char>::eof())throw std::runtime_error("unconsumed native packet bytes");
  return 0;
}
