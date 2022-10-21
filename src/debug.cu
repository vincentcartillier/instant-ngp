
#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>

#include <filesystem/path.h>

#include <stb_image/stb_image.h>

#define STB_IMAGE_IMPLEMENTATION

using namespace args;
using namespace ngp;
using namespace std;
using namespace tcnn;
using namespace Eigen;
namespace fs = ::filesystem;

int main(int argc, char** argv) {
	ArgumentParser parser{
		"neural graphics primitives debug\n"
		"version " NGP_VERSION,
		"",
	};

	HelpFlag help_flag{
		parser,
		"HELP",
		"Display this help menu.",
		{'h', "help"},
	};

	ValueFlag<string> network_config_flag{
		parser,
		"CONFIG",
		"Path to the network config. Uses the scene's default if unspecified.",
		{'n', 'c', "network", "config"},
	};

	Flag no_gui_flag{
		parser,
		"NO_GUI",
		"Disables the GUI and instead reports training progress on the command line.",
		{"no-gui"},
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disables training on startup.",
		{"no-train"},
	};

	ValueFlag<string> scene_flag{
		parser,
		"SCENE",
		"The scene to load. Can be NeRF dataset, a *.obj mesh for training a SDF, an image, or a *.nvdb volume.",
		{'s', "scene"},
	};

	ValueFlag<string> snapshot_flag{
		parser,
		"SNAPSHOT",
		"Optional snapshot to load upon startup.",
		{"snapshot"},
	};

	ValueFlag<uint32_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI.",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution height of the GUI.",
		{"height"},
	};

	Flag version_flag{
		parser,
		"VERSION",
		"Display the version of neural graphics primitives.",
		{'v', "version"},
	};

	// Parse command line arguments and react to parsing
	// errors using exceptions.

    tlog::none() << " neural graphics primitives version -- DEBUG" NGP_VERSION;

    // Create NGP
	ETestbedMode mode = ETestbedMode::Nerf;
	Testbed instant_ngp{mode};

    instant_ngp.m_nerf.sharpen = 0.f;
    instant_ngp.m_exposure = 0.f;

    std::string network = "configs/nerf/base.json";
    instant_ngp.reload_network_from_file(network);

    instant_ngp.m_is_slam_mode = true;
    instant_ngp.m_nerf.render_with_lens_distortion = true;
    instant_ngp.m_nerf.training.depth_supervision_lambda = 1.f;
    instant_ngp.m_fov_axis = 0;


    // Create dataset
    tlog::info()<<"Loading images";
    uint32_t aabb_scale=4;
    instant_ngp.create_empty_nerf_dataset(2, aabb_scale, false);

    float fx=517.3f;
    float fy=516.5f;
    float cx=318.6f;
    float cy=255.3f;
    float k1=0.f;
    float k2=0.f;
    float p1=0.f;
    float p2=0.f;
    float scale=0.333333f;
    float depth_scale=5000;

    int comp=0;
    Eigen::Vector2i res;
    std::string image_path = "/srv/essa-lab/flash3/vcartillier3/nerf-slam/Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png";
    uint8_t* img = stbi_load(image_path.c_str(), &res.x(), &res.y(), &comp, 4);

    int wa; int ha;
    std::string depth_path = "/srv/essa-lab/flash3/vcartillier3/nerf-slam/Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png";
    uint16_t* depth_pixels = stbi_load_16(depth_path.c_str(), &wa, &ha, &comp, 1);

    uint32_t frame_idx = 0;

    instant_ngp.m_nerf.training.dataset.set_training_image(
        frame_idx,
        res,
        img,
        depth_pixels,
        depth_scale * scale,
        false,
        EImageDataType::Byte,
        EDepthDataType::UShort
    );

    instant_ngp.m_nerf.training.set_camera_intrinsics(
        frame_idx, fx, fy, cx, cy, k1, k2, p1, p2
    );

    Eigen::Matrix<float, 3, 4> c2w {
        { 0.05365186, -0.18462093, 0.9813443, 2.5967727 },
        { 0.99646974, -0.05365186, -0.06457236, 2.4037294 },
        { 0.06457236, 0.9813443, 0.18109065, 1.6236416 }
    };

    instant_ngp.m_nerf.training.set_camera_extrinsics(
        frame_idx,
        c2w,
        true
    );

    CUDA_CHECK_THROW(cudaDeviceSynchronize());


    comp=0;
    image_path = "/srv/essa-lab/flash3/vcartillier3/nerf-slam/Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png";
    img = stbi_load(image_path.c_str(), &res.x(), &res.y(), &comp, 4);

    depth_path = "/srv/essa-lab/flash3/vcartillier3/nerf-slam/Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/depth/1305031453.404816.png";
    depth_pixels = stbi_load_16(depth_path.c_str(), &wa, &ha, &comp, 1);

    frame_idx = 1;

    instant_ngp.m_nerf.training.dataset.set_training_image(
        frame_idx,
        res,
        img,
        depth_pixels,
        depth_scale * scale,
        false,
        EImageDataType::Byte,
        EDepthDataType::UShort
    );

    instant_ngp.m_nerf.training.set_camera_intrinsics(
        frame_idx, fx, fy, cx, cy, k1, k2, p1, p2
    );

    Eigen::Matrix<float, 3, 4> c2w_2 {
        { 0.05365186, -0.18462093, 0.9813443, 2.5967727 },
        { 0.99646974, -0.05365186, -0.06457236, 2.4037294 },
        { 0.06457236, 0.9813443, 0.18109065, 1.6236416 }
    };

    // actually pose for frame#2
    // // [[ 0.06412726 -0.183196    0.98098266  2.6210103 ]
    // //  [ 0.9967727  -0.03580809 -0.07184654  2.411902  ]
    // //  [ 0.04828911  0.982424    0.18030849  1.6151757 ]
    // //  [ 0.          0.          0.          1.        ]]

    instant_ngp.m_nerf.training.set_camera_extrinsics(
        frame_idx,
        c2w_2,
        true
    );


    // train mapping on first image
    tlog::info()<<" Start Mapping";
    instant_ngp.m_nerf.training.n_images_for_training = 2;
    instant_ngp.m_nerf.training.n_images_for_training_slam = 1;

    std::vector<uint32_t> idx_images_for_training_slam{0};
    instant_ngp.m_nerf.training.idx_images_for_training_slam = idx_images_for_training_slam;

    instant_ngp.m_train = true;
    uint32_t batch_size=256000;
    for (uint32_t i=0; i<100; ++i) {
        instant_ngp.train(batch_size);
    }

    //tlog::info()<<" Render first frame";

    // train tracking on second image
    tlog::info()<<" Start Tracking";
    instant_ngp.m_nerf.training.indice_image_for_tracking_pose = 1;
    instant_ngp.m_nerf.training.extrinsic_learning_rate_pos = 0.01 * 0.01;
    instant_ngp.m_nerf.training.extrinsic_learning_rate_rot = 0.002 * 0.01;
    instant_ngp.m_nerf.training.separate_pos_and_rot_lr = true;
    // instant_ngp.m_nerf.training.track_loss_type = ngp.LossType.L2;
    // instant_ngp.m_nerf.training.track_depth_loss_type = ngp.LossType.L1
    instant_ngp.m_tracking_kernel_window_size = 1;
    instant_ngp.m_sample_away_from_border_margin_w = 0;
    instant_ngp.m_sample_away_from_border_margin_h = 0;

    batch_size=256000;
    for (uint32_t i=0; i<10; ++i) {
        instant_ngp.track_pose(batch_size);
    }



}
