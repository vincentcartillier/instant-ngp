
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

    std::string network = "configs/nerf/base_debug.json";
    instant_ngp.reload_network_from_file(network);

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
    float depth_scale=1/5000.f;

    int comp=0;
    ivec2 res;
    std::string image_path = "/srv/essa-lab/flash3/vcartillier3/nerf-slam/Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031453.359684.png";
    uint8_t* img = stbi_load(image_path.c_str(), &res.x, &res.y, &comp, 4);

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

    mat4x3 c2w {
        // { 0.05365186, -0.18462093, 0.9813443,  2.5967727 },
        // { 0.99646974, -0.05365186, -0.06457236, 2.4037294 },
        // { 0.06457236, 0.9813443,   0.18109065, 1.6236416 }
        0.05365186, 0.99646974,0.06457236,
        -0.18462093, -0.05365186, 0.9813443, 
        0.9813443, -0.06457236, 0.18109065,
        2.5967727, 2.4037294, 1.6236416 
    };

    instant_ngp.m_nerf.training.set_camera_extrinsics(
        frame_idx,
        c2w,
        true
    );

    CUDA_CHECK_THROW(cudaDeviceSynchronize());


    comp=0;
    image_path = "/srv/essa-lab/flash3/vcartillier3/nerf-slam/Datasets/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgb/1305031453.391690.png";
    img = stbi_load(image_path.c_str(), &res.x, &res.y, &comp, 4);

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

    mat4x3 c2w_2 {
        // { 0.05365186, -0.18462093, 0.9813443,  2.5967727 },
        // { 0.99646974, -0.05365186, -0.06457236, 2.4037294 },
        // { 0.06457236, 0.9813443,   0.18109065, 1.6236416 }
        0.05365186, 0.99646974,0.06457236,
        -0.18462093, -0.05365186, 0.9813443, 
        0.9813443, -0.06457236, 0.18109065,
        2.5967727, 2.4037294, 1.6236416 

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


    //DEBUG
    // USE ScanNet dataset - scene0106
    instant_ngp.clear_training_data();
    instant_ngp.load_training_data("/srv/essa-lab/flash3/vcartillier3/nerf-slam/runs/ScanNet_scene0106/2/preprocessed_dataset/transforms.json");
    float poses_scale = 1.67;
    scale = 0.059;


    // train mapping on first image
    tlog::info()<<" Start Mapping";
    instant_ngp.m_nerf.training.n_images_for_training = 2;

    std::vector<uint32_t> idx_images_for_training_slam{0};
    instant_ngp.m_nerf.training.idx_images_for_mapping = idx_images_for_training_slam;
    
	instant_ngp.m_use_depth_guided_sampling=false;
    instant_ngp.m_add_free_space_loss = false;
    instant_ngp.m_nerf.training.depth_supervision_lambda= 1.0;
    instant_ngp.m_nerf.training.free_space_supervision_lambda= 1.0;
    instant_ngp.m_nerf.training.free_space_supervision_distance= 0.1 * poses_scale * scale;

    //settings
    instant_ngp.m_add_DSnerf_loss = true;
    instant_ngp.m_use_DSnerf_loss_with_sech2_dist = true;
	instant_ngp.m_nerf.training.DS_nerf_supervision_lambda = 1.0f;
	instant_ngp.m_nerf.training.DS_nerf_supervision_depth_sigma = 0.001f;
	instant_ngp.m_nerf.training.DS_nerf_supervision_sech2_scale = 10000.f;
	instant_ngp.m_nerf.training.DS_nerf_supervision_sech2_norm = 22026.4648f;
	instant_ngp.m_nerf.training.DS_nerf_supervision_sech2_int_A = -3876.8233442812702;
	instant_ngp.m_nerf.training.DS_nerf_supervision_sech2_int_B = 2202.6465749406784;

    instant_ngp.m_use_volsdf_in_nerf = false;
    instant_ngp.m_use_coslam_sdf_in_nerf = false;
    instant_ngp.m_nerf.training.volsdf_beta = 0.1;
    instant_ngp.m_add_sdf_loss = false;
    instant_ngp.m_add_sdf_free_space_loss = false;
    instant_ngp.m_nerf.training.sdf_supervision_lambda= 100000.0;
    instant_ngp.m_nerf.training.sdf_free_space_supervision_lambda= 1.0;

    //instant_ngp.m_nerf.training.sdf_supervision_lambda= 5000.0;
    instant_ngp.m_nerf.training.truncation_distance = 0.1 * poses_scale * scale;

	instant_ngp.m_nerf.training.truncation_distance_for_depth_guided_sampling = 0.1 * poses_scale * scale;
	instant_ngp.m_nerf.training.dt_for_depth_guided_sampling = 0.01 * poses_scale * scale;

    instant_ngp.m_nerf.training.depth_loss_type = ELossType::L2;
    instant_ngp.m_nerf.density_activation = ENerfActivation::None;

    instant_ngp.m_train = true;
    instant_ngp.m_train_encoding = true;
    instant_ngp.m_train_network = true;
    uint32_t batch_size=256000;
    for (uint32_t i=0; i<50; ++i) {
        instant_ngp.map(batch_size);
        tlog::info()<<"  ----- mapping loss: "<<instant_ngp.m_loss_scalar.val();
    }

    tlog::info()<<" Render first frame";
    // auto frame = instant_ngp.render_to_cpu(res.x, res.y, 8, true, -1.f, -1.f, 30.0f, 1.0f);

	auto sample_start_cam_matrix = instant_ngp.m_camera;
	auto sample_end_cam_matrix = instant_ngp.m_camera;
	auto prev_camera_matrix = instant_ngp.m_camera;
	
    instant_ngp.m_windowless_render_surface.resize({res.x, res.y});
	instant_ngp.m_windowless_render_surface.reset_accumulation();


        instant_ngp.render_frame(
			instant_ngp.m_stream.get(),
			sample_start_cam_matrix,
			sample_end_cam_matrix,
			prev_camera_matrix,
			instant_ngp.m_screen_center,
			instant_ngp.m_relative_focal_length,
			{0.0f, 0.0f, 0.0f, 1.0f},
			{},
			{},
			instant_ngp.m_visualized_dimension,
			instant_ngp.m_windowless_render_surface,
			false
		);
	

    // train tracking on second image
    tlog::info()<<" Start Tracking";
    instant_ngp.m_nerf.training.indice_image_for_tracking_pose = 1;
    instant_ngp.m_nerf.training.extrinsic_learning_rate_pos = 0.0005;
    instant_ngp.m_nerf.training.extrinsic_learning_rate_rot = 0.001;
    // instant_ngp.m_nerf.training.track_loss_type = ngp.LossType.L2;
    // instant_ngp.m_nerf.training.track_depth_loss_type = ngp.LossType.L1
    instant_ngp.m_nerf.training.depth_supervision_lambda = 0.0f;
    instant_ngp.m_tracking_mode=0;

    batch_size=256000 * 5;

    for (uint32_t i=0; i<1000; ++i) {
        instant_ngp.track(batch_size);
    }



}
