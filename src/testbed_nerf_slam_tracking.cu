#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/encodings/spherical_harmonics.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>


#ifdef copysign
#undef copysign
#endif

using namespace tcnn;

NGP_NAMESPACE_BEGIN

void Testbed::track(uint32_t batch_size) {
	if (!m_training_data_available || m_camera_path.rendering) {
		m_train = false;
		return;
	}

	if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Cannot train without a mode. -> Mode has to be Nerf for SLAM"};
	}

	set_all_devices_dirty();

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (!m_trainer) {
		reload_network_from_file();
		if (!m_trainer) {
			throw std::runtime_error{"Unable to create a neural network trainer."};
		}
	}
	
    if (!m_dlss) {
		// No immediate redraw necessary
		reset_accumulation(false, false);
	}
	
    // Find leaf optimizer and update its settings
	json* leaf_optimizer_config = &m_network_config["optimizer"];
	while (leaf_optimizer_config->contains("nested")) {
		leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
	}
	(*leaf_optimizer_config)["optimize_matrix_params"] = m_train_network;
	(*leaf_optimizer_config)["optimize_non_matrix_params"] = m_train_encoding;

    if ((m_train_network!=false) or (m_train_encoding!=false)) {
		throw std::runtime_error{"Tracking only. No grid or MLP updates. You'll have to turn off train_nertwork and train_encoding during tracking."};
    }

	m_optimizer->update_hyperparams(m_network_config["optimizer"]);

	bool get_loss_scalar = true;
	{
		
		train_nerf_slam_tracking(batch_size, get_loss_scalar, m_stream.get());

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

}


void Testbed::get_receptive_field_of_gaussian_pyramid_at_level(uint32_t level, std::vector<int>& rf) {
    if (rf.empty()){
        rf = {0,0,0,0};
    }
    if (level==0){
        return;
    } else {
        rf[0] = 2*rf[0] - 2;
        rf[1] = 2*rf[1] + 2;
        rf[2] = 2*rf[2] - 2;
        rf[3] = 2*rf[3] + 2;
        return get_receptive_field_of_gaussian_pyramid_at_level(level-1, rf);
    }
}

void Testbed::sample_pixels_for_tracking_with_gaussian_pyramid(
    const uint32_t max_rays_per_batch,
    uint32_t& ray_counter,
    uint32_t& super_ray_counter,
    uint32_t& ray_counter_for_gradient,
    std::vector<float>& xy_image_pixel_indices,
    std::vector<uint32_t>& xy_image_pixel_indices_int,
    std::vector<uint32_t>& xy_image_super_pixel_at_level_indices_int_cpu,
    std::vector<uint32_t>& ray_mapping,
    const bool snap_to_pixel_centers,
    const uint32_t sample_away_from_border_margin_h,
    const uint32_t sample_away_from_border_margin_w,
    default_rng_t& rng,
    const std::vector<int>& rf,
    const uint32_t& super_ray_window_size,
    const uint32_t& ray_stride,
	const ivec2& resolution,
    const ivec2& resolution_at_level,
    const uint32_t& level
) {
    // get the at level sampling margins
    int margin_r = (int) ceil(((float) (sample_away_from_border_margin_h + (uint32_t) rf[1])) / pow(2.0, (float) level));
    int margin_c = (int) ceil(((float) (sample_away_from_border_margin_w + (uint32_t) rf[3])) / pow(2.0, (float) level));
    ivec2 margins_at_level(margin_c, margin_r);
    ivec2 bounds_at_level = resolution_at_level - 2*margins_at_level;

    //init vars
    vec2 rng_xy;
    ivec2 xy_at_level_int;
    vec2 xy_at_level;
    vec2 xy;
    ivec2 half_kernel_size(rf[1], rf[3]);
    vec2 half_kernel_size_float = vec2(half_kernel_size);
    ivec2 tmp_uv;
    vec2 tmp_d;
    vec2 tmp_xy_int;
    vec2 tmp_xy;
    uint32_t key;

    uint32_t cpt=0;
    std::unordered_map<uint32_t, uint32_t> tmp_dict;
    while ( (ray_counter + ray_stride) <= max_rays_per_batch) {

        ++super_ray_counter;

        // sample a pixel at level
        rng_xy.x = rng.next_float();
        rng_xy.y = rng.next_float();

        xy_at_level_int = ivec2(rng_xy * vec2(bounds_at_level));
        xy_at_level_int = xy_at_level_int + margins_at_level; // sampled pixel (int) at level

		xy_at_level_int = clamp(xy_at_level_int, margins_at_level, resolution_at_level-margins_at_level - ivec2(1));

        xy_image_super_pixel_at_level_indices_int_cpu.push_back(xy_at_level_int.x);
        xy_image_super_pixel_at_level_indices_int_cpu.push_back(xy_at_level_int.y);

        xy_at_level = vec2(xy_at_level_int);

        // grab the corresponding pixels in OG rez
		float _level = pow(2.0, (float) level);
        xy = xy_at_level * _level;

        // populate the output vectors
	    for (uint32_t u = 0; u < super_ray_window_size; ++u) {
	        for (uint32_t v = 0; v < super_ray_window_size; ++v) {

                ++ray_counter;

                tmp_uv = ivec2(v,u);
                tmp_d = vec2(tmp_uv) - half_kernel_size_float;
                tmp_xy_int = xy + tmp_d;

	            if (snap_to_pixel_centers) {
	            	tmp_xy = (tmp_xy_int + vec2(0.5f)) / vec2(resolution);
	            } else {
	            	tmp_xy = tmp_xy_int / vec2(resolution);
                }

                xy_image_pixel_indices.push_back(tmp_xy.x);
                xy_image_pixel_indices.push_back(tmp_xy.y);

                xy_image_pixel_indices_int.push_back(tmp_xy_int.x);
                xy_image_pixel_indices_int.push_back(tmp_xy_int.y);

                // check if ray exists
                key = tmp_xy_int.x + tmp_xy_int.y * resolution.x;
                //if (tmp_dict.count(key) == 0) {
                if (true) {
                    ray_mapping.push_back(cpt);
                    tmp_dict[key] = cpt;
                    ++ray_counter_for_gradient;
                } else {
                    ray_mapping.push_back(tmp_dict[key]);
                }

                ++cpt;

            }
        }

    }

}

std::vector<float> Testbed::make_5tap_kernel() {
    uint32_t kernel_size = 5;
    std::vector<float> kernel(kernel_size * kernel_size);

    float a = 3.0 / 8.0;
    std::vector<float> tmp{0.25f - a * 0.5f, 0.25f, a, 0.25f, 0.25f - a * 0.5f};

    uint32_t cpt=0;
    for (uint32_t i=0; i < kernel_size; i++){
        for (uint32_t j=0; j < kernel_size; j++){
            kernel[cpt] = tmp[i] * tmp[j];
            ++cpt;
        }
    }
    return kernel;
}









void Testbed::track_steps(
    const uint32_t cam_id, 
    const uint32_t target_batch_size, 
    const uint32_t margin_h, 
    const uint32_t margin_w, 
    const uint32_t tracking_mode, 
    const int num_rays_to_sample, 
    float lr,
    float pos_lr,
    float rot_lr,
    const bool separate_pos_and_rot_lr,
    const std::vector<std::map<std::string, float> >& tracking_hyperparameters
    ) {
    
    //assumes the camera pose of cam_id has been initialized already

    // init variables before tracking
    m_nerf.training.sample_image_proportional_to_error = false;
    m_nerf.training.optimize_extrinsics = false;
    m_nerf.training.optimize_exposure = false;
    m_nerf.training.optimize_extra_dims = false;
    m_nerf.training.optimize_distortion = false;
    m_nerf.training.optimize_focal_length = false;
    m_nerf.training.include_sharpness_in_error = false;
    m_nerf.training.indice_image_for_tracking_pose = cam_id;
    m_nerf.training.n_steps_between_cam_updates = 1;
    m_nerf.training.n_steps_since_cam_update = 0;
    m_nerf.training.n_steps_since_error_map_update = 0;
    m_nerf.training.m_sample_away_from_border_margin_h_tracking = margin_h;
    m_nerf.training.m_sample_away_from_border_margin_w_tracking = margin_w;
    m_train_encoding = false;
    m_train_network = false;
    m_train = true;
    m_max_level_rand_training = false;
    m_nerf.training.use_depth_var_in_tracking_loss = true;
    m_tracking_mode = tracking_mode;
    if (num_rays_to_sample > 0) {
        m_nerf.training.m_target_num_rays_for_tracking = num_rays_to_sample;
        m_nerf.training.m_set_fix_num_rays_to_sample = true;
    } else {
        m_nerf.training.m_set_fix_num_rays_to_sample = false;
    }
    if (separate_pos_and_rot_lr) {
        m_nerf.training.extrinsic_learning_rate_pos = pos_lr;
        m_nerf.training.extrinsic_learning_rate_rot = rot_lr;
    } else {
        m_nerf.training.extrinsic_learning_rate_pos = lr;
        m_nerf.training.extrinsic_learning_rate_rot = lr;
    }

    //do tracking:
    float min_loss;
    float cur_loss;
    auto cur_c2w = m_nerf.training.get_camera_extrinsics(cam_id); 
    auto final_c2w = m_nerf.training.get_camera_extrinsics(cam_id); 
    for (uint32_t i=0; i<tracking_hyperparameters.size(); i++) {
        
        std::map<std::string, float> e = tracking_hyperparameters[i];

        uint32_t iterations = e["iterations"];
        float lr_factor = e["lr_factor"];
        uint32_t n_steps_between_cam_updates_tracking = e["n_steps_between_cam_updates"];

        m_nerf.training.n_steps_between_cam_updates = n_steps_between_cam_updates_tracking;

        if (separate_pos_and_rot_lr) {
            m_nerf.training.extrinsic_learning_rate_pos = pos_lr * lr_factor;
            m_nerf.training.extrinsic_learning_rate_rot = rot_lr * lr_factor;
        } else {
            m_nerf.training.extrinsic_learning_rate_pos = lr * lr_factor;
            m_nerf.training.extrinsic_learning_rate_rot = lr * lr_factor;
        }

        min_loss = 10000000.;
        cur_loss = 0.;
        cur_c2w = m_nerf.training.get_camera_extrinsics(cam_id); 
        
        for (uint32_t ite=0; ite<iterations; ite++) {
            track(target_batch_size);
            float loss = m_loss_scalar_tracking.val();
            uint32_t measured_batch_size = m_nerf.training.counters_rgb_tracking.measured_batch_size;
            loss = loss * (float)target_batch_size / (float)measured_batch_size;
            cur_loss += loss;

            if ((ite+1)%n_steps_between_cam_updates_tracking==0) {
                cur_loss /= (float)n_steps_between_cam_updates_tracking;
                if (cur_loss < min_loss) {
                    min_loss = cur_loss;
                    final_c2w = cur_c2w; ///!\ need to make sure this is properly copied.
                }
                cur_loss = 0.;
                cur_c2w = m_nerf.training.get_camera_extrinsics(cam_id); 
            }
        }
    }
    m_nerf.training.set_camera_extrinsics(cam_id, final_c2w, true); 
}






NGP_NAMESPACE_END