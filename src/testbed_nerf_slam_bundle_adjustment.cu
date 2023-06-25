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

void Testbed::bundle_adjustment(uint32_t batch_size) {
	if (!m_training_data_available || m_camera_path.rendering) {
		m_train = false;
		return;
	}

	if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Cannot train without a mode."};
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

	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (m_nerf.training.optimize_extra_dims) {
			if (m_nerf.training.dataset.n_extra_learnable_dims == 0) {
				m_nerf.training.dataset.n_extra_learnable_dims = 16;
				reset_network();
			}
		}
	}

	if (!m_dlss) {
		// No immediate redraw necessary
		reset_accumulation(false, false);
	}

	uint32_t n_prep_to_skip = m_testbed_mode == ETestbedMode::Nerf ? tcnn::clamp(m_ba_step / 16u, 1u, 16u) : 1u;
	if ((m_ba_step % n_prep_to_skip == 0) || (m_reset_prep_nerf_mapping) ) {
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_prep_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count() / n_prep_to_skip);
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf: training_prep_nerf_ba(batch_size, m_stream.get()); break;
			default: throw std::runtime_error{"Invalid training mode (for SLAM it has to be Nerf)."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
		
		m_reset_prep_nerf_mapping = false;
	}

	// Find leaf optimizer and update its settings
	json* leaf_optimizer_config = &m_network_config["optimizer"];
	while (leaf_optimizer_config->contains("nested")) {
		leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
	}
	(*leaf_optimizer_config)["optimize_matrix_params"] = m_train_network;
	(*leaf_optimizer_config)["optimize_non_matrix_params"] = m_train_encoding;
	m_optimizer->update_hyperparams(m_network_config["optimizer"]);

	//bool get_loss_scalar = true;
	bool get_loss_scalar = m_ba_step % 16 == 0;

	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count());
		}};

		train_nerf_slam_bundle_adjustment(batch_size, get_loss_scalar, m_stream.get());

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	if (get_loss_scalar) {
		update_loss_graph();
	}
}




void Testbed::training_prep_nerf_ba(uint32_t batch_size, cudaStream_t stream) {
	if (m_nerf.training.n_images_for_training == 0) {
		return;
	}

	float alpha = m_nerf.training.density_grid_decay;
	uint32_t n_cascades = m_nerf.max_cascade+1;

	//DEBUG	
	//DEBUG	
	alpha = 1.0;
	update_density_grid_nerf_ba(alpha, NERF_GRID_N_CELLS() * n_cascades, 0, stream);
	//DEBUG	
	//DEBUG	

	//if (m_ba_step < 256) {
	//	update_density_grid_nerf_ba(alpha, NERF_GRID_N_CELLS() * n_cascades, 0, stream);
	//} else {
	//	update_density_grid_nerf_ba(alpha, NERF_GRID_N_CELLS() / 4 * n_cascades, NERF_GRID_N_CELLS() / 4 * n_cascades, stream);
	//}
}


float Testbed::get_max_level() {
	if (!m_network) return -1.f;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		float maxlevel = hg_enc->max_level();
		return maxlevel;
	} else {
		return -1.f;
	}
}



void Testbed::sample_pixels_for_ba_with_gaussian_pyramid(
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
    const uint32_t& level,
    const std::vector<uint32_t>& idx_images_for_mapping,
    std::vector<uint32_t>& image_ids
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

    uint32_t cpt=0;
    while ( (ray_counter + ray_stride) < max_rays_per_batch) {

        ++super_ray_counter;
		
		// sample image
		int rng_i = std::rand() % idx_images_for_mapping.size();
		uint32_t img = idx_images_for_mapping[rng_i];

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
				//NOTE: hacky way to not do it - very unlikely since we're sampling across diff images
                ray_mapping.push_back(cpt);
                ++ray_counter_for_gradient;

				// add image id info
    			image_ids.push_back(img);

                ++cpt;

            }
        }

    }

}











NGP_NAMESPACE_END