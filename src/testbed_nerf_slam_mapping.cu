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

void Testbed::map(uint32_t batch_size) {
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

	uint32_t n_prep_to_skip = m_testbed_mode == ETestbedMode::Nerf ? tcnn::clamp(m_training_step / 16u, 1u, 16u) : 1u;
	if (m_training_step % n_prep_to_skip == 0) {
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_prep_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count() / n_prep_to_skip);
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf: training_prep_nerf(batch_size, m_stream.get()); break;
			default: throw std::runtime_error{"Invalid training mode (for SLAM it has to be Nerf)."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	// Find leaf optimizer and update its settings
	json* leaf_optimizer_config = &m_network_config["optimizer"];
	while (leaf_optimizer_config->contains("nested")) {
		leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
	}
	(*leaf_optimizer_config)["optimize_matrix_params"] = m_train_network;
	(*leaf_optimizer_config)["optimize_non_matrix_params"] = m_train_encoding;
	m_optimizer->update_hyperparams(m_network_config["optimizer"]);

	//bool get_loss_scalar = m_training_step % 16 == 0;
	bool get_loss_scalar = true;

	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count());
		}};

		train_nerf_slam(batch_size, get_loss_scalar, m_stream.get());

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	if (get_loss_scalar) {
		update_loss_graph();
	}
}


NGP_NAMESPACE_END