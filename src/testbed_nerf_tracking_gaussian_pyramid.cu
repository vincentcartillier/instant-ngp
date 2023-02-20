#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/envmap.cuh>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <cmath>

#include <testbed_nerf_utils.cu>

#ifdef copysign
#undef copysign
#endif

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN

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


__global__ void generate_training_samples_for_tracking_gp(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	float cone_angle_constant,
	const float* __restrict__ distortion_data,
	const Vector2i distortion_resolution,
	const float* __restrict__ extra_dims_gpu,
	uint32_t n_extra_dims,
	const uint32_t indice_image_for_tracking_pose,
	int32_t* __restrict__ mapping_indices,
    const uint32_t* __restrict__ existing_ray_mapping_gpu,
    const float* __restrict__ xy_image_pixel_indices,
	const bool use_view_dir
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	mapping_indices[i] = -1; //  default to "not existing"

    if (existing_ray_mapping_gpu[i]!=i) return;

    uint32_t img = indice_image_for_tracking_pose;

	Eigen::Vector2i resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
    Vector2f xy = Vector2f(xy_image_pixel_indices[2*i], xy_image_pixel_indices[2*i+1]);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(xy, resolution, 0);
	if (read_rgba(xy, resolution, metadata[img].pixels, metadata[img].image_data_type).x() < 0.0f) {
		return;
	}

	float motionblur_time = 0.0;
	const Vector2f focal_length = metadata[img].focal_length;
	const Vector2f principal_point = metadata[img].principal_point;
	const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
	const Lens lens = metadata[img].lens;

	const Matrix<float, 3, 4> xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, xy, motionblur_time);

	Ray ray_unnormalized;
	const Ray* rays_in_unnormalized = metadata[img].rays;
	if (rays_in_unnormalized) {
		// Rays have been explicitly supplied. Read them.
		ray_unnormalized = rays_in_unnormalized[pix_idx];
	} else {
		// Rays need to be inferred from the camera matrix
		ray_unnormalized.o = xform.col(3);
		if (lens.mode == ELensMode::FTheta) {
			ray_unnormalized.d = f_theta_undistortion(xy - principal_point, lens.params, {0.f, 0.f, 1.f});
		} else if (lens.mode == ELensMode::LatLong) {
			ray_unnormalized.d = latlong_to_dir(xy);
		} else {
			ray_unnormalized.d = {
				(xy.x()-principal_point.x())*resolution.x() / focal_length.x(),
				(xy.y()-principal_point.y())*resolution.y() / focal_length.y(),
				1.0f,
			};

			if (lens.mode == ELensMode::OpenCV) {
				iterative_opencv_lens_undistortion(lens.params, &ray_unnormalized.d.x(), &ray_unnormalized.d.y());
			}
		}

		if (distortion_data) {
			ray_unnormalized.d.head<2>() += read_image<2>(distortion_data, distortion_resolution, xy);
		}

		ray_unnormalized.d = (xform.block<3, 3>(0, 0) * ray_unnormalized.d); // NOT normalized
	}

	Eigen::Vector3f ray_d_normalized = ray_unnormalized.d.normalized();

	Vector2f tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
	float cone_angle = calc_cone_angle(ray_d_normalized.dot(xform.col(2)), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x() = fmaxf(tminmax.x(), 0.0f);

	float startt = tminmax.x();
	startt += calc_dt(startt, cone_angle) * random_val(rng);
	Vector3f idir = ray_d_normalized.cwiseInverse();

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t=startt;
	Vector3f pos;

	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, res);
		}
	}
	if (j == 0) {
		return;
	}
	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	if (base + numsteps > max_samples) {
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	ray_indices_out[ray_idx] = i;
	mapping_indices[i] = ray_idx;
	rays_out_unnormalized[ray_idx] = ray_unnormalized;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	Vector3f warped_dir;
	if (use_view_dir) {
		warped_dir = warp_direction(ray_d_normalized);
	} else {
		warped_dir = Vector3f::Zero();
	}
	t=startt;
	j=0;
	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
			++j;
			t += dt;
		} else {
			uint32_t res = NERF_GRIDSIZE()>>mip;
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, res);
		}
	}
}


__global__ void compute_depth_variance_gp(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t* __restrict__ ray_counter,
	int padded_output_width,
	const tcnn::network_precision_t* network_output,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	const float* __restrict__ reconstructed_rgbd,
	float* __restrict__ reconstructed_depth_var,
	float* __restrict__ reconstructed_color_var
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *ray_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

    const float rec_depth = reconstructed_rgbd[i*4+3];
    const float rec_color = (reconstructed_rgbd[i*4+0] + reconstructed_rgbd[i*4+1] + reconstructed_rgbd[i*4+2]) / 3.0;

	float T = 1.f;

	float EPSILON = 1e-4f;

    float depth_var = 0.f;
    float color_var = 0.f;

    Eigen::Vector3f ray_o = rays_in_unnormalized[i].o;

	for (uint32_t j=0; j < numsteps; ++j) {
		if (T < EPSILON) {
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		float density = network_to_density(float(local_network_output[3]), density_activation);

        const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = (pos - ray_o).norm();
        float cur_color = (rgb.x()+rgb.y()+rgb.z()) / 3.0;

        float tmp = (cur_depth - rec_depth) * (cur_depth - rec_depth);
        float tmp_color = (cur_color - rec_color) * (cur_color - rec_color);


		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;

        depth_var += weight * tmp;
        color_var += weight * tmp_color;
		T *= (1.f - alpha);

		network_output += padded_output_width;
		coords_in += 1;
	}

    reconstructed_depth_var[i] = depth_var;
    reconstructed_color_var[i] = color_var;
}




__global__ void compute_GT_and_reconstructed_rgbd_gp(
	const uint32_t n_rays,
	BoundingBox aabb,
	default_rng_t rng,
	const uint32_t target_batch_size,
	const uint32_t* __restrict__ ray_counter,
	int padded_output_width,
	const float* __restrict__ envmap_data,
	const Vector2i envmap_resolution,
	Array3f background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const TrainingImageMetadata* __restrict__ metadata,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	const Eigen::Array3f* __restrict__ exposure,
	float depth_supervision_lambda,
	const uint32_t indice_image_for_tracking_pose,
    const float* __restrict__ xy_image_pixel_indices,
	float* __restrict__ ground_truth_rgbd,
	float* __restrict__ reconstructed_rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *ray_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float EPSILON = 1e-4f;

	Array3f rgb_ray = Array3f::Zero();
	Vector3f hitpoint = Vector3f::Zero();
	float depth_ray = 0.f;

    Eigen::Vector3f ray_o = rays_in_unnormalized[i].o;

    uint32_t compacted_numsteps = 0;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T < EPSILON) {
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
		const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = (pos - ray_o).norm();
		float density = network_to_density(float(local_network_output[3]), density_activation);


		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray += weight * rgb;
		hitpoint += weight * pos;
		depth_ray += weight * cur_depth;
		T *= (1.f - alpha);

		network_output += padded_output_width;
		coords_in += 1;
	}
	hitpoint /= (1.0f - T);

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	uint32_t img = indice_image_for_tracking_pose;
	Eigen::Vector2i resolution = metadata[img].resolution;

    Vector2f xy = Vector2f(xy_image_pixel_indices[2*ray_idx], xy_image_pixel_indices[2*ray_idx+1]);

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	Array3f pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	Array4f envmap_value;
	Vector3f dir;
	if (envmap_data) {
		dir = rays_in_unnormalized[i].d.normalized();
		envmap_value = read_envmap(envmap_data, envmap_resolution, dir);
		background_color = envmap_value.head<3>() + background_color * (1.0f - envmap_value.w());
	}

	Array3f exposure_scale = (0.6931471805599453f * exposure[img]).exp();

    Array4f texsamp = read_rgba(xy, resolution, metadata[img].pixels, metadata[img].image_data_type);

	Array3f rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.head<3>() + (1.0f - texsamp.w()) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.w() > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.head<3>() / texsamp.w()) * texsamp.w() + (1.0f - texsamp.w()) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

    float target_depth = rays_in_unnormalized[i].d.norm() * ((depth_supervision_lambda > 0.0f && metadata[img].depth) ? read_depth(xy, resolution, metadata[img].depth) : -1.0f);

    ground_truth_rgbd[i*4+0] = rgbtarget.x();
    ground_truth_rgbd[i*4+1] = rgbtarget.y();
    ground_truth_rgbd[i*4+2] = rgbtarget.z();
    ground_truth_rgbd[i*4+3] = target_depth;

    reconstructed_rgbd[i*4+0] = rgb_ray.x();
    reconstructed_rgbd[i*4+1] = rgb_ray.y();
    reconstructed_rgbd[i*4+2] = rgb_ray.z();
    reconstructed_rgbd[i*4+3] = depth_ray;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(target_batch_size - min(target_batch_size, compacted_base), compacted_numsteps);
	numsteps_out[i*2+0] = compacted_numsteps;
	numsteps_out[i*2+1] = compacted_base;

}



__global__ void apply_photometric_correction_to_GT(
	const uint32_t n_rays,
	const uint32_t* __restrict__ ray_counter,
	const uint32_t indice_image_for_tracking_pose,
    const float* __restrict__ image_photometric_correction_params_coef,
    const float* __restrict__ image_photometric_correction_params_intercept,
	float* __restrict__ ground_truth_rgbd
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *ray_counter) { return; }

	float coef = image_photometric_correction_params_coef[indice_image_for_tracking_pose];
	float intercept = image_photometric_correction_params_intercept[indice_image_for_tracking_pose];

    ground_truth_rgbd[i*4+0] = coef * ground_truth_rgbd[i*4+0] + intercept;
    ground_truth_rgbd[i*4+1] = coef * ground_truth_rgbd[i*4+1] + intercept;
    ground_truth_rgbd[i*4+2] = coef * ground_truth_rgbd[i*4+2] + intercept;
}















__global__ void convolution_gaussian_pyramid(
	const uint32_t n_target_rays,
	const uint32_t n_input_rays,
    const uint32_t window_size_input,
    const uint32_t window_size_output,
    const uint32_t ray_stride_input,
    const uint32_t ray_stride_output,
    const float* __restrict__ kernel,
    const uint32_t* __restrict__ existing_ray_mapping_gpu,
    const int32_t* __restrict__ mapping_indices,
    float* __restrict__ gt_rgbd_input,
    float* __restrict__ rec_rgbd_input,
    float* __restrict__ rec_depth_var_input,
    float* __restrict__ rec_color_var_input,
    float* __restrict__ rec_depth_var_output,
    float* __restrict__ rec_color_var_output,
    float* __restrict__ gt_rgbd_output,
    float* __restrict__ rec_rgbd_output,
    float* __restrict__ gradients
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_target_rays) { return; }

    // given i compute the window indices in input array
    // to do so first compute position of pixel at level
    // j = i%ray_strid_at_level # get the super ray indice
    // u = j / window_size_at_level
    // v = j % window_size_at_level;
    uint32_t i_prime = i / ray_stride_output;
    uint32_t j = i % ray_stride_output;
    uint32_t u = j / window_size_output;
    uint32_t v = j % window_size_output;

    // get corresponding indices in input array
    // get the start index of the super ray in input
    // a = j * ray_strid_at_prev_level
    // u_prime = u * 2 + 2
    // v_prime = v * 2 + 2
    uint32_t a = i_prime * ray_stride_input;
    uint32_t u_prime = u * 2 + 2;
    uint32_t v_prime = v * 2 + 2;

    // loop thru n = u_prime - kernel_ws/2 to u_prime + ws/2 +1
    // loop thru m = v_prime - kernel_ws/2 to v_prime + ws/2 +1
    // map (n,m,a) to index in input array
    // index = a + n * window_size_at_prev_level + m
    uint32_t cpt=0;
	Array3f rec_rgb = Array3f::Zero();
	Array3f gt_rgb = Array3f::Zero();
    float rec_depth=0.f;
    float gt_depth=0.f;
    uint32_t index;
    uint32_t index_map;
    uint32_t gi;

    float norm=0.f;
    float norm_depth=0.f;

    float rec_depth_var = 0.f;
    float rec_color_var = 0.f;

    for (uint32_t n=(u_prime-2); n<(u_prime+2+1); ++n) {
        for (uint32_t m=(v_prime-2); m<(v_prime+2+1); ++m) {
            index = a + n * window_size_input + m;
            //if mapping check mapped index
            if (existing_ray_mapping_gpu) {
                index_map = existing_ray_mapping_gpu[index];
            } else {
                index_map = index;
            }

            if (mapping_indices) {
                int32_t ray_idx = mapping_indices[index_map];
                if (ray_idx < 0) {
                    continue;
                } else {
                    index_map = ray_idx;
                }
            }

            norm += kernel[cpt];

            gt_rgb.x() += kernel[cpt] * gt_rgbd_input[index_map*4+0];
            gt_rgb.y() += kernel[cpt] * gt_rgbd_input[index_map*4+1];
            gt_rgb.z() += kernel[cpt] * gt_rgbd_input[index_map*4+2];

            rec_rgb.x() += kernel[cpt] * rec_rgbd_input[index_map*4+0];
            rec_rgb.y() += kernel[cpt] * rec_rgbd_input[index_map*4+1];
            rec_rgb.z() += kernel[cpt] * rec_rgbd_input[index_map*4+2];
            rec_depth   += kernel[cpt] * rec_rgbd_input[index_map*4+3];

            if (rec_depth_var_input) {
                rec_depth_var += kernel[cpt] * rec_depth_var_input[index_map];
            }
            if (rec_color_var_input) {
                rec_color_var += kernel[cpt] * rec_color_var_input[index_map];
            }

            float tmp_gt_depth = gt_rgbd_input[index_map*4+3];
            if (tmp_gt_depth > 0) {
                gt_depth   += kernel[cpt] * tmp_gt_depth;
                norm_depth += kernel[cpt];
            }

            gi = i * n_input_rays + index;
            gradients[gi] = kernel[cpt];

            ++cpt;
        }
    }

    if (norm > 0) {
        gt_rgb /= norm;
        rec_rgb /= norm;
        rec_depth /= norm;
        rec_depth_var /= norm;
        rec_color_var /= norm;
    }

    if (norm_depth>0) {
        gt_depth /= norm_depth;
    }

    gt_rgbd_output[i*4+0] = gt_rgb.x();
    gt_rgbd_output[i*4+1] = gt_rgb.y();
    gt_rgbd_output[i*4+2] = gt_rgb.z();
    gt_rgbd_output[i*4+3] = gt_depth;

    rec_rgbd_output[i*4+0] = rec_rgb.x();
    rec_rgbd_output[i*4+1] = rec_rgb.y();
    rec_rgbd_output[i*4+2] = rec_rgb.z();
    rec_rgbd_output[i*4+3] = rec_depth;

    if (rec_depth_var_output) {
        rec_depth_var_output[i] = rec_depth_var;
    }

    if (rec_color_var_output) {
        rec_color_var_output[i] = rec_color_var;
    }


}

__global__ void compute_loss_gp(
	const uint32_t n_rays,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	float* __restrict__ loss_depth_output,
    const bool use_depth_var_in_loss,
	const bool use_color_var_in_loss,
    const uint32_t* __restrict__ existing_ray_mapping_gpu,
    const int32_t* __restrict__ mapping_indices,
    const float* __restrict__ ground_truth_rgbd,
	const float* __restrict__ reconstructed_rgbd,
	const float* __restrict__ reconstructed_depth_var,
	const float* __restrict__ reconstructed_color_var,
	float* __restrict__ losses_and_gradients,
	uint32_t* __restrict__ super_ray_counter,
	uint32_t* __restrict__ super_ray_counter_depth
) {

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) { return; }

    uint32_t super_i;
    if (existing_ray_mapping_gpu) {
        super_i = existing_ray_mapping_gpu[i];
    } else {
        super_i = i;
    }
    if (mapping_indices) {
        int32_t ray_idx = mapping_indices[super_i];
        if (ray_idx < 0) {
            // No output for the sampled ray
            losses_and_gradients[i*4+0] = 0.f;
            losses_and_gradients[i*4+1] = 0.f;
            losses_and_gradients[i*4+2] = 0.f;
            losses_and_gradients[i*4+3] = 0.f;
	        if (loss_output) {
	        	loss_output[i] = 0.f;
	        }
	        if (loss_depth_output) {
                loss_depth_output[i] = 0.f;
	        }
            return;
        } else {
            super_i = ray_idx;
        }
    }

    uint32_t num_super_rays_in_loss = atomicAdd(super_ray_counter, 1);	 // first entry in the array is a counter

	Array3f avg_rgb_ray {
        reconstructed_rgbd[super_i*4+0],
        reconstructed_rgbd[super_i*4+1],
        reconstructed_rgbd[super_i*4+2]
    };
	float avg_depth_ray = reconstructed_rgbd[super_i*4+3];

    Array3f avg_rgb_ray_target {
        ground_truth_rgbd[super_i*4+0],
        ground_truth_rgbd[super_i*4+1],
        ground_truth_rgbd[super_i*4+2]
    };
    float avg_depth_ray_target = ground_truth_rgbd[super_i*4+3];

    LossAndGradient lg = loss_and_gradient(avg_rgb_ray_target, avg_rgb_ray, loss_type);
    LossAndGradient lg_depth;

    if (avg_depth_ray_target>0) {
	    lg_depth = loss_and_gradient(Array3f::Constant(avg_depth_ray_target), Array3f::Constant(avg_depth_ray), depth_loss_type);

        if ((lg_depth.loss.x()==0.f)) {
            lg_depth.gradient.x() = 0.f;
        }
	    uint32_t num_super_rays_depth_in_loss = atomicAdd(super_ray_counter_depth, 1);	 // first entry in the array is a counter

    } else {
	    lg_depth = loss_and_gradient(Array3f::Constant(0.f), Array3f::Constant(0.f), depth_loss_type);
        lg_depth.gradient.x() = 0.f;
    }

    if (use_depth_var_in_loss) {
	    float rec_depth_var = reconstructed_depth_var[super_i];

        if (rec_depth_var < 1e-6){
            rec_depth_var = 1e-6;
        }
        
		float rec_depth_std = sqrt(rec_depth_var);
        
		lg_depth.loss.x() /= rec_depth_std;
        lg_depth.gradient.x() /= rec_depth_std;

    }
	
	if (use_color_var_in_loss) {
	    float rec_color_var = reconstructed_color_var[super_i];
        
		if (rec_color_var < 1e-6){
            rec_color_var = 1e-6;
        }
        
		float rec_color_std = sqrt(rec_color_var);

        lg.loss /= rec_color_std;
        lg.gradient /= rec_color_std;

	}

    losses_and_gradients[i*4+0] = lg.gradient.x();
    losses_and_gradients[i*4+1] = lg.gradient.y();
    losses_and_gradients[i*4+2] = lg.gradient.z();
    losses_and_gradients[i*4+3] = lg_depth.gradient.x();

    float mean_loss = lg.loss.mean();

	if (loss_output) {
		loss_output[i] = mean_loss;
	}
	if (loss_depth_output) {
        loss_depth_output[i] = lg_depth.loss.x();
	}
}


__global__ void backprop_thru_convs(
    const uint32_t input_n_rays,
    const uint32_t output_n_rays, //  conv output n_rays
    const float* __restrict__ prev_partials,
    const float* __restrict__ gradients,
    float* __restrict__ new_partials
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= input_n_rays) { return; }

	Array3f rgb_grad = Array3f::Zero();
    float depth_grad = 0;
    for (uint32_t j=0; j<output_n_rays; ++j) {

        uint32_t index = j * input_n_rays + i;

        rgb_grad.x() += gradients[index] * prev_partials[j*4+0];
        rgb_grad.y() += gradients[index] * prev_partials[j*4+1];
        rgb_grad.z() += gradients[index] * prev_partials[j*4+2];

        depth_grad += gradients[index] * prev_partials[j*4+3];

    }

    new_partials[i*4+0] = rgb_grad.x();
    new_partials[i*4+1] = rgb_grad.y();
    new_partials[i*4+2] = rgb_grad.z();
    new_partials[i*4+3] = depth_grad;
}








__global__ void compute_gradients_wrt_photometric_params_and_update_partial_derivatives(
	const uint32_t one,
	const uint32_t n_rays,
	const uint32_t* __restrict__ existing_ray_mapping_gpu,
    const int32_t* __restrict__ mapping_indices,
	const uint32_t indice_image_for_tracking_pose,
    const float* __restrict__ image_photometric_correction_params_coef,
    const float* __restrict__ image_photometric_correction_params_intercept,
	const float* __restrict__ ground_truth_rgbd,
	float* __restrict__ dL_dC_prime, 
    float* __restrict__ image_photometric_correction_gradient_coef,
    float* __restrict__ image_photometric_correction_gradient_intercept
) {

	float coef = image_photometric_correction_params_coef[indice_image_for_tracking_pose];
	if (coef==0.) { return; }
	float intercept = image_photometric_correction_params_intercept[indice_image_for_tracking_pose];

	float dL_d_coef = 0.f;
	float dL_d_intercept = 0.f;

	float tmp_c;
	uint32_t index_map;
	int32_t ray_idx;

	for (uint32_t k=0; k<n_rays; k++) {

        index_map = existing_ray_mapping_gpu[k];
        ray_idx = mapping_indices[index_map];
        if (ray_idx < 0) {
            continue;
        } else {
            index_map = ray_idx;
        }

		tmp_c = ( ground_truth_rgbd[index_map*4 + 0] - intercept ) / coef;
		dL_d_coef += - dL_dC_prime[k*4 + 0] * tmp_c; 
		dL_d_intercept += - dL_dC_prime[k*4 + 0]; 
		dL_dC_prime[k*4 + 0] *= coef;

		tmp_c = ( ground_truth_rgbd[index_map*4 + 1] - intercept ) / coef;
		dL_d_coef += - dL_dC_prime[k*4 + 1] * tmp_c; 
		dL_d_intercept += - dL_dC_prime[k*4 + 1]; 
		dL_dC_prime[k*4 + 1] *= coef;
		
		tmp_c = ( ground_truth_rgbd[index_map*4 + 2] - intercept ) / coef;
		dL_d_coef += - dL_dC_prime[k*4 + 2] * tmp_c; 
		dL_d_intercept += - dL_dC_prime[k*4 + 2]; 
		dL_dC_prime[k*4 + 2] *= coef;

	}
	image_photometric_correction_gradient_coef[indice_image_for_tracking_pose] = dL_d_coef;
	image_photometric_correction_gradient_intercept[indice_image_for_tracking_pose] = dL_d_intercept;
}



__global__ void compute_gradient_gp(
	const uint32_t n_rays,
	BoundingBox aabb,
	float loss_scale,
	int padded_output_width,
	const TrainingImageMetadata* __restrict__ metadata,
	const tcnn::network_precision_t* network_output,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	uint32_t* __restrict__ numsteps_compacted,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	float* __restrict__ loss_depth_output,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	float depth_supervision_lambda,
	float near_distance,
    float* __restrict__ xy_image_pixel_indices,
	const int32_t* __restrict__ mapping_indices,
	const float* __restrict__ ground_truth_rgbd,
	const float* __restrict__ reconstructed_rgbd,
	const uint32_t* __restrict__ ray_counter,
    const uint32_t* __restrict__ existing_ray_mapping_gpu,
	float* __restrict__ losses_and_gradients,
	const uint32_t* __restrict__ super_ray_counter,
	const uint32_t* __restrict__ super_ray_counter_depth
) {

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) { return; }

    uint32_t prev_i = existing_ray_mapping_gpu[i];

    int32_t ray_idx = mapping_indices[prev_i];

    if (ray_idx < 0){
        return;
    }

    Array3f dloss_db = {
        losses_and_gradients[i*4+0],
        losses_and_gradients[i*4+1],
        losses_and_gradients[i*4+2]
    };

    Array3f rgb_ray = {
        reconstructed_rgbd[4*ray_idx],
        reconstructed_rgbd[4*ray_idx+1],
        reconstructed_rgbd[4*ray_idx+2],
    };

    float depth_ray = reconstructed_rgbd[4*ray_idx+3];

    float depth_ray_target = ground_truth_rgbd[4*ray_idx+3];

    float depth_loss_gradient = depth_ray_target > 0.0f ? depth_supervision_lambda * losses_and_gradients[i*4+3] : 0;

    uint32_t total_n_rays = *super_ray_counter;
    uint32_t total_n_rays_depth = *super_ray_counter_depth;

    float loss_scale_rgb = loss_scale / (float)total_n_rays;
    float loss_scale_depth;
    if (total_n_rays_depth>0){
        loss_scale_depth = loss_scale / (float)total_n_rays_depth;
    } else {
        loss_scale_depth = 0.0;
    }

    // No regularization for pose optimization
    const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
    const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
    Array3f rgb_ray2 = { 0.f,0.f,0.f };
	float depth_ray2 = 0.f;
	float T = 1.f;

    uint32_t base = numsteps_in[ray_idx*2+1];
	uint32_t base_compact = numsteps_compacted[ray_idx*2+1];
	uint32_t numsteps_compact = numsteps_compacted[ray_idx*2];

    if (numsteps_compact==0) {
        return;
    }

    coords_out += base_compact;
	dloss_doutput += base_compact * padded_output_width;

	coords_in += base;
	network_output += base * padded_output_width;

	Eigen::Vector3f ray_o = rays_in_unnormalized[ray_idx].o;

    for (uint32_t j=0; j < numsteps_compact; ++j) {

        // Compact network inputs
	    NerfCoordinate* coord_out = coords_out(j);
	    const NerfCoordinate* coord_in = coords_in(j);
	    coord_out->copy(*coord_in, coords_out.stride_in_bytes);

	    const Vector3f pos = unwarp_position(coord_in->pos.p, aabb);
	    float depth = (pos - ray_o).norm();

	    float dt = unwarp_dt(coord_in->dt);
	    const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
	    const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
	    const float density = network_to_density(float(local_network_output[3]), density_activation);
	    const float alpha = 1.f - __expf(-density * dt);
	    const float weight = alpha * T;
	    rgb_ray2 += weight * rgb;
	    depth_ray2 += weight * depth;
	    T *= (1.f - alpha);

	    // we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
	    const Array3f suffix = rgb_ray - rgb_ray2;
	    const Array3f dloss_by_drgb = dloss_db;

	    tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;

	    // chain rule to go from dloss/drgb to dloss/dmlp_output
        local_dL_doutput[0] = (loss_scale_rgb/3.0) * (dloss_by_drgb.x() * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
	    local_dL_doutput[1] = (loss_scale_rgb/3.0) * (dloss_by_drgb.y() * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
	    local_dL_doutput[2] = (loss_scale_rgb/3.0) * (dloss_by_drgb.z() * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

	    float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
	    const float depth_suffix = depth_ray - depth_ray2;

        // if no target depth for that ray then no depth supervision
        float depth_supervision = 0.0f;
        if (depth_ray_target > 0.0f) {
	        depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);
        }

	    const Array3f tmp = dloss_db;
	    float dloss_by_dmlp = density_derivative * (
	    	dt * (loss_scale_rgb / 3.0 * tmp.matrix().dot((T * rgb - suffix).matrix()) + loss_scale_depth * depth_supervision)
	    );

	    local_dL_doutput[3] =
	    	dloss_by_dmlp +
	    	(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
	    	(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
	    	;

	    *(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

	    dloss_doutput += padded_output_width;
	    network_output += padded_output_width;
    }
}



__global__ void compute_camera_gradient_gp(
	const uint32_t n_rays,
	const BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	const TrainingXForm* training_xforms,
	Vector3f* cam_pos_gradient,
	Vector3f* cam_rot_gradient,
	const TrainingImageMetadata* __restrict__ metadata,
	const uint32_t* __restrict__ ray_indices,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_compacted,
	PitchedPtr<NerfCoordinate> coords,
	PitchedPtr<NerfCoordinate> coords_gradient,
	float* __restrict__ distortion_gradient,
	float* __restrict__ distortion_gradient_weight,
	const Vector2i distortion_resolution,
	Vector2f* cam_focal_length_gradient,
	const uint32_t indice_image_for_tracking_pose,
    const float* __restrict__ xy_image_pixel_indices,
    const uint32_t ray_stride,
    float* __restrict__ super_ray_gradients
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_compacted[i*2];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}

	uint32_t base = numsteps_compacted[i*2+1];
	coords += base;
	coords_gradient += base;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices[i];
    uint32_t img = indice_image_for_tracking_pose;


	Eigen::Vector2i resolution = metadata[img].resolution;

	const Matrix<float, 3, 4>& xform = training_xforms[img].start;

	Ray ray = rays_in_unnormalized[i];
	ray.d = ray.d.normalized();
	Ray ray_gradient = { Vector3f::Zero(), Vector3f::Zero() };

	// Compute ray gradient
	for (uint32_t j = 0; j < numsteps; ++j) {
		// pos = ray.o + t * ray.d;

		const Vector3f warped_pos = coords(j)->pos.p;
		Vector3f pos_gradient = coords_gradient(j)->pos.p.cwiseProduct(warp_position_derivative(warped_pos, aabb));
        // in case gradient are nan at this point
        if ( isnan(pos_gradient.x()) or isnan(pos_gradient.y()) or isnan(pos_gradient.z())
          or isinf(pos_gradient.x()) or isinf(pos_gradient.y()) or isinf(pos_gradient.z()) ) {
                pos_gradient.x() = 0.f;
                pos_gradient.y() = 0.f;
                pos_gradient.z() = 0.f;
        }

		ray_gradient.o += pos_gradient;

		const Vector3f pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = (pos - ray.o).norm();
		Vector3f dir_gradient = coords_gradient(j)->dir.d.cwiseProduct(warp_direction_derivative(coords(j)->dir.d));

        // in case gradient are nan at this point
        if ( isnan(dir_gradient.x()) or isnan(dir_gradient.y()) or isnan(dir_gradient.z())
          or isinf(dir_gradient.x()) or isinf(dir_gradient.y()) or isinf(dir_gradient.z()) ) {
                dir_gradient.x() = 0.f;
                dir_gradient.y() = 0.f;
                dir_gradient.z() = 0.f;
        }

		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

    uint32_t super_i = ray_idx / ray_stride;
    super_ray_gradients[super_i * 6 + 0] += ray_gradient.o.x();
    super_ray_gradients[super_i * 6 + 1] += ray_gradient.o.y();
    super_ray_gradients[super_i * 6 + 2] += ray_gradient.o.z();
    super_ray_gradients[super_i * 6 + 3] += ray_gradient.d.x();
    super_ray_gradients[super_i * 6 + 4] += ray_gradient.d.y();
    super_ray_gradients[super_i * 6 + 5] += ray_gradient.d.z();

	float xy_pdf = 1.0f;
    Vector2f xy = Vector2f(xy_image_pixel_indices[2*ray_idx], xy_image_pixel_indices[2*ray_idx+1]);

	if (distortion_gradient) {
		// Projection of the raydir gradient onto the plane normal to raydir,
		// because that's the only degree of motion that the raydir has.
		Vector3f orthogonal_ray_gradient = ray_gradient.d - ray.d * ray_gradient.d.dot(ray.d);

		// Rotate ray gradient to obtain image plane gradient.
		// This has the effect of projecting the (already projected) ray gradient from the
		// tangent plane of the sphere onto the image plane (which is correct!).
		Vector3f image_plane_gradient = xform.block<3,3>(0,0).inverse() * orthogonal_ray_gradient;

		// Splat the resulting 2D image plane gradient into the distortion params
		deposit_image_gradient<2>(image_plane_gradient.head<2>() / xy_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, xy);
	}

	if (cam_pos_gradient) {
		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / xy_pdf);
		}
	}

	if (cam_rot_gradient) {
		// Rotation is averaged in log-space (i.e. by averaging angle-axes).
		// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
		// orthogonal, leading to the angle_axis magnitude to equal the magnitude
		// of ray_gradient.d.
		Vector3f angle_axis = ray.d.cross(ray_gradient.d);

		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / xy_pdf);
		}
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
	const Vector2i& resolution,
    const Vector2i& resolution_at_level,
    const uint32_t& level
) {
    // get the at level sampling margins
    int margin_r = (int) ceil(((float) (sample_away_from_border_margin_h + (uint32_t) rf[1])) / pow(2.0, (float) level));
    int margin_c = (int) ceil(((float) (sample_away_from_border_margin_w + (uint32_t) rf[3])) / pow(2.0, (float) level));
    Vector2i margins_at_level(margin_c, margin_r);
    Vector2i bounds_at_level = resolution_at_level - 2*margins_at_level;

    //init vars
    Vector2f rng_xy;
    Vector2i xy_at_level_int;
    Vector2f xy_at_level;
    Vector2f xy;
    Vector2i half_kernel_size(rf[1], rf[3]);
    Vector2f half_kernel_size_float = half_kernel_size.cast<float>();
    Vector2i tmp_uv;
    Vector2f tmp_d;
    Vector2f tmp_xy_int;
    Vector2f tmp_xy;
    uint32_t key;

    uint32_t cpt=0;
    std::unordered_map<uint32_t, uint32_t> tmp_dict;
    while ( (ray_counter + ray_stride) < max_rays_per_batch) {

        ++super_ray_counter;

        // sample a pixel at level
        rng_xy.x() = rng.next_float();
        rng_xy.y() = rng.next_float();

        xy_at_level_int = rng_xy.cwiseProduct(bounds_at_level.cast<float>()).cast<int>();
        xy_at_level_int = xy_at_level_int + margins_at_level; // sampled pixel (int) at level

        xy_at_level_int = xy_at_level_int.cwiseMax(margins_at_level).cwiseMin(resolution_at_level - margins_at_level - Vector2i::Ones());

        xy_image_super_pixel_at_level_indices_int_cpu.push_back(xy_at_level_int.x());
        xy_image_super_pixel_at_level_indices_int_cpu.push_back(xy_at_level_int.y());

        xy_at_level = xy_at_level_int.cast<float>();

        // grab the corresponding pixels in OG rez
        xy = xy_at_level * pow(2.0, (float) level);

        // populate the output vectors
	    for (uint32_t u = 0; u < super_ray_window_size; ++u) {
	        for (uint32_t v = 0; v < super_ray_window_size; ++v) {

                ++ray_counter;

                tmp_uv = Vector2i(v,u);
                tmp_d = tmp_uv.cast<float>() - half_kernel_size_float;
                tmp_xy_int = xy + tmp_d;

	            if (snap_to_pixel_centers) {
	            	tmp_xy = (tmp_xy_int + Vector2f::Constant(0.5f)).cwiseQuotient(resolution.cast<float>());
	            } else {
	            	tmp_xy = tmp_xy_int.cwiseQuotient(resolution.cast<float>());
                }

                xy_image_pixel_indices.push_back(tmp_xy.x());
                xy_image_pixel_indices.push_back(tmp_xy.y());

                xy_image_pixel_indices_int.push_back(tmp_xy_int.x());
                xy_image_pixel_indices_int.push_back(tmp_xy_int.y());

                // check if ray exists
                key = tmp_xy_int.x() + tmp_xy_int.y() * resolution.x();
                if (tmp_dict.count(key) == 0) {
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


void Testbed::track_pose_gaussian_pyramid_nerf_slam(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {

	if (m_nerf.training.indice_image_for_tracking_pose == 0) {
        // no tracking for first frame.
		return;
	}

    m_nerf.training.counters_rgb_track.rays_per_batch = m_nerf.training.rays_per_tracking_batch;
	m_nerf.training.counters_rgb_track.prepare_for_training_steps(stream);

	if (m_nerf.training.n_steps_since_cam_update_tracking == 0) {
	    CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
	    CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
        m_nerf.training.tracking_gradients_super_rays.clear();
    }


	if (m_nerf.training.n_steps_since_photometric_correction_update==0 && 
	    m_nerf.training.train_with_photometric_corrections_in_tracking) {
		//NOTE: TODO: we don't need the entire array of gradients in tracking
		// We can only keep the one gradient of the currently tracked image.
		CUDA_CHECK_THROW(
			cudaMemsetAsync(
				m_nerf.training.image_photometric_correction_gradient_coef_gpu.data(), 
				0, 
				m_nerf.training.image_photometric_correction_gradient_coef_gpu.get_bytes(), 
				stream
			)
		);
		CUDA_CHECK_THROW(
			cudaMemsetAsync(
				m_nerf.training.image_photometric_correction_gradient_intercept_gpu.data(), 
				0, 
				m_nerf.training.image_photometric_correction_gradient_intercept_gpu.get_bytes(), 
				stream
			)
		);
		CUDA_CHECK_THROW(
			cudaMemsetAsync(
				m_nerf.training.image_photometric_correction_gradient_ray_count_gpu.data(), 
				0, 
				m_nerf.training.image_photometric_correction_gradient_ray_count_gpu.get_bytes(), 
				stream
			)
		);
	}

	track_pose_gaussian_pyramid_nerf_slam_step(target_batch_size, m_nerf.training.counters_rgb_track, stream);

    ++m_training_step_track;

    std::vector<float> losses_scalar = m_nerf.training.counters_rgb_track.update_after_training(target_batch_size, get_loss_scalar, stream, true, true);


    float loss_scalar = losses_scalar[0];
    float loss_depth_scalar = losses_scalar[1];
	bool zero_records = m_nerf.training.counters_rgb_track.measured_batch_size == 0;
	if (get_loss_scalar) {
        m_tracking_loss = loss_scalar;
        m_tracking_loss_depth = loss_depth_scalar;
		m_loss_scalar_track.update(loss_scalar);
	}

	if (zero_records) {
		m_loss_scalar_track.set(0.f);
		tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
		m_train = false;
	}

	m_nerf.training.n_steps_since_cam_update_tracking += 1;

	// Get extrinsics gradients

	if (m_nerf.training.n_steps_since_cam_update_tracking >= m_nerf.training.n_steps_between_cam_updates_tracking) {

		float per_camera_loss_scale = 1.0 / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates_tracking;

		{
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// Optimization step
            uint32_t i = m_nerf.training.indice_image_for_tracking_pose;
			Vector3f pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
			Vector3f rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

			float l2_reg = m_nerf.training.extrinsic_l2_reg;
			pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
			rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

            if (m_nerf.training.separate_pos_and_rot_lr) {
				m_nerf.training.cam_pos_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_pos);
				m_nerf.training.cam_rot_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_rot);
            } else {
				m_nerf.training.cam_pos_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate);
				m_nerf.training.cam_rot_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate);
            }

            if ( std::isnan(pos_gradient.x()) or std::isnan(pos_gradient.y()) or std::isnan(pos_gradient.z()) or std::isnan(rot_gradient.x()) or std::isnan(rot_gradient.y()) or std::isnan(rot_gradient.z()) ) {
                pos_gradient.x() = 0.f;
                pos_gradient.y() = 0.f;
                pos_gradient.z() = 0.f;
                rot_gradient.x() = 0.f;
                rot_gradient.y() = 0.f;
                rot_gradient.z() = 0.f;
            }

            float max_step_pos = 10;
            float max_step_rot = 10;
            pos_gradient.x() = max( -max_step_pos, min(pos_gradient.x(), max_step_pos) );
            pos_gradient.y() = max( -max_step_pos, min(pos_gradient.y(), max_step_pos) );
            pos_gradient.z() = max( -max_step_pos, min(pos_gradient.z(), max_step_pos) );

            rot_gradient.x() = max( -max_step_rot, min(rot_gradient.x(), max_step_rot) );
            rot_gradient.y() = max( -max_step_rot, min(rot_gradient.y(), max_step_rot) );
            rot_gradient.z() = max( -max_step_rot, min(rot_gradient.z(), max_step_rot) );

            float norm_pos = pos_gradient.norm();
            float norm_rot = rot_gradient.norm();

            m_tracking_pos_gradient_norm=norm_pos;
            m_tracking_rot_gradient_norm=norm_rot;

			m_nerf.training.cam_pos_offset[i].step(pos_gradient);
			m_nerf.training.cam_rot_offset[i].step(rot_gradient);

			m_nerf.training.update_transforms(i, i+1);
		}

        m_nerf.training.n_steps_since_cam_update_tracking = 0;
	}

	if (m_nerf.training.train_with_photometric_corrections_in_tracking) {
		
		m_nerf.training.n_steps_since_photometric_correction_update += 1;

		if (m_nerf.training.n_steps_since_photometric_correction_update >= m_nerf.training.n_steps_between_photometric_correction_updates) {
			// float per_camera_loss_scale = 10.f / LOSS_SCALE / (float)m_nerf.training.n_steps_between_confidence_scores_updates;
			// float per_camera_loss_scale = 1.0f / (float)m_nerf.training.n_steps_between_confidence_scores_updates;
			CUDA_CHECK_THROW(
				cudaMemcpyAsync(m_nerf.training.image_photometric_correction_gradient_coef.data(), 
							    m_nerf.training.image_photometric_correction_gradient_coef_gpu.data(), 
								m_nerf.training.image_photometric_correction_gradient_coef_gpu.get_bytes(), 
								cudaMemcpyDeviceToHost, stream)
			);
			CUDA_CHECK_THROW(
				cudaMemcpyAsync(m_nerf.training.image_photometric_correction_gradient_intercept.data(), 
							    m_nerf.training.image_photometric_correction_gradient_intercept_gpu.data(), 
								m_nerf.training.image_photometric_correction_gradient_intercept_gpu.get_bytes(), 
								cudaMemcpyDeviceToHost, stream)
			);
			CUDA_CHECK_THROW(
				cudaMemcpyAsync(m_nerf.training.image_photometric_correction_gradient_ray_count.data(), 
							    m_nerf.training.image_photometric_correction_gradient_ray_count_gpu.data(), 
								m_nerf.training.image_photometric_correction_gradient_ray_count_gpu.get_bytes(), 
								cudaMemcpyDeviceToHost, stream)
			);
			
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			float per_camera_loss_scale = 1.0 / LOSS_SCALE / (float)m_nerf.training.n_steps_between_photometric_correction_updates;

            uint32_t i = m_nerf.training.indice_image_for_tracking_pose;
			
			uint32_t ray_cpt = m_nerf.training.image_photometric_correction_gradient_ray_count[i];
			
			float grad_coef = m_nerf.training.image_photometric_correction_gradient_coef[i] * per_camera_loss_scale;
			float grad_intercept = m_nerf.training.image_photometric_correction_gradient_intercept[i] * per_camera_loss_scale;
			
			ArrayXf grad_coef_arr = ArrayXf::Constant(1, grad_coef);
			ArrayXf grad_intercept_arr = ArrayXf::Constant(1, grad_intercept);
			
			m_nerf.training.image_photometric_correction_variables_coef[i].set_learning_rate(m_nerf.training.image_photometric_correction_lr);
			m_nerf.training.image_photometric_correction_variables_intercept[i].set_learning_rate(m_nerf.training.image_photometric_correction_lr);
			
			m_nerf.training.image_photometric_correction_variables_coef[i].step(grad_coef_arr);
			m_nerf.training.image_photometric_correction_variables_intercept[i].step(grad_intercept_arr);
			
			m_nerf.training.image_photometric_correction_params_coef[i] = m_nerf.training.image_photometric_correction_variables_coef[i].variable()[0];
			m_nerf.training.image_photometric_correction_params_intercept[i] = m_nerf.training.image_photometric_correction_variables_intercept[i].variable()[0];
		
			CUDA_CHECK_THROW(
				cudaMemcpyAsync(m_nerf.training.image_photometric_correction_params_coef_gpu.data(), 
							    m_nerf.training.image_photometric_correction_params_coef.data(), 
								m_nerf.training.image_photometric_correction_params_coef_gpu.get_bytes(), 
								cudaMemcpyHostToDevice, stream)
			);
	
			CUDA_CHECK_THROW(
				cudaMemcpyAsync(m_nerf.training.image_photometric_correction_params_intercept_gpu.data(), 
							    m_nerf.training.image_photometric_correction_params_intercept.data(), 
								m_nerf.training.image_photometric_correction_params_intercept_gpu.get_bytes(), 
								cudaMemcpyHostToDevice, stream)
			);

			m_nerf.training.n_steps_since_photometric_correction_update = 0;

		}
	}
}


void Testbed::track_pose_gaussian_pyramid_nerf_slam_step(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

    //NOTE: get settings/hyperparams for tracking
    const uint32_t kernel_window_size = 5;
    const uint32_t gaussian_pyramid_level = m_tracking_gaussian_pyramid_level;
    const uint32_t sample_away_from_border_margin_h = m_sample_away_from_border_margin_h;
    const uint32_t sample_away_from_border_margin_w = m_sample_away_from_border_margin_w;

    //NOTE: get downsized image dimensions
	Vector2i resolution = m_nerf.training.dataset.metadata[m_nerf.training.indice_image_for_tracking_pose].resolution;
    Vector2i resolution_at_level = (resolution.cast<float>() / pow(2.0, (float) gaussian_pyramid_level)).cast<int>();

    //NOTE: get receptive field at level L
    std::vector<int> receptive_field;
    get_receptive_field_of_gaussian_pyramid_at_level(gaussian_pyramid_level, receptive_field);

    const uint32_t super_ray_window_size = receptive_field[1] * 2 + 1;
    const uint32_t ray_stride = super_ray_window_size * super_ray_window_size;
    uint32_t n_super_rays=0;
    uint32_t n_total_rays=0; // get all rays needed to compute super rays values (pixels at level)
    uint32_t n_total_rays_for_gradient=0; //  subset of unique rays.
    std::vector<float> xy_image_pixel_indices_cpu;
    std::vector<uint32_t> xy_image_super_pixel_at_level_indices_int_cpu;
    std::vector<uint32_t> xy_image_pixel_indices_int_cpu;
    std::vector<uint32_t> existing_ray_mapping;
    sample_pixels_for_tracking_with_gaussian_pyramid(
        counters.rays_per_batch,
        n_total_rays,
        n_super_rays,
        n_total_rays_for_gradient,
        xy_image_pixel_indices_cpu,
        xy_image_pixel_indices_int_cpu,
        xy_image_super_pixel_at_level_indices_int_cpu,
        existing_ray_mapping,
		m_nerf.training.snap_to_pixel_centers,
        sample_away_from_border_margin_h,
        sample_away_from_border_margin_w,
        m_rng,
        receptive_field,
        super_ray_window_size,
        ray_stride,
	    resolution,
        resolution_at_level,
        gaussian_pyramid_level
    );

    m_nerf.training.sampled_pixels_for_tracking=xy_image_pixel_indices_int_cpu;
    m_nerf.training.sampled_pixels_for_tracking_at_level=xy_image_super_pixel_at_level_indices_int_cpu;

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		uint32_t, // ray_indices
		Ray, // rays
		uint32_t, // numsteps
		float, // coords
		float, // max_level
		network_precision_t, // mlp_out
		network_precision_t, // dloss_dmlp_out
		float, // coords_compacted
		float, // coords_gradient
		float, // max_level_compacted
		uint32_t, // ray_counter
		float, // xy_pixel_indices
		int32_t, // mapping_indices
		uint32_t, // numsteps_compacted
		uint32_t, // num super rays counter
		uint32_t, // num super rays counter depth
		uint32_t, // mapping of existing rays
		float  // super ray gradients
	>(
		stream, &alloc,
		n_total_rays_for_gradient,
		n_total_rays_for_gradient,
		n_total_rays_for_gradient * 2,
		max_samples * floats_per_coord,
		max_samples,
		std::max(target_batch_size, max_samples) * padded_output_width,
		target_batch_size * padded_output_width,
		target_batch_size * floats_per_coord,
		target_batch_size * floats_per_coord,
		target_batch_size,
		1,
		n_total_rays * 2,
		n_total_rays, // 12
		n_total_rays_for_gradient * 2,
        1,
        1,
		n_total_rays,
		n_super_rays * 6
	);

	// NOTE: C++17 structured binding
	uint32_t* ray_indices = std::get<0>(scratch);
	Ray* rays_unnormalized = std::get<1>(scratch);
	uint32_t* numsteps = std::get<2>(scratch);
	float* coords = std::get<3>(scratch);
	float* max_level = std::get<4>(scratch);
	network_precision_t* mlp_out = std::get<5>(scratch);
	network_precision_t* dloss_dmlp_out = std::get<6>(scratch);
	float* coords_compacted = std::get<7>(scratch);
	float* coords_gradient = std::get<8>(scratch);
	float* max_level_compacted = std::get<9>(scratch);
	uint32_t* ray_counter = std::get<10>(scratch);
	float* xy_image_pixel_indices = std::get<11>(scratch);
	int32_t* mapping_indices = std::get<12>(scratch);
	uint32_t* numsteps_compacted = std::get<13>(scratch);
	uint32_t* super_ray_counter = std::get<14>(scratch);
	uint32_t* super_ray_counter_depth = std::get<15>(scratch);
    uint32_t* existing_ray_mapping_gpu = std::get<16>(scratch);
	float* super_ray_gradients = std::get<17>(scratch);

	uint32_t max_inference = next_multiple(std::min(n_total_rays_for_gradient * 1024, max_samples), tcnn::batch_size_granularity);
    counters.measured_batch_size_before_compaction = max_inference;

	GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
	GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_training_step_track == 0) {
		counters.n_rays_total = 0;
	}

	counters.n_rays_total += n_total_rays;

    m_track_pose_nerf_num_super_rays_targeted_in_tracking_step=n_super_rays;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter_depth, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_gradients, 0, n_super_rays * 6 * sizeof(uint32_t), stream));

    // create 5-tap kernel
    // TODO: move that outside of tracking loop
    // make kernel __constant__ global var
    // see http://alexminnaar.com/2019/07/12/implementing-convolutions-in-cuda.html
    std::vector<float> kernel = make_5tap_kernel();
    tcnn::GPUMemory<float> kernel_gpu;
    kernel_gpu.enlarge(kernel_window_size * kernel_window_size);
    CUDA_CHECK_THROW(
       cudaMemcpy(
          kernel_gpu.data(),
          kernel.data(),
          kernel_window_size * kernel_window_size * sizeof(float),
          cudaMemcpyHostToDevice
       )
    );

    //NOTE: ship to Device sampled pixels
    CUDA_CHECK_THROW(
       cudaMemcpy(
          std::get<11>(scratch),
          xy_image_pixel_indices_cpu.data(),
          n_total_rays * 2 * sizeof(float),
          cudaMemcpyHostToDevice
       )
    );
    CUDA_CHECK_THROW(
       cudaMemcpy(
          std::get<16>(scratch),
          existing_ray_mapping.data(),
          n_total_rays * sizeof(uint32_t),
          cudaMemcpyHostToDevice
       )
    );

    //NOTE: get sample along each rays
	linear_kernel(generate_training_samples_for_tracking_gp, 0, stream,
        n_total_rays,
		m_aabb,
		max_inference,
		m_rng,
		ray_counter,
		counters.numsteps_counter.data(),
		ray_indices,
		rays_unnormalized,
		numsteps,
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
        m_nerf.training.dataset.metadata_gpu.data(),
		m_nerf.training.transforms_gpu.data(),
		m_nerf.density_grid_bitfield.data(),
		m_nerf.cone_angle_constant,
		m_distortion.map->params(),
		m_distortion.resolution,
		m_nerf.training.extra_dims_gpu.data(),
		m_nerf_network->n_extra_dims(),
        m_nerf.training.indice_image_for_tracking_pose,
		mapping_indices,
        existing_ray_mapping_gpu,
        xy_image_pixel_indices,
		m_nerf.training.use_view_dir_in_nerf
	);

    //DEBUG: check the sampled pixels
    m_nerf.training.sampled_ray_indices_for_tracking_gradient.resize(n_total_rays_for_gradient);
    CUDA_CHECK_THROW(
       cudaMemcpy(
          m_nerf.training.sampled_ray_indices_for_tracking_gradient.data(),
          std::get<0>(scratch),
          n_total_rays_for_gradient * sizeof(uint32_t),
          cudaMemcpyDeviceToHost
       )
    );
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(
          &m_track_pose_nerf_num_rays_in_tracking_step,
          std::get<10>(scratch),
          sizeof(uint32_t),
          cudaMemcpyDeviceToHost,
          stream
       )
    );

    if (m_track_pose_nerf_num_rays_in_tracking_step != n_total_rays_for_gradient) {
        tlog::warning()<<" num rays for gradient is different than the required num of rays: "<<m_track_pose_nerf_num_rays_in_tracking_step<< " != "<< n_total_rays_for_gradient << ". Consider increasing batch_size";
    }

    //NOTE: get network values for each points
	m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);


    tcnn::GPUMemory<float> reconstructed_rgbd_gpu;
    tcnn::GPUMemory<float> ground_truth_rgbd_gpu;
    reconstructed_rgbd_gpu.enlarge(n_total_rays_for_gradient * 4);
    ground_truth_rgbd_gpu.enlarge(n_total_rays_for_gradient * 4);

    //NOTE: get RGBD values prediciton + GT.
	linear_kernel(compute_GT_and_reconstructed_rgbd_gp, 0, stream,
        n_total_rays_for_gradient,
		m_aabb,
		m_rng,
		target_batch_size,
		ray_counter,
		padded_output_width,
		m_envmap.envmap->params(),
		m_envmap.resolution,
		m_background_color.head<3>(),
		m_color_space,
		m_nerf.training.random_bg_color,
		m_nerf.training.linear_colors,
		m_nerf.training.dataset.metadata_gpu.data(),
		mlp_out,
		counters.numsteps_counter_compacted.data(),
		ray_indices,
		rays_unnormalized,
		numsteps,
		numsteps_compacted,
		PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
		m_nerf.rgb_activation,
		m_nerf.density_activation,
		m_nerf.density_grid.data(),
		m_nerf.density_grid_mean.data(),
		m_nerf.training.cam_exposure_gpu.data(),
		m_nerf.training.depth_supervision_lambda,
        m_nerf.training.indice_image_for_tracking_pose,
        xy_image_pixel_indices,
        ground_truth_rgbd_gpu.data(),
        reconstructed_rgbd_gpu.data()
	);

	//NOTE: Using photometric corrections
	// compute corrected GT RGB
	if (m_nerf.training.train_with_photometric_corrections_in_tracking) {
		linear_kernel(apply_photometric_correction_to_GT, 0, stream,
        	n_total_rays_for_gradient,
			ray_counter,
        	m_nerf.training.indice_image_for_tracking_pose,
			m_nerf.training.image_photometric_correction_params_coef_gpu.data(),
			m_nerf.training.image_photometric_correction_params_intercept_gpu.data(),
        	ground_truth_rgbd_gpu.data()
		);
	}

    //NOTE: get Depth reconstruction variance (ie confidence)
    tcnn::GPUMemory<float> reconstructed_depth_var_gpu;
    tcnn::GPUMemory<float> reconstructed_color_var_gpu;
    if (m_tracking_use_depth_var_in_loss) {
        reconstructed_depth_var_gpu.enlarge(n_total_rays_for_gradient);
        reconstructed_color_var_gpu.enlarge(n_total_rays_for_gradient);
	    linear_kernel(compute_depth_variance_gp, 0, stream,
            n_total_rays_for_gradient,
	    	m_aabb,
	    	ray_counter,
	    	padded_output_width,
	    	mlp_out,
	    	rays_unnormalized,
	    	numsteps,
	    	PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		    m_nerf.rgb_activation,
	    	m_nerf.density_activation,
            reconstructed_rgbd_gpu.data(),
            reconstructed_depth_var_gpu.data(),
            reconstructed_color_var_gpu.data()
	    );
        m_tracking_reconstructed_depth_var.resize(n_total_rays_for_gradient);
        reconstructed_depth_var_gpu.copy_to_host(m_tracking_reconstructed_depth_var.data(), n_total_rays_for_gradient);
        m_tracking_reconstructed_color_var.resize(n_total_rays_for_gradient);
        reconstructed_color_var_gpu.copy_to_host(m_tracking_reconstructed_color_var.data(), n_total_rays_for_gradient);
    }


    //NOTE: compute Gaussian pyramids (ie convs)
    std::vector<tcnn::GPUMemory<float> > ground_truth_rgbd_tensors;
    std::vector<tcnn::GPUMemory<float> > reconstructed_rgbd_tensors;
    std::vector<tcnn::GPUMemory<float> > reconstructed_depth_var_tensors;
    std::vector<tcnn::GPUMemory<float> > reconstructed_color_var_tensors;
    std::vector<tcnn::GPUMemory<float> > gradients_tensors;
    std::vector<uint32_t> dimensions;

    uint32_t cur_super_ray_window_size = super_ray_window_size;
    uint32_t cur_ray_stride = ray_stride;
    uint32_t cur_dim = n_total_rays;

    ground_truth_rgbd_tensors.push_back(ground_truth_rgbd_gpu);
    reconstructed_rgbd_tensors.push_back(reconstructed_rgbd_gpu);
    dimensions.push_back(cur_dim);
    if (m_tracking_use_depth_var_in_loss) {
        reconstructed_depth_var_tensors.push_back(reconstructed_depth_var_gpu);
        reconstructed_color_var_tensors.push_back(reconstructed_color_var_gpu);
    }

    for (size_t l=0; l<gaussian_pyramid_level; ++l) {

        std::vector<int> tmp_receptive_field;
        get_receptive_field_of_gaussian_pyramid_at_level(gaussian_pyramid_level-l-1, tmp_receptive_field);
        uint32_t tmp_super_ray_window_size = tmp_receptive_field[1] * 2 + 1;
        uint32_t tmp_ray_stride = tmp_super_ray_window_size * tmp_super_ray_window_size;
        uint32_t tmp_dim = n_super_rays * tmp_ray_stride;

        tcnn::GPUMemory<float> new_gt_rgbd;
        tcnn::GPUMemory<float> new_rec_rgbd;
        tcnn::GPUMemory<float> gradients;

        new_gt_rgbd.enlarge(tmp_dim*4);
        new_rec_rgbd.enlarge(tmp_dim*4);
        gradients.enlarge(tmp_dim * cur_dim); // L1 x L2 matrix
        gradients.memset(0);

        tcnn::GPUMemory<float> new_rec_depth_var;
        tcnn::GPUMemory<float> new_rec_color_var;
        if (m_tracking_use_depth_var_in_loss) {
            new_rec_depth_var.enlarge(tmp_dim);
            new_rec_color_var.enlarge(tmp_dim);
        }

	    linear_kernel(convolution_gaussian_pyramid, 0, stream,
            tmp_dim,
            cur_dim,
            cur_super_ray_window_size,
            tmp_super_ray_window_size,
            cur_ray_stride,
            tmp_ray_stride,
            kernel_gpu.data(),
            l==0 ? existing_ray_mapping_gpu : nullptr,
            l==0 ? mapping_indices : nullptr,
            ground_truth_rgbd_tensors.back().data(),
            reconstructed_rgbd_tensors.back().data(),
            m_tracking_use_depth_var_in_loss ? reconstructed_depth_var_tensors.back().data() : nullptr,
            m_tracking_use_depth_var_in_loss ? reconstructed_color_var_tensors.back().data() : nullptr,
            m_tracking_use_depth_var_in_loss ? new_rec_depth_var.data() : nullptr,
            m_tracking_use_depth_var_in_loss ? new_rec_color_var.data() : nullptr,
            new_gt_rgbd.data(),
            new_rec_rgbd.data(),
            gradients.data()
	    );

        ground_truth_rgbd_tensors.push_back(new_gt_rgbd);
        reconstructed_rgbd_tensors.push_back(new_rec_rgbd);
        gradients_tensors.push_back(gradients);
        dimensions.push_back(tmp_dim);

        if (m_tracking_use_depth_var_in_loss) {
            reconstructed_depth_var_tensors.push_back(new_rec_depth_var);
            reconstructed_color_var_tensors.push_back(new_rec_color_var);
        }

        cur_super_ray_window_size = tmp_super_ray_window_size;
        cur_ray_stride = tmp_ray_stride;
        cur_dim = tmp_dim;

    }

    if (m_tracking_use_depth_var_in_loss) {
        if (gaussian_pyramid_level>0) {
            m_tracking_reconstructed_depth_var_at_level.resize(n_super_rays);
            m_tracking_reconstructed_color_var_at_level.resize(n_super_rays);
            reconstructed_depth_var_tensors.back().copy_to_host(m_tracking_reconstructed_depth_var_at_level.data(), n_super_rays);
            reconstructed_color_var_tensors.back().copy_to_host(m_tracking_reconstructed_color_var_at_level.data(), n_super_rays);
        } else {
            m_tracking_reconstructed_depth_var_at_level.resize(n_total_rays_for_gradient);
            m_tracking_reconstructed_color_var_at_level.resize(n_total_rays_for_gradient);
            reconstructed_depth_var_tensors.back().copy_to_host(m_tracking_reconstructed_depth_var_at_level.data(), n_total_rays_for_gradient);
            reconstructed_color_var_tensors.back().copy_to_host(m_tracking_reconstructed_color_var_at_level.data(), n_total_rays_for_gradient);
        }
    }


    tcnn::GPUMemory<float> dL_dB;
    dL_dB.enlarge(n_super_rays * 4);

    //NOTE: compute loss.
	linear_kernel(compute_loss_gp, 0, stream,
		n_super_rays,
		m_nerf.training.track_loss_type,
		m_nerf.training.track_depth_loss_type,
		counters.loss.data(),
		counters.loss_depth.data(),
        m_tracking_use_depth_var_in_loss,
		m_tracking_use_color_var_in_loss,
        gaussian_pyramid_level==0 ? existing_ray_mapping_gpu : nullptr,
        gaussian_pyramid_level==0 ? mapping_indices : nullptr,
        ground_truth_rgbd_tensors.back().data(),
        reconstructed_rgbd_tensors.back().data(),
        m_tracking_use_depth_var_in_loss ? reconstructed_depth_var_tensors.back().data(): nullptr,
        m_tracking_use_depth_var_in_loss ? reconstructed_color_var_tensors.back().data(): nullptr,
        dL_dB.data(),
		super_ray_counter,
		super_ray_counter_depth
	);

    // NOTE: store Losses on host
    counters.loss_cpu.resize(n_super_rays);
    counters.loss_depth_cpu.resize(n_super_rays);
    counters.loss.copy_to_host(counters.loss_cpu.data(), n_super_rays);
    counters.loss_depth.copy_to_host(counters.loss_depth_cpu.data(), n_super_rays);

    //NOTE: Backprop thru convs
    std::vector<tcnn::GPUMemory<float> > partial_derivatives;
    partial_derivatives.push_back(dL_dB);

    for (size_t l=0; l<gaussian_pyramid_level; ++l) {

        uint32_t cur_dim = dimensions.back();
        dimensions.pop_back();
        uint32_t tmp_dim = dimensions.back();

        tcnn::GPUMemory<float> tmp_dL_dB;
        tmp_dL_dB.enlarge(tmp_dim * 4);

        linear_kernel(backprop_thru_convs, 0, stream,
            tmp_dim,
            cur_dim,
            partial_derivatives.back().data(),
            gradients_tensors.back().data(),
            tmp_dL_dB.data()
        );

        partial_derivatives.push_back(tmp_dL_dB);
        gradients_tensors.pop_back();
    }


	//NOTE: Using photometric corrections
	// compute gradients wrt photometric parameters
	// update latest partial derivative dL_dB
	if (m_nerf.training.train_with_photometric_corrections_in_tracking) {
		linear_kernel(compute_gradients_wrt_photometric_params_and_update_partial_derivatives, 0, stream,
        	1,
			n_total_rays,
            existing_ray_mapping_gpu,
            mapping_indices,
        	m_nerf.training.indice_image_for_tracking_pose,
			m_nerf.training.image_photometric_correction_params_coef_gpu.data(),
			m_nerf.training.image_photometric_correction_params_intercept_gpu.data(),
        	ground_truth_rgbd_gpu.data(),
			partial_derivatives.back().data(),
			m_nerf.training.image_photometric_correction_gradient_coef_gpu.data(), 
			m_nerf.training.image_photometric_correction_gradient_intercept_gpu.data()
		);
	}


	CUDA_CHECK_THROW(
       cudaMemcpyAsync(
          &counters.super_rays_counter,
          std::get<14>(scratch),
          sizeof(uint32_t),
          cudaMemcpyDeviceToHost,
          stream
       )
    );

	CUDA_CHECK_THROW(
       cudaMemcpyAsync(
          &counters.super_rays_counter_depth,
          std::get<15>(scratch),
          sizeof(uint32_t),
          cudaMemcpyDeviceToHost,
          stream
       )
    );

    //NOTE: compute loss and gradients.
	linear_kernel(compute_gradient_gp, 0, stream,
        n_total_rays,
		m_aabb,
		LOSS_SCALE,
		padded_output_width,
		m_nerf.training.dataset.metadata_gpu.data(),
		mlp_out,
		ray_indices,
		rays_unnormalized,
		numsteps,
        numsteps_compacted,
		PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
		dloss_dmlp_out,
		m_nerf.training.track_loss_type,
		m_nerf.training.track_depth_loss_type,
		counters.loss.data(),
		counters.loss_depth.data(),
		m_nerf.rgb_activation,
		m_nerf.density_activation,
		m_nerf.density_grid.data(),
		m_nerf.density_grid_mean.data(),
		m_nerf.training.depth_supervision_lambda,
		m_nerf.training.near_distance,
        xy_image_pixel_indices,
		mapping_indices,
        ground_truth_rgbd_gpu.data(),
        reconstructed_rgbd_gpu.data(),
		ray_counter,
        existing_ray_mapping_gpu,
        partial_derivatives.back().data(),
		super_ray_counter,
		super_ray_counter_depth
	);

	fill_rollover_and_rescale<network_precision_t><<<n_blocks_linear(target_batch_size*padded_output_width), n_threads_linear, 0, stream>>>(
		target_batch_size, padded_output_width, counters.numsteps_counter_compacted.data(), dloss_dmlp_out
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size * floats_per_coord), n_threads_linear, 0, stream>>>(
		target_batch_size, floats_per_coord, counters.numsteps_counter_compacted.data(), (float*)coords_compacted
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, counters.numsteps_counter_compacted.data(), max_level_compacted
	);

	bool train_camera = true;
	bool prepare_input_gradients = train_camera;
	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);

	{
		auto ctx = m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, prepare_input_gradients);
		m_network->backward(stream, *ctx, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, EGradientMode::Overwrite);
	}

	// Compute camera gradients
	linear_kernel(compute_camera_gradient_gp, 0, stream,
		counters.rays_per_batch,
		m_aabb,
		ray_counter,
		m_nerf.training.transforms_gpu.data(),
		m_nerf.training.cam_pos_gradient_gpu.data(),
		m_nerf.training.cam_rot_gradient_gpu.data(),
		m_nerf.training.dataset.metadata_gpu.data(),
		ray_indices,
		rays_unnormalized,
		numsteps_compacted,
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
		m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
		m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
		m_distortion.resolution,
		m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
        m_nerf.training.indice_image_for_tracking_pose,
        xy_image_pixel_indices,
        ray_stride,
	    super_ray_gradients
	);

    uint32_t prev_size = m_nerf.training.tracking_gradients_super_rays.size();
    uint32_t new_size = prev_size + n_super_rays * 6;
    m_nerf.training.tracking_gradients_super_rays.resize(new_size);

    CUDA_CHECK_THROW(
       cudaMemcpy(
          &m_nerf.training.tracking_gradients_super_rays.data()[prev_size],
          std::get<17>(scratch),
          n_super_rays * 6 * sizeof(float),
          cudaMemcpyDeviceToHost
       )
    );

	m_rng.advance();

}


NGP_NAMESPACE_END
