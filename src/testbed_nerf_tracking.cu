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


std::vector<float> Testbed::make_gaussian_kernel_debug(const uint32_t kernel_size, const float sigma) {

    std::vector<float> kernel(kernel_size * kernel_size);

    if (kernel_size==1) {
        kernel[0] = 1.f;
        return kernel;
    }

    uint32_t hw = kernel_size / 2;

    // const double pi = 3.14159265358979323846;
    uint32_t cpt=0;
    float norm = 0.0f;
    for (uint32_t i=0; i < kernel_size; i++){
        for (uint32_t j=0; j < kernel_size; j++){

            // float g = 1/(2*pi*sigma*sigma) * std::exp( -( static_cast<float>( (i-hw)*(i-hw) + (j-hw)*(j-hw) ) ) / static_cast<float>(2*sigma*sigma) );
            float g =  std::exp( -( static_cast<float>( (i-hw)*(i-hw) + (j-hw)*(j-hw) ) ) / static_cast<float>(2*sigma*sigma) );

            kernel[cpt] = g;
            norm += g;
            ++cpt;
        }
    }

    cpt=0;
    for (uint32_t i=0; i < kernel_size; i++){
        for (uint32_t j=0; j < kernel_size; j++){
            kernel[cpt] /= norm;
            ++cpt;
        }
    }

    return kernel;
}

__global__ void sample_training_pixels_for_tracking(
	const uint32_t n_rays,
    const uint32_t ray_stride,
	default_rng_t rng,
    const uint32_t kernel_window_size,
    const uint32_t sample_away_from_border_margin_h,
    const uint32_t sample_away_from_border_margin_w,
	const TrainingImageMetadata* __restrict__ metadata,
	const uint32_t indice_image_for_tracking_pose,
	bool snap_to_pixel_centers,
    float* __restrict__ xy_image_pixel_indices,
    uint32_t* __restrict__ xy_image_pixel_indices_int
) {
	const uint32_t super_i = threadIdx.x + blockIdx.x * blockDim.x;
	if (super_i >= n_rays) return;

    uint32_t img = indice_image_for_tracking_pose;
    uint32_t half_kernel_window_size = kernel_window_size / 2;

	Eigen::Vector2i resolution = metadata[img].resolution;
	Eigen::Vector2i margins = Eigen::Vector2i(sample_away_from_border_margin_w, sample_away_from_border_margin_h);
	Eigen::Vector2i half_kernel_size = Eigen::Vector2i(half_kernel_window_size, half_kernel_window_size);

	rng.advance(super_i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	Vector2f xy = nerf_random_image_pos_for_tracking(rng, resolution, snap_to_pixel_centers, margins, half_kernel_size);

    uint32_t base_i = super_i * ray_stride * 2;
    uint32_t cpt=0;
    Vector2f resolution_float = resolution.cast<float>();
    Vector2f half_kernel_size_float = half_kernel_size.cast<float>();
    // populate xy_image_pixel_indices with nearby indices  (within window)
	for (uint32_t u = 0; u < kernel_window_size; ++u) {
	    for (uint32_t v = 0; v < kernel_window_size; ++v) {

            Vector2i tmp_uv = Vector2i(v,u);
            Vector2f tmp_d = (tmp_uv.cast<float>() - half_kernel_size_float).cwiseQuotient(resolution_float);
            Vector2f tmp_xy = xy + tmp_d;
            xy_image_pixel_indices[base_i + cpt] = tmp_xy.x();
            xy_image_pixel_indices[base_i + cpt + 1] = tmp_xy.y();

            Vector2i pixel_indices_int = image_pos(tmp_xy, resolution);
            xy_image_pixel_indices_int[base_i + cpt] = pixel_indices_int.x();
            xy_image_pixel_indices_int[base_i + cpt + 1] = pixel_indices_int.y();

            cpt+=2;
        }
    }
}

__global__ void generate_training_samples_for_tracking(
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
    const float* __restrict__ xy_image_pixel_indices
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

    uint32_t img = indice_image_for_tracking_pose;

	Eigen::Vector2i resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
    Vector2f xy = Vector2f(xy_image_pixel_indices[2*i], xy_image_pixel_indices[2*i+1]);

	mapping_indices[i] = -1; //  default to not existing

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

	Vector3f warped_dir = warp_direction(ray_d_normalized);
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


__global__ void compute_depth_variance(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t* __restrict__ ray_counter,
	int padded_output_width,
	const tcnn::network_precision_t* network_output,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	ENerfActivation density_activation,
	const float* __restrict__ reconstructed_rgbd,
	float* __restrict__ reconstructed_depth_var
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *ray_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

    const float rec_depth = reconstructed_rgbd[i*4+3];

	float T = 1.f;

	float EPSILON = 1e-4f;

    float depth_var = 0.f;

    Eigen::Vector3f ray_o = rays_in_unnormalized[i].o;

	for (uint32_t j=0; j < numsteps; ++j) {
		if (T < EPSILON) {
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		float density = network_to_density(float(local_network_output[3]), density_activation);

        const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = (pos - ray_o).norm();

        float tmp = (cur_depth - rec_depth) * (cur_depth - rec_depth);

		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;

        depth_var += weight * tmp;
		T *= (1.f - alpha);

		network_output += padded_output_width;
		coords_in += 1;
	}

    reconstructed_depth_var[i] = depth_var;
}




__global__ void compute_GT_and_reconstructed_rgbd(
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



__global__ void compute_loss(
	const uint32_t n_rays,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	float* __restrict__ loss_depth_output,
    const uint32_t ray_stride,
	const uint32_t kernel_window_size,
	const int32_t* __restrict__ mapping_indices,
	const float* __restrict__ ground_truth_rgbd,
	const float* __restrict__ reconstructed_rgbd,
	const float* __restrict__ reconstructed_depth_var,
    const float* __restrict__ kernel,
	float* __restrict__ losses_and_gradients,
	uint32_t* __restrict__ super_ray_counter,
	uint32_t* __restrict__ super_ray_counter_depth
) {

	const uint32_t super_i = threadIdx.x + blockIdx.x * blockDim.x;
	if (super_i >= n_rays) { return; }

    // avg values within window
	float avg_depth_ray = 0.f;
	Array3f avg_rgb_ray = Array3f::Zero();
	float avg_depth_ray_var = 0.f;

    float avg_depth_ray_target = 0.f;
	Array3f avg_rgb_ray_target = Array3f::Zero();

    float norm = 0.f;
    float norm_depth = 0.f;

    uint32_t i;
    uint32_t base_i = super_i*ray_stride;

    uint32_t cpt=0;
    bool is_there_at_least_one_ray_in_super_ray=false;
    bool is_there_at_least_one_depth_ray_in_super_ray=false;
    for (uint32_t u = 0; u < kernel_window_size; ++u) {
	    for (uint32_t v = 0; v < kernel_window_size; ++v) {

            i = base_i + cpt;

            int32_t ray_idx = mapping_indices[i];

            if (ray_idx < 0){
                //NOTE: if a ray is missing we can also discard the super ray.
                // ie break istead of continue
                continue;
                cpt++;
            }

            is_there_at_least_one_ray_in_super_ray=true;

	        Array3f rgb_ray = {
                reconstructed_rgbd[4*ray_idx],
                reconstructed_rgbd[4*ray_idx+1],
                reconstructed_rgbd[4*ray_idx+2],
            };


            avg_rgb_ray += kernel[cpt] * rgb_ray;

	        Array3f rgb_ray_target = {
                ground_truth_rgbd[4*ray_idx],
                ground_truth_rgbd[4*ray_idx+1],
                ground_truth_rgbd[4*ray_idx+2],
            };

            avg_rgb_ray_target += kernel[cpt] * rgb_ray_target;

            norm += kernel[cpt];

            float depth_ray = reconstructed_rgbd[4*ray_idx+3];
            float depth_ray_var = reconstructed_depth_var[ray_idx];

            float depth_ray_target = ground_truth_rgbd[4*ray_idx+3];
            // handle cases where depth is 0.0 or -1.0
            if (depth_ray_target > 0.0) {
                avg_depth_ray += kernel[cpt] * depth_ray;
                avg_depth_ray_target += kernel[cpt] * depth_ray_target;
                norm_depth += kernel[cpt];
	            avg_depth_ray_var += kernel[cpt] * depth_ray_var;
                is_there_at_least_one_depth_ray_in_super_ray=true;
            }

            cpt++;
        }
    }

    losses_and_gradients[super_i*10+0] = 0.0f;
    losses_and_gradients[super_i*10+1] = 0.0f;
    losses_and_gradients[super_i*10+2] = 0.0f;
    losses_and_gradients[super_i*10+3] = 0.0f;
    losses_and_gradients[super_i*10+4] = 0.0f;
    losses_and_gradients[super_i*10+5] = 0.0f;
    losses_and_gradients[super_i*10+6] = 0.0f;
    losses_and_gradients[super_i*10+7] = 0.0f;
    losses_and_gradients[super_i*10+8] = -1;
    losses_and_gradients[super_i*10+9] = -1;

    //If all rays in super ray have 0 numsteps
    if (!is_there_at_least_one_ray_in_super_ray){
        return;
    }

	uint32_t num_super_rays_in_loss = atomicAdd(super_ray_counter, 1);	 // first entry in the array is a counter

    avg_rgb_ray /= norm;
    avg_rgb_ray_target /= norm;

    if (is_there_at_least_one_depth_ray_in_super_ray){
        avg_depth_ray /= norm_depth;
        avg_depth_ray_target /= norm_depth;
        avg_depth_ray_var /= norm_depth;
	    uint32_t num_super_rays_depth_in_loss = atomicAdd(super_ray_counter_depth, 1);	 // first entry in the array is a counter
    } else {
	    avg_depth_ray = 0.f;
        avg_depth_ray_target = 0.f;
	    avg_depth_ray_var = 0.f;
    }


	// Step again, this time computing loss
    LossAndGradient lg = loss_and_gradient(avg_rgb_ray_target, avg_rgb_ray, loss_type);
	LossAndGradient lg_depth = loss_and_gradient(Array3f::Constant(avg_depth_ray_target), Array3f::Constant(avg_depth_ray), depth_loss_type);

    // if ((lg_depth.loss.x()==0) or (norm_depth<=1e-4f)) {
    if ((lg_depth.loss.x()==0.f)) {
        lg_depth.gradient.x() = 0.f;
    }

    if (is_there_at_least_one_depth_ray_in_super_ray) {
        if (avg_depth_ray_var > 1e-6) {
            lg_depth.loss.x() /= sqrt(avg_depth_ray_var);
            lg_depth.gradient.x() /= sqrt(avg_depth_ray_var);
        } else {
            lg_depth.loss.x() /= sqrt(1e-6);
            lg_depth.gradient.x() /= sqrt(1e-6);
        }
    }


    losses_and_gradients[super_i*10+0] = lg.loss.x();
    losses_and_gradients[super_i*10+1] = lg.loss.y();
    losses_and_gradients[super_i*10+2] = lg.loss.z();
    losses_and_gradients[super_i*10+3] = lg.gradient.x();
    losses_and_gradients[super_i*10+4] = lg.gradient.y();
    losses_and_gradients[super_i*10+5] = lg.gradient.z();

    losses_and_gradients[super_i*10+6] = lg_depth.loss.x();
    losses_and_gradients[super_i*10+7] = lg_depth.gradient.x();

    losses_and_gradients[super_i*10+8] = norm;
    losses_and_gradients[super_i*10+9] = norm_depth;


    float mean_loss = lg.loss.mean();
	if (loss_output) {
		loss_output[super_i] = mean_loss;
	}
	if (loss_depth_output) {
        loss_depth_output[super_i] = lg_depth.loss.x();
	}
}





__global__ void compute_gradient(
	const uint32_t n_rays,
	const uint32_t n_effective_rays,
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
    const uint32_t ray_stride,
	const uint32_t kernel_window_size,
    float* __restrict__ xy_image_pixel_indices,
	const int32_t* __restrict__ mapping_indices,
	const float* __restrict__ ground_truth_rgbd,
	const float* __restrict__ reconstructed_rgbd,
	const uint32_t* __restrict__ ray_counter,
    const float* __restrict__ kernel,
	const uint32_t n_super_rays,
	float* __restrict__ losses_and_gradients,
	const uint32_t* __restrict__ super_ray_counter,
	const uint32_t* __restrict__ super_ray_counter_depth
) {

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	//if (i >= *ray_counter) { return; }
	if (i >= n_effective_rays) { return; }

    int32_t ray_idx = mapping_indices[i];

    if (ray_idx < 0){
        return;
    }

    uint32_t super_i = i / ray_stride;
    uint32_t cpt = i % ray_stride;

    // uint32_t u = cpt / kernel_window_size;
    // uint32_t v = cpt % kernel_window_size;

    Array3f dloss_db = {
        losses_and_gradients[super_i*10+3],
        losses_and_gradients[super_i*10+4],
        losses_and_gradients[super_i*10+5]
    };

    float norm = losses_and_gradients[super_i*10+8];

    float norm_depth = losses_and_gradients[super_i*10+9];

    Array3f rgb_ray = {
        reconstructed_rgbd[4*ray_idx],
        reconstructed_rgbd[4*ray_idx+1],
        reconstructed_rgbd[4*ray_idx+2],
    };

    float depth_ray = reconstructed_rgbd[4*ray_idx+3];

    float depth_ray_target = ground_truth_rgbd[4*ray_idx+3];

    float depth_loss_gradient = depth_ray_target > 0.0f ? depth_supervision_lambda * losses_and_gradients[super_i*10+7] : 0;

    uint32_t total_n_rays = *super_ray_counter;
    uint32_t total_n_rays_depth = *super_ray_counter_depth;

    float loss_scale_rgb = loss_scale / (float)total_n_rays;
    float loss_scale_depth;
    if (total_n_rays_depth>0){
        loss_scale_depth = loss_scale / (float)total_n_rays_depth;
    } else {
        loss_scale_depth = 0.0;
    }
    // loss_scale /= total_n_rays;
    // float loss_scale_rgb = loss_scale;
    // float loss_scale_depth = loss_scale;
    // printf(" %f, %f |", loss_scale_rgb, loss_scale_depth);

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


    // No losses and gradients for this ray.
    // zero-out all its gradients.
    if (norm < 0) {
        // for (uint32_t j=0; j < numsteps_compact; ++j) {

        //     tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput = {0.0,0.0,0.0,0.0};

	    //     *(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

	    //     dloss_doutput += padded_output_width;
        // }
        return;
    }


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
	    const Array3f dloss_by_drgb = weight * dloss_db * kernel[cpt] / norm;

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
	        depth_supervision = depth_loss_gradient * (kernel[cpt] / norm_depth) * (T * depth - depth_suffix);
        }

	    const Array3f tmp = dloss_db * kernel[cpt] / norm;
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



__global__ void compute_camera_gradient(
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
    float* __restrict__ xy_image_pixel_indices
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
		const Vector3f pos_gradient = coords_gradient(j)->pos.p.cwiseProduct(warp_position_derivative(warped_pos, aabb));
		ray_gradient.o += pos_gradient;

		const Vector3f pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = (pos - ray.o).norm();
		const Vector3f dir_gradient = coords_gradient(j)->dir.d.cwiseProduct(warp_direction_derivative(coords(j)->dir.d));
		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

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


void Testbed::reset_camera_optimizer(uint32_t cam_id) {
    m_nerf.training.cam_pos_offset[cam_id].reset_state();
    m_nerf.training.cam_rot_offset[cam_id].reset_state();
}

void Testbed::track_pose_nerf_slam_opti(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {

	if (m_nerf.training.indice_image_for_tracking_pose == 0) {
        // no tracking for first frame.
		return;
	}

	m_nerf.training.counters_rgb_track.prepare_for_training_steps(stream);
    m_nerf.training.counters_rgb_track.rays_per_batch = m_nerf.training.rays_per_tracking_batch;

	if (m_nerf.training.n_steps_since_cam_update_tracking == 0) {
	    CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
	    CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
    }

	track_pose_nerf_slam_step_opti(target_batch_size, m_nerf.training.counters_rgb_track, stream);

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
}


void Testbed::track_pose_nerf_slam_step_opti(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

    //NOTE: get settings/hyperparams for tracking
    const float sigma = m_tracking_sigma_gaussian_kernel;
    const uint32_t kernel_window_size = m_tracking_kernel_window_size;
    uint32_t ray_stride = kernel_window_size*kernel_window_size;
    uint32_t sample_away_from_border_margin_h = m_sample_away_from_border_margin_h;
    uint32_t sample_away_from_border_margin_w = m_sample_away_from_border_margin_w;
    uint32_t n_super_rays = counters.rays_per_batch / ray_stride; // get the number of rays for which we have enough room to get the corresponding nearby rays (within window)

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
		float, // ground_truth_rgbd
		float,  // reconstructed_rgbd
		uint32_t, // numsteps_compacted
		float,  // losses_and_gradients
		uint32_t, // num super rays counter
		uint32_t, // num super rays counter depth
		float,  // reconstructed_depth_var
		uint32_t // xy_pixel_indices_int
	>(
		stream, &alloc,
		counters.rays_per_batch,
		counters.rays_per_batch,
		counters.rays_per_batch * 2,
		max_samples * floats_per_coord,
		max_samples,
		std::max(target_batch_size, max_samples) * padded_output_width,
		target_batch_size * padded_output_width,
		target_batch_size * floats_per_coord,
		target_batch_size * floats_per_coord,
		target_batch_size,
		1,
		counters.rays_per_batch * 2,
		counters.rays_per_batch,
		counters.rays_per_batch * 4,
		counters.rays_per_batch * 4,
		counters.rays_per_batch * 2,
		n_super_rays * 10,
        1,
        1,
		counters.rays_per_batch,
		counters.rays_per_batch * 2
	);

	// TODO: C++17 structured binding
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
	float* ground_truth_rgbd = std::get<13>(scratch);
	float* reconstructed_rgbd = std::get<14>(scratch);
	uint32_t* numsteps_compacted = std::get<15>(scratch);
	float* losses_and_gradients = std::get<16>(scratch);
	uint32_t* super_ray_counter = std::get<17>(scratch);
	uint32_t* super_ray_counter_depth = std::get<18>(scratch);
	float* reconstructed_depth_var = std::get<19>(scratch);
	uint32_t* xy_image_pixel_indices_int = std::get<20>(scratch);

	uint32_t max_inference;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference = max_samples;
	} else {
		max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	}

	GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
	GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_training_step_track == 0) {
		counters.n_rays_total = 0;
	}

	counters.n_rays_total += counters.rays_per_batch;
	m_nerf.training.n_rays_since_error_map_update += counters.rays_per_batch;

    m_track_pose_nerf_num_super_rays_targeted_in_tracking_step=n_super_rays;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter_depth, 0, sizeof(uint32_t), stream));

    // create gaussian kernel
    std::vector<float> kernel = make_gaussian_kernel_debug(kernel_window_size, sigma);

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

    //NOTE: get sample xy pixel locations
    linear_kernel(sample_training_pixels_for_tracking, 0, stream,
		n_super_rays,
        ray_stride,
		m_rng,
        kernel_window_size,
        sample_away_from_border_margin_h,
        sample_away_from_border_margin_w,
		m_nerf.training.dataset.metadata_gpu.data(),
        m_nerf.training.indice_image_for_tracking_pose,
		m_nerf.training.snap_to_pixel_centers,
        xy_image_pixel_indices,
        xy_image_pixel_indices_int
	);

    //DEBUG: check the sampled pixels
    m_nerf.training.sampled_pixels_for_tracking.resize(counters.rays_per_batch * 2);
    CUDA_CHECK_THROW(
       cudaMemcpy(
          m_nerf.training.sampled_pixels_for_tracking.data(),
          std::get<20>(scratch),
          counters.rays_per_batch * 2 * sizeof(uint32_t),
          cudaMemcpyDeviceToHost
       )
    );

    //NOTE: get sample along each rays
	linear_kernel(generate_training_samples_for_tracking, 0, stream,
		n_super_rays*ray_stride,
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
        xy_image_pixel_indices
	);

    //DEBUG: check the sampled pixels
    m_nerf.training.sampled_ray_indices_for_tracking_gradient.resize(counters.rays_per_batch);
    CUDA_CHECK_THROW(
       cudaMemcpy(
          m_nerf.training.sampled_ray_indices_for_tracking_gradient.data(),
          std::get<0>(scratch),
          counters.rays_per_batch * sizeof(uint32_t),
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


    //NOTE: get network values for each points
	m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);


    //NOTE: get RGBD values prediciton + GT.
	linear_kernel(compute_GT_and_reconstructed_rgbd, 0, stream,
		counters.rays_per_batch,
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
        ground_truth_rgbd,
        reconstructed_rgbd
	);


    //NOTE: get RGBD values prediciton + GT.
	linear_kernel(compute_depth_variance, 0, stream,
		counters.rays_per_batch,
		m_aabb,
		ray_counter,
		padded_output_width,
		mlp_out,
		rays_unnormalized,
		numsteps,
		PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		m_nerf.density_activation,
        reconstructed_rgbd,
        reconstructed_depth_var
	);


    //NOTE: compute loss.
	linear_kernel(compute_loss, 0, stream,
		n_super_rays,
		m_nerf.training.track_loss_type,
		m_nerf.training.track_depth_loss_type,
		counters.loss.data(),
		counters.loss_depth.data(),
        ray_stride,
        kernel_window_size,
		mapping_indices,
        ground_truth_rgbd,
        reconstructed_rgbd,
        reconstructed_depth_var,
        kernel_gpu.data(),
        losses_and_gradients,
		super_ray_counter,
		super_ray_counter_depth
	);


	CUDA_CHECK_THROW(
       cudaMemcpyAsync(
          &counters.super_rays_counter,
          std::get<17>(scratch),
          sizeof(uint32_t),
          cudaMemcpyDeviceToHost,
          stream
       )
    );

	CUDA_CHECK_THROW(
       cudaMemcpyAsync(
          &counters.super_rays_counter_depth,
          std::get<18>(scratch),
          sizeof(uint32_t),
          cudaMemcpyDeviceToHost,
          stream
       )
    );

    //NOTE: compute loss and gradients.
	linear_kernel(compute_gradient, 0, stream,
		counters.rays_per_batch,
		n_super_rays*ray_stride,
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
        ray_stride,
        kernel_window_size,
        xy_image_pixel_indices,
		mapping_indices,
        ground_truth_rgbd,
        reconstructed_rgbd,
		ray_counter,
        kernel_gpu.data(),
		n_super_rays,
        losses_and_gradients,
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
	linear_kernel(compute_camera_gradient, 0, stream,
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
        xy_image_pixel_indices
	);

	m_rng.advance();

}


NGP_NAMESPACE_END
