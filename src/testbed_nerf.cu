/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed_nerf.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

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

inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
//inline constexpr __device__ uint32_t NERF_STEPS() { return 256; } // finest number of steps per unit length
inline constexpr __device__ uint32_t NERF_CASCADES() { return 8; }

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 16; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }
//inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.001f; }

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

Testbed::NetworkDims Testbed::network_dims_nerf() const {
	NetworkDims dims;
	dims.n_input = sizeof(NerfCoordinate) / sizeof(float);
	dims.n_output = 4;
	dims.n_pos = sizeof(NerfPosition) / sizeof(float);
	return dims;
}

inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
	return NERF_GRID_N_CELLS() * mip;
}

inline __host__ __device__ float calc_cone_angle(float cosine, const vec2& focal_length, float cone_angle_constant) {
	// Pixel size. Doesn't always yield a good performance vs. quality
	// trade off. Especially if training pixels have a much different
	// size than rendering pixels.
	// return cosine*cosine / focal_length.mean();

	return cone_angle_constant;
}

inline __host__ __device__ float to_stepping_space(float t, float cone_angle) {
	if (cone_angle <= 1e-5f) {
		return t / MIN_CONE_STEPSIZE();
	}

	float log1p_c = logf(1.0f + cone_angle);

	float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
	float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

	float at = expf(a * log1p_c);
	float bt = expf(b * log1p_c);

	if (t <= at) {
		return (t - at) / MIN_CONE_STEPSIZE() + a;
	} else if (t <= bt) {
		return logf(t) / log1p_c;
	} else {
		return (t - bt) / MAX_CONE_STEPSIZE() + b;
	}
}

inline __host__ __device__ float from_stepping_space(float n, float cone_angle) {
	if (cone_angle <= 1e-5f) {
		return n * MIN_CONE_STEPSIZE();
	}

	float log1p_c = logf(1.0f + cone_angle);

	float a = (logf(MIN_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;
	float b = (logf(MAX_CONE_STEPSIZE()) - logf(log1p_c)) / log1p_c;

	float at = expf(a * log1p_c);
	float bt = expf(b * log1p_c);

	if (n <= a) {
		return (n - a) * MIN_CONE_STEPSIZE() + at;
	} else if (n <= b) {
		return expf(n * log1p_c);
	} else {
		return (n - b) * MAX_CONE_STEPSIZE() + bt;
	}
}

inline __host__ __device__ float advance_n_steps(float t, float cone_angle, float n) {
	return from_stepping_space(to_stepping_space(t, cone_angle) + n, cone_angle);
}

inline __host__ __device__ float calc_dt(float t, float cone_angle) {
	return advance_n_steps(t, cone_angle, 1.0f) - t;
}

inline __host__ __device__ float calc_dt_with_depth_guided_sampling(
	float t, 
	float cone_angle,
	const float depth, 
	const float target_depth, 
	const float dt_for_depth_guided_sampling, 
	const float truncation_distance_for_depth_guided_sampling
) {
	float dt;
	if ((depth >= (target_depth - truncation_distance_for_depth_guided_sampling)) &
		(depth <  (target_depth + truncation_distance_for_depth_guided_sampling)) &
		(target_depth > 0.)){
		dt = dt_for_depth_guided_sampling;
	} else {
		dt = calc_dt(t, cone_angle);
	}
	return dt;
}

inline __host__ __device__ float calc_dt_slaim(
	const float cur_z, 
	const float target_z, 
	const float dt_for_regular_sampling,
	const float dt_for_depth_guided_sampling, 
	const float truncation_distance_for_depth_guided_sampling, 
	const float cam_fwd_factor, 
	const bool use_depth_guided_sampling
) {
	float dt;
	float dz;
	if (use_depth_guided_sampling) {
		if ((cur_z >= (target_z - truncation_distance_for_depth_guided_sampling)) &
			(cur_z <  (target_z + truncation_distance_for_depth_guided_sampling)) &
			(target_z > 0.)
		) {
			dz = dt_for_depth_guided_sampling;
		} else {
			dz = dt_for_regular_sampling;
		}
	} else {
		dz = dt_for_regular_sampling;
	}
	dt = dz / cam_fwd_factor;
	return dt;
}


struct LossAndGradient {
	vec3 loss;
	vec3 gradient;

	__host__ __device__ LossAndGradient operator*(float scalar) {
		return {loss * scalar, gradient * scalar};
	}

	__host__ __device__ LossAndGradient operator/(float scalar) {
		return {loss / scalar, gradient / scalar};
	}
};

inline __device__ vec3 copysign(const vec3& a, const vec3& b) {
	return {
		copysignf(a.x, b.x),
		copysignf(a.y, b.y),
		copysignf(a.z, b.z),
	};
}

inline __device__ LossAndGradient l2_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	return {
		difference * difference,
		2.0f * difference
	};
}

inline __device__ LossAndGradient relative_l2_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 denom = prediction * prediction + vec3(1e-2f);
	return {
		difference * difference / denom,
		2.0f * difference / denom
	};
}

inline __device__ LossAndGradient l1_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	return {
		abs(difference),
		copysign(vec3(1.0f), difference),
	};
}

inline __device__ LossAndGradient huber_loss(const vec3& target, const vec3& prediction, float alpha = 1) {
	vec3 difference = prediction - target;
	vec3 abs_diff = abs(difference);
	vec3 square = 0.5f/alpha * difference * difference;
	return {
		{
			abs_diff.x > alpha ? (abs_diff.x - 0.5f * alpha) : square.x,
			abs_diff.y > alpha ? (abs_diff.y - 0.5f * alpha) : square.y,
			abs_diff.z > alpha ? (abs_diff.z - 0.5f * alpha) : square.z,
		},
		{
			abs_diff.x > alpha ? (difference.x > 0 ? 1.0f : -1.0f) : (difference.x / alpha),
			abs_diff.y > alpha ? (difference.y > 0 ? 1.0f : -1.0f) : (difference.y / alpha),
			abs_diff.z > alpha ? (difference.z > 0 ? 1.0f : -1.0f) : (difference.z / alpha),
		},
	};
}

inline __device__ LossAndGradient log_l1_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 divisor = abs(difference) + vec3(1.0f);
	return {
		log(divisor),
		copysign(vec3(1.0f) / divisor, difference),
	};
}

inline __device__ LossAndGradient smape_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 denom = 0.5f * (abs(prediction) + abs(target)) + vec3(1e-2f);
	return {
		abs(difference) / denom,
		copysign(vec3(1.0f) / denom, difference),
	};
}

inline __device__ LossAndGradient mape_loss(const vec3& target, const vec3& prediction) {
	vec3 difference = prediction - target;
	vec3 denom = abs(prediction) + vec3(1e-2f);
	return {
		abs(difference) / denom,
		copysign(vec3(1.0f) / denom, difference),
	};
}

inline __device__ float distance_to_next_voxel(const vec3& pos, const vec3& dir, const vec3& idir, float res) { // dda like step
	vec3 p = res * (pos - vec3(0.5f));
	float tx = (floorf(p.x + 0.5f + 0.5f * sign(dir.x)) - p.x) * idir.x;
	float ty = (floorf(p.y + 0.5f + 0.5f * sign(dir.y)) - p.y) * idir.y;
	float tz = (floorf(p.z + 0.5f + 0.5f * sign(dir.z)) - p.z) * idir.z;
	float t = min(min(tx, ty), tz);

	return fmaxf(t / res, 0.0f);
}

inline __device__ float advance_to_next_voxel(float t, float cone_angle, const vec3& pos, const vec3& dir, const vec3& idir, uint32_t mip) {
	float res = scalbnf(NERF_GRIDSIZE(), -(int)mip);

	float t_target = t + distance_to_next_voxel(pos, dir, idir, res);

	// Analytic stepping in multiples of 1 in the "log-space" of our exponential stepping routine
	t = to_stepping_space(t, cone_angle);
	t_target = to_stepping_space(t_target, cone_angle);

	return from_stepping_space(t + ceilf(fmaxf(t_target - t, 0.5f)), cone_angle);
}

__device__ float network_to_rgb(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_rgb_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

template <typename T>
__device__ vec3 network_to_rgb_derivative_vec(const T& val, ENerfActivation activation) {
	return {
		network_to_rgb_derivative(float(val[0]), activation),
		network_to_rgb_derivative(float(val[1]), activation),
		network_to_rgb_derivative(float(val[2]), activation),
	};
}

__device__ float network_to_density(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(val);
		case ENerfActivation::Exponential10x: return __expf(10*val);
		default: assert(false);
	}
	return 0.0f;
}

__device__ float network_to_density_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
		case ENerfActivation::Exponential10x: return 10*__expf(tcnn::clamp(10*val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ __inline__ float sigmoid(float a)
{
        return 1.0 / (1.0 + __expf(-a));
}

__device__ float coslam_sdf_to_weight(float val, float beta) {
	return sigmoid(val / beta) * sigmoid(-val / beta);
}

__device__ float coslam_sdf_to_weight_derivative(float val, float beta) {
	return 1/beta * sigmoid(val / beta) * sigmoid(-val / beta) * (sigmoid(-val / beta) - sigmoid(val / beta));
}

__device__ float style_sdf_to_density(float val, float beta) {
	return 1 / beta * sigmoid(-val / beta);
}

__device__ float style_sdf_to_density_derivative(float val, float beta) {
	return - 1 / beta / beta * sigmoid(-val/beta) * (1 - sigmoid(-val/beta));
}

__device__ float vol_sdf_to_density(float val, float beta) {
	float density;
	float alpha = 1/beta;
	float s = -val;
	if (s<=0){
		density = 0.5f * __expf(s / beta);
	} else {
		density = (1 - 0.5f * __expf(-s / beta));
	}
	density = alpha * density;
	return density;
}

__device__ float vol_sdf_to_density_derivative(float val, float beta) {
	float density;
	float alpha = 1/beta;
	float s = -val;
	if (s<=0){
		density = 0.5f / beta * __expf( s / beta);
	} else {
		density = 0.5f / beta * __expf(-s / beta);
	}
	density = - alpha * density;
	return density;
}


template <typename T>
__device__ vec3 network_to_rgb_vec(const T& val, ENerfActivation activation) {
	return {
		network_to_rgb(float(val[0]), activation),
		network_to_rgb(float(val[1]), activation),
		network_to_rgb(float(val[2]), activation),
	};
}

__device__ vec3 warp_position(const vec3& pos, const BoundingBox& aabb) {
	// return {tcnn::logistic(pos.x - 0.5f), tcnn::logistic(pos.y - 0.5f), tcnn::logistic(pos.z - 0.5f)};
	// return pos;

	return aabb.relative_pos(pos);
}

__device__ vec3 unwarp_position(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
	// return pos;

	return aabb.min + pos * aabb.diag();
}

__device__ vec3 unwarp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
	// return pos;

	return aabb.diag();
}

__device__ vec3 warp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
	return vec3(1.0f) / unwarp_position_derivative(pos, aabb);
}

__host__ __device__ vec3 warp_direction(const vec3& dir) {
	return (dir + vec3(1.0f)) * 0.5f;
}

__device__ vec3 unwarp_direction(const vec3& dir) {
	return dir * 2.0f - vec3(1.0f);
}

__device__ vec3 warp_direction_derivative(const vec3& dir) {
	return vec3(0.5f);
}

__device__ vec3 unwarp_direction_derivative(const vec3& dir) {
	return vec3(2.0f);
}

__device__ float warp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

__device__ float unwarp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}

__device__ uint32_t cascaded_grid_idx_at(vec3 pos, uint32_t mip) {
	float mip_scale = scalbnf(1.0f, -mip);
	pos -= vec3(0.5f);
	pos *= mip_scale;
	pos += vec3(0.5f);

	ivec3 i = pos * (float)NERF_GRIDSIZE();
	if (i.x < 0 || i.x >= NERF_GRIDSIZE() || i.y < 0 || i.y >= NERF_GRIDSIZE() || i.z < 0 || i.z >= NERF_GRIDSIZE()) {
		return 0xFFFFFFFF;
	}

	return tcnn::morton3D(i.x, i.y, i.z);
}

__device__ bool density_grid_occupied_at(const vec3& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return false;
	}
	return density_grid_bitfield[idx/8+grid_mip_offset(mip)/8] & (1<<(idx%8));
}

__device__ float cascaded_grid_at(vec3 pos, const float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return 0.0f;
	}
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__device__ float& cascaded_grid_at(vec3 pos, float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		idx = 0;
		printf("WARNING: invalid cascaded grid access.");
	}
	return cascaded_grid[idx+grid_mip_offset(mip)];
}

__global__ void extract_srgb_with_activation(const uint32_t n_elements,	const uint32_t rgb_stride, const float* __restrict__ rgbd, float* __restrict__ rgb, ENerfActivation rgb_activation, bool from_linear) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	const uint32_t elem_idx = i / 3;
	const uint32_t dim_idx = i - elem_idx * 3;

	float c = network_to_rgb(rgbd[elem_idx*4 + dim_idx], rgb_activation);
	if (from_linear) {
		c = linear_to_srgb(c);
	}

	rgb[elem_idx*rgb_stride + dim_idx] = c;
}

__global__ void mark_untrained_density_grid(const uint32_t n_elements,  float* __restrict__ grid_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	bool clear_visible_voxels
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t level = i / NERF_GRID_N_CELLS();
	uint32_t pos_idx = i % NERF_GRID_N_CELLS();

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	float voxel_size = scalbnf(1.0f / NERF_GRIDSIZE(), level);
	vec3 pos = (vec3{(float)x, (float)y, (float)z} / (float)NERF_GRIDSIZE() - vec3(0.5f)) * scalbnf(1.0f, level) + vec3(0.5f);

	vec3 corners[8] = {
		pos + vec3{0.0f,       0.0f,       0.0f      },
		pos + vec3{voxel_size, 0.0f,       0.0f      },
		pos + vec3{0.0f,       voxel_size, 0.0f      },
		pos + vec3{voxel_size, voxel_size, 0.0f      },
		pos + vec3{0.0f,       0.0f,       voxel_size},
		pos + vec3{voxel_size, 0.0f,       voxel_size},
		pos + vec3{0.0f,       voxel_size, voxel_size},
		pos + vec3{voxel_size, voxel_size, voxel_size},
	};

	// Number of training views that need to see a voxel cell
	// at minimum for that cell to be marked trainable.
	// Floaters can be reduced by increasing this value to 2,
	// but at the cost of certain reconstruction artifacts.
	const uint32_t min_count = 1;
	uint32_t count = 0;

	for (uint32_t j = 0; j < n_training_images && count < min_count; ++j) {
		const auto& xform = training_xforms[j].start;
		const auto& m = metadata[j];

		if (m.lens.mode == ELensMode::FTheta || m.lens.mode == ELensMode::LatLong || m.lens.mode == ELensMode::Equirectangular) {
			// FTheta lenses don't have a forward mapping, so are assumed seeing everything. Latlong and equirect lenses
			// by definition see everything.
			++count;
			continue;
		}

		for (uint32_t k = 0; k < 8; ++k) {
			// Only consider voxel corners in front of the camera
			vec3 dir = normalize(corners[k] - xform[3]);
			if (dot(dir, xform[2]) < 1e-4f) {
				continue;
			}

			// Check if voxel corner projects onto the image plane, i.e. uv must be in (0, 1)^2
			vec2 uv = pos_to_uv(corners[k], m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), {}, m.lens);

			// `pos_to_uv` is _not_ injective in the presence of lens distortion (which breaks down outside of the image plane).
			// So we need to check whether the produced uv location generates a ray that matches the ray that we started with.
			Ray ray = uv_to_ray(0.0f, uv, m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, m.lens);
			if (distance(normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
				++count;
				break;
			}
		}
	}

	if (clear_visible_voxels || (grid_out[i] < 0) != (count < min_count)) {
		grid_out[i] = (count >= min_count) ? 0.f : -1.f;
	}
}

__global__ void generate_grid_samples_nerf_uniform(ivec3 res_3d, const uint32_t step, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb, NerfPosition* __restrict__ out) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z) {
		return;
	}

	uint32_t i = x + y * res_3d.x + z * res_3d.x * res_3d.y;
	vec3 pos = vec3{(float)x, (float)y, (float)z} / vec3(res_3d - ivec3(1));
	pos = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);
	out[i] = { warp_position(pos, train_aabb), warp_dt(MIN_CONE_STEPSIZE()) };
}

// generate samples for uniform grid including constant ray direction
__global__ void generate_grid_samples_nerf_uniform_dir(ivec3 res_3d, const uint32_t step, BoundingBox render_aabb, mat3 render_aabb_to_local, BoundingBox train_aabb, vec3 ray_dir, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims, bool voxel_centers) {
	// check grid_in for negative values -> must be negative on output
	uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	uint32_t z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= res_3d.x || y >= res_3d.y || z >= res_3d.z) {
		return;
	}

	uint32_t i = x+ y*res_3d.x + z*res_3d.x*res_3d.y;
	vec3 pos;
	if (voxel_centers) {
		pos = vec3{(float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f} / vec3(res_3d);
	} else {
		pos = vec3{(float)x, (float)y, (float)z} / vec3(res_3d - ivec3(1));
	}

	pos = transpose(render_aabb_to_local) * (pos * (render_aabb.max - render_aabb.min) + render_aabb.min);

	network_input(i)->set_with_optional_extra_dims(warp_position(pos, train_aabb), warp_direction(ray_dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

inline __device__ uint32_t mip_from_pos(const vec3& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	int exponent;
	float maxval = compMax(abs(pos - vec3(0.5f)));
	frexpf(maxval, &exponent);
	return (uint32_t)tcnn::clamp(exponent+1, 0, (int)max_cascade);
}

inline __device__ uint32_t mip_from_dt(float dt, const vec3& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	uint32_t mip = mip_from_pos(pos, max_cascade);
	dt *= 2 * NERF_GRIDSIZE();
	if (dt < 1.0f) {
		return mip;
	}

	int exponent;
	frexpf(dt, &exponent);
	return (uint32_t)tcnn::clamp((int)mip, exponent, (int)max_cascade);
}

__global__ void generate_grid_samples_nerf_nonuniform(const uint32_t n_elements, default_rng_t rng, const uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, uint32_t* __restrict__ indices, uint32_t n_cascades, float thresh) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// 1 random number to select the level, 3 to select the position.
	rng.advance(i*4);
	uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

	// Select grid cell that has density
	uint32_t idx;
	for (uint32_t j = 0; j < 10; ++j) {
		idx = ((i+step*n_elements) * 56924617 + j * 19349663 + 96925573) % NERF_GRID_N_CELLS();
		idx += level * NERF_GRID_N_CELLS();
		if (grid_in[idx] > thresh) {
			break;
		}
	}

	// Random position within that cellq
	uint32_t pos_idx = idx % NERF_GRID_N_CELLS();

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	vec3 pos = ((vec3{(float)x, (float)y, (float)z} + random_val_3d(rng)) / (float)NERF_GRIDSIZE() - vec3(0.5f)) * scalbnf(1.0f, level) + vec3(0.5f);

	out[i] = { warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE()) };
	indices[i] = idx;
}

__global__ void splat_grid_samples_nerf_max_nearest_neighbor(const uint32_t n_elements, const uint32_t* __restrict__ indices, 
															const tcnn::network_precision_t* network_output, float* __restrict__ grid_out, 
															ENerfActivation rgb_activation, ENerfActivation density_activation,
															const bool use_volsdf, const bool use_stylesdf, const bool use_coslam_sdf, const float beta
															) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t local_idx = indices[i];

	// Current setting: optical thickness of the smallest possible stepsize.
	// Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
	uint32_t level = 0;//local_idx / NERF_GRID_N_CELLS();

	float mlp = network_to_density(float(network_output[i]), density_activation);
	if (use_volsdf){
		mlp = vol_sdf_to_density(mlp, beta);
	} else if (use_stylesdf) {
		mlp = style_sdf_to_density(mlp, beta);
	} else if (use_coslam_sdf){
		mlp = vol_sdf_to_density(mlp, beta/10.);
	}
	
	float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);

	// Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
	// uint atomicMax is thus perfectly acceptable.
	atomicMax((uint32_t*)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

__global__ void grid_samples_half_to_float(const uint32_t n_elements, BoundingBox aabb, float* dst, const tcnn::network_precision_t* network_output, ENerfActivation density_activation, const NerfPosition* __restrict__ coords_in, const float* __restrict__ grid_in, uint32_t max_cascade, const bool use_sdf) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// let's interpolate for marching cubes based on the raw MLP output, not the density (exponentiated) version
	//float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
	float mlp = float(network_output[i]);
	if (use_sdf) {
		mlp = -mlp;
	}

	if (grid_in) {
		vec3 pos = unwarp_position(coords_in[i].p, aabb);
		float grid_density = cascaded_grid_at(pos, grid_in, mip_from_pos(pos, max_cascade));
		if (grid_density < NERF_MIN_OPTICAL_THICKNESS()) {
			mlp = -10000.0f;
		}
	}

	dst[i] = mlp;
}

__global__ void ema_grid_samples_nerf(const uint32_t n_elements,
	float decay,
	const uint32_t count,
	float* __restrict__ grid_out,
	const float* __restrict__ grid_in
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	float importance = grid_in[i];

	// float ema_debias_old = 1 - (float)powf(decay, count);
	// float ema_debias_new = 1 - (float)powf(decay, count+1);

	// float filtered_val = ((grid_out[i] * decay * ema_debias_old + importance * (1 - decay)) / ema_debias_new);
	// grid_out[i] = filtered_val;

	// Maximum instead of EMA allows capture of very thin features.
	// Basically, we want the grid cell turned on as soon as _ANYTHING_ visible is in there.

	float prev_val = grid_out[i];
	float val = (prev_val<0.f) ? prev_val : fmaxf(prev_val * decay, importance);
	grid_out[i] = val;
}

__global__ void decay_sharpness_grid_nerf(const uint32_t n_elements, float decay, float* __restrict__ grid) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	grid[i] *= decay;
}


__global__ void grid_to_bitfield_slam(
	const uint32_t n_elements,
	const uint32_t n_nonzero_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr,
	const uint8_t* __restrict__ grid_unvisible
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	if (i >= n_nonzero_elements) {
		grid_bitfield[i] = 0;
		return;
	}

	uint8_t bits = 0;

	float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		//if cell is unvisible (not yet viewed during SLAM)	-> set it to occupied as default
		if (grid_unvisible[i*8+j]>0) {
			bits |= ((uint8_t)1 << j);
		} else {
			bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
		}
	}

	grid_bitfield[i] = bits;
}

__global__ void grid_to_bitfield(
	const uint32_t n_elements,
	const uint32_t n_nonzero_elements,
	const float* __restrict__ grid,
	uint8_t* __restrict__ grid_bitfield,
	const float* __restrict__ mean_density_ptr
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	if (i >= n_nonzero_elements) {
		grid_bitfield[i] = 0;
		return;
	}

	uint8_t bits = 0;

	float thresh = std::min(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		// -- DEBUG  // this is a hack to bypass the importance sampling strategy of NGP
		// we're basically setting the entire grid a dense so each sample a long rays will be evaluated.
		//bits |= ((uint8_t)1 << j);
		bits |= grid[i*8+j] > thresh ? ((uint8_t)1 << j) : 0;
	}

	grid_bitfield[i] = bits;
}

__global__ void bitfield_max_pool(const uint32_t n_elements,
	const uint8_t* __restrict__ prev_level,
	uint8_t* __restrict__ next_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint8_t bits = 0;

	NGP_PRAGMA_UNROLL
	for (uint8_t j = 0; j < 8; ++j) {
		// If any bit is set in the previous level, set this
		// level's bit. (Max pooling.)
		bits |= prev_level[i*8+j] > 0 ? ((uint8_t)1 << j) : 0;
	}

	uint32_t x = tcnn::morton3D_invert(i>>0) + NERF_GRIDSIZE()/8;
	uint32_t y = tcnn::morton3D_invert(i>>1) + NERF_GRIDSIZE()/8;
	uint32_t z = tcnn::morton3D_invert(i>>2) + NERF_GRIDSIZE()/8;

	next_level[tcnn::morton3D(x, y, z)] |= bits;
}

template <bool MIP_FROM_DT=false>
__device__ float if_unoccupied_advance_to_next_occupied_voxel(
	float t,
	float cone_angle,
	const Ray& ray,
	const vec3& idir,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	BoundingBox aabb,
	mat3 aabb_to_local = mat3(1.0f)
) {
	while (true) {
		vec3 pos = ray(t);
		if (t >= MAX_DEPTH() || !aabb.contains(aabb_to_local * pos)) {
			return MAX_DEPTH();
		}

		uint32_t mip = tcnn::clamp(MIP_FROM_DT ? mip_from_dt(calc_dt(t, cone_angle), pos) : mip_from_pos(pos), min_mip, max_mip);

		if (!density_grid || density_grid_occupied_at(pos, density_grid, mip)) {
			return t;
		}

		// Find largest empty voxel surrounding us, such that we can advance as far as possible in the next step.
		// Other places that do voxel stepping don't need this, because they don't rely on thread coherence as
		// much as this one here.
		while (mip < max_mip && !density_grid_occupied_at(pos, density_grid, mip+1)) {
			++mip;
		}

		t = advance_to_next_voxel(t, cone_angle, pos, ray.d, idir, mip);
	}
}

__device__ void advance_pos_nerf(
	NerfPayload& payload,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	const vec3& camera_fwd,
	const vec2& focal_length,
	uint32_t sample_index,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant
) {
	if (!payload.alive) {
		return;
	}

	vec3 origin = payload.origin;
	vec3 dir = payload.dir;
	vec3 idir = vec3(1.0f) / dir;

	float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

	float t = advance_n_steps(payload.t, cone_angle, ld_random_val(sample_index, payload.idx * 786433));
	if (density_grid) {
		t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
	}
	if (t >= MAX_DEPTH()) {
		payload.alive = false;
	} else {
		payload.t = t;
	}
}

__global__ void advance_pos_nerf_kernel(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	vec3 camera_fwd,
	vec2 focal_length,
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	advance_pos_nerf(payloads[i], render_aabb, render_aabb_to_local, camera_fwd, focal_length, sample_index, density_grid, min_mip, max_mip, cone_angle_constant);
}

__global__ void generate_nerf_network_inputs_from_positions(const uint32_t n_elements, BoundingBox aabb, const vec3* __restrict__ pos, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	vec3 dir = normalize(pos[i] - vec3(0.5f)); // choose outward pointing directions, for want of a better choice
	network_input(i)->set_with_optional_extra_dims(warp_position(pos[i], aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__global__ void generate_nerf_network_inputs_at_current_position(const uint32_t n_elements, BoundingBox aabb, const NerfPayload* __restrict__ payloads, PitchedPtr<NerfCoordinate> network_input, const float* extra_dims) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	vec3 dir = payloads[i].dir;
	network_input(i)->set_with_optional_extra_dims(warp_position(payloads[i].origin + dir * payloads[i].t, aabb), warp_direction(dir), warp_dt(MIN_CONE_STEPSIZE()), extra_dims, network_input.stride_in_bytes);
}

__device__ vec4 compute_nerf_rgba(const vec4& network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
	vec4 rgba = network_output;

	float density = network_to_density(rgba.a, density_activation);
	float alpha = 1.f;
	if (density_as_alpha) {
		rgba.a = density;
	} else {
		rgba.a = alpha = tcnn::clamp(1.f - __expf(-density * depth), 0.0f, 1.0f);
	}

	rgba.rgb = network_to_rgb_vec(rgba.rgb, rgb_activation) * alpha;
	return rgba;
}

__global__ void compute_nerf_rgba_kernel(const uint32_t n_elements, vec4* network_output, ENerfActivation rgb_activation, ENerfActivation density_activation, float depth, bool density_as_alpha = false) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	network_output[i] = compute_nerf_rgba(network_output[i], rgb_activation, density_activation, depth, density_as_alpha);
}

__global__ void generate_next_nerf_network_inputs(
	const uint32_t n_elements,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	BoundingBox train_aabb,
	vec2 focal_length,
	vec3 camera_fwd,
	NerfPayload* __restrict__ payloads,
	PitchedPtr<NerfCoordinate> network_input,
	uint32_t n_steps,
	const uint8_t* __restrict__ density_grid,
	uint32_t min_mip,
	uint32_t max_mip,
	float cone_angle_constant,
	const float* extra_dims,
	const bool use_custom_ray_marching,
	const float dt_for_regular_sampling
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	vec3 origin = payload.origin;
	vec3 dir = payload.dir;
	vec3 idir = vec3(1.0f) / dir;

	float cone_angle = calc_cone_angle(dot(dir, camera_fwd), focal_length, cone_angle_constant);

	float cam_fwd_factor = dot(camera_fwd, dir);

	float t = payload.t;

	for (uint32_t j = 0; j < n_steps; ++j) {
		if (density_grid) {
			t = if_unoccupied_advance_to_next_occupied_voxel(t, cone_angle, {origin, dir}, idir, density_grid, min_mip, max_mip, render_aabb, render_aabb_to_local);
		}
		if (t >= MAX_DEPTH()) {
			payload.n_steps = j;
			return;
		}

		float dt;
		if (use_custom_ray_marching){
			vec3 pos = origin + t*dir;
			float cur_z = dot(camera_fwd, pos - origin);
			
			//TODO: make use of target depth if any
			// (for depth guided sampling)
			dt = calc_dt_slaim(
				cur_z, -1.f, dt_for_regular_sampling, 
				-1.f, -1.f,
				cam_fwd_factor, false
			);
		} else {
			dt = calc_dt(t, cone_angle);
		}
		network_input(i + j * n_elements)->set_with_optional_extra_dims(warp_position(origin + dir * t, train_aabb), warp_direction(dir), warp_dt(dt), extra_dims, network_input.stride_in_bytes); // XXXCONE
		t += dt;
	}

	payload.t = t;
	payload.n_steps = n_steps;
}


__global__ void composite_kernel_nerf_sdf(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t current_step,
	BoundingBox aabb,
	float glow_y_cutoff,
	int glow_mode,
	mat4x3 camera_matrix,
	vec2 focal_length,
	float depth_scale,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* payloads,
	PitchedPtr<NerfCoordinate> network_input,
	const tcnn::network_precision_t* __restrict__ network_output,
	uint32_t padded_output_width,
	uint32_t n_steps,
	ERenderMode render_mode,
	const uint8_t* __restrict__ density_grid,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_transmittance,
    float truncation_distance
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	vec4 local_rgba = rgba[i];
	float local_depth = depth[i];
	vec3 origin = payload.origin;
	vec3 cam_fwd = camera_matrix[2];
	// Composite in the last n steps
	uint32_t actual_n_steps = payload.n_steps;
	uint32_t j = 0;

	float zmin = payload.zmin;
	bool found_surface = payload.found_surface;
	float prev_sdf = payload.prev_sdf;
	float norm_weights = payload.norm_weights;

	for (; j < actual_n_steps; ++j) {
		tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output;
		local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
		local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
		local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
		local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
		const NerfCoordinate* input = network_input(i + j * n_elements);
		vec3 warped_pos = input->pos.p;
		vec3 pos = unwarp_position(warped_pos, aabb);

		float cur_depth = distance(pos, origin);

		float cur_z = dot(cam_fwd, pos - origin);

		float sdf = network_to_density(float(local_network_output[3]), density_activation);

        const float weight = sigmoid(sdf / truncation_distance) * sigmoid(-sdf / truncation_distance);

        prev_sdf *= sdf;

        if (prev_sdf < 0. & !found_surface) {
            found_surface = true;
            zmin = cur_z;
			payload.zmin = zmin;
			payload.found_surface = found_surface;
        }
        if (found_surface) {
		    if (cur_z > (zmin + truncation_distance)) {
				local_rgba /= norm_weights;
				norm_weights = 1.0;
		    	break;
		    }
        }

		prev_sdf = sdf;

        norm_weights += weight;

		vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);

		if (render_mode == ERenderMode::Normals) {
			// Network input contains the gradient of the network output w.r.t. input.
			// So to compute density gradients, we need to apply the chain rule.
			// The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
			vec3 normal = -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
			rgb = normalize(normal);
		} else if (render_mode == ERenderMode::Positions) {
			if (show_accel >= 0) {
				uint32_t mip = max(show_accel, mip_from_pos(pos));
				uint32_t res = NERF_GRIDSIZE() >> mip;
				int ix = pos.x * res;
				int iy = pos.y * res;
				int iz = pos.z * res;
				default_rng_t rng(ix + iy * 232323 + iz * 727272);
				rgb.x = 1.f - mip * (1.f / (NERF_CASCADES() - 1));
				rgb.y = rng.next_float();
				rgb.z = rng.next_float();
			} else {
				rgb = (pos - vec3(0.5f)) / 2.0f + vec3(0.5f);
			}
		} else if (render_mode == ERenderMode::EncodingVis) {
			rgb = warped_pos;
		} else if (render_mode == ERenderMode::Depth) {
			rgb = vec3(dot(cam_fwd, pos - origin) * depth_scale);
		} else if (render_mode == ERenderMode::AO) {
			//rgb = vec3(alpha);
			rgb = vec3(weight); // this is likely wrong.
		}

		local_rgba += vec4(rgb * weight, weight);
		if (weight > payload.max_weight) {
			payload.max_weight = weight;
			local_depth = dot(cam_fwd, pos - camera_matrix[3]);
		}

		//if (local_rgba.a > (1.0f - min_transmittance)) {
		//	local_rgba /= local_rgba.a;
		//	break;
		//}
	}

	if (j < n_steps) {
		local_rgba /= norm_weights;

		payload.alive = false;
		payload.n_steps = j + current_step;
	}

	payload.prev_sdf = prev_sdf;
	payload.norm_weights = norm_weights;

	rgba[i] = local_rgba;
	depth[i] = local_depth;
}




__global__ void composite_kernel_nerf(
	const uint32_t n_elements,
	const uint32_t stride,
	const uint32_t current_step,
	BoundingBox aabb,
	float glow_y_cutoff,
	int glow_mode,
	mat4x3 camera_matrix,
	vec2 focal_length,
	float depth_scale,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* payloads,
	PitchedPtr<NerfCoordinate> network_input,
	const tcnn::network_precision_t* __restrict__ network_output,
	uint32_t padded_output_width,
	uint32_t n_steps,
	ERenderMode render_mode,
	const uint8_t* __restrict__ density_grid,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	float min_transmittance,
	const bool use_style_sdf,
	const bool use_vol_sdf,
	const bool use_coslam_sdf,
	const float beta,
	const float truncation_distance
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& payload = payloads[i];

	if (!payload.alive) {
		return;
	}

	vec4 local_rgba = rgba[i];
	float local_depth = depth[i];
	vec3 origin = payload.origin;
	vec3 cam_fwd = camera_matrix[2];
	// Composite in the last n steps
	uint32_t actual_n_steps = payload.n_steps;
	uint32_t j = 0;

	// coslam sdf related stuff
	bool found_surface = payload.found_surface;
	float surface_d = payload.zmin;

	for (; j < actual_n_steps; ++j) {
		tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output;
		local_network_output[0] = network_output[i + j * n_elements + 0 * stride];
		local_network_output[1] = network_output[i + j * n_elements + 1 * stride];
		local_network_output[2] = network_output[i + j * n_elements + 2 * stride];
		local_network_output[3] = network_output[i + j * n_elements + 3 * stride];
		const NerfCoordinate* input = network_input(i + j * n_elements);
		vec3 warped_pos = input->pos.p;
		vec3 pos = unwarp_position(warped_pos, aabb);
		
		float cur_depth = distance(pos, origin);
		float weight;
		float alpha;

		if (use_coslam_sdf) {
			float sdf = network_to_density(float(local_network_output[3]), density_activation);
			if (sdf<0. && !found_surface){
				found_surface=true;
				surface_d=cur_depth;
				payload.found_surface=true;
				payload.zmin=cur_depth;
			}
			if (found_surface && (cur_depth>surface_d+truncation_distance)){
				weight=0.f;
			} else {
				weight = coslam_sdf_to_weight(sdf, beta);
			}
		} else {
			float T = 1.f - local_rgba.a;
			float dt = unwarp_dt(input->dt);
			float density = network_to_density(float(local_network_output[3]), density_activation);
			if (use_vol_sdf) {
				density = vol_sdf_to_density(density, beta);
			} else if (use_style_sdf) {
				density = style_sdf_to_density(density, beta);
			}
			alpha = 1.f - __expf(-density * dt);
			if (show_accel >= 0) {
				alpha = 1.f;
			}
			weight = alpha * T;
		}

		vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);

		if (glow_mode) { // random grid visualizations ftw!
#if 0
			if (0) {  // extremely startrek edition
				float glow_y = (pos.y - (glow_y_cutoff - 0.5f)) * 2.f;
				if (glow_y>1.f) glow_y=max(0.f,21.f-glow_y*20.f);
				if (glow_y>0.f) {
					float line;
					line =max(0.f,cosf(pos.y*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.x*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.z*2.f*3.141592653589793f * 16.f)-0.95f);
					line+=max(0.f,cosf(pos.y*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.x*4.f*3.141592653589793f * 16.f)-0.975f);
					line+=max(0.f,cosf(pos.z*4.f*3.141592653589793f * 16.f)-0.975f);
					glow_y=glow_y*glow_y*0.5f + glow_y*line*25.f;
					rgb.y+=glow_y;
					rgb.z+=glow_y*0.5f;
					rgb.x+=glow_y*0.25f;
				}
			}
#endif
			float glow = 0.f;

			bool green_grid = glow_mode & 1;
			bool green_cutline = glow_mode & 2;
			bool mask_to_alpha = glow_mode & 4;

			// less used?
			bool radial_mode = glow_mode & 8;
			bool grid_mode = glow_mode & 16; // makes object rgb go black!

			{
				float dist;
				if (radial_mode) {
					dist = distance(pos, camera_matrix[3]);
					dist = min(dist, (4.5f - pos.y) * 0.333f);
				} else {
					dist = pos.y;
				}

				if (grid_mode) {
					glow = 1.f / max(1.f, dist);
				} else {
					float y = glow_y_cutoff - dist; // - (ii*0.005f);
					float mask = 0.f;
					if (y > 0.f) {
						y *= 80.f;
						mask = min(1.f, y);
						//if (mask_mode) {
						//	rgb.x=rgb.y=rgb.z=mask; // mask mode
						//} else
						{
							if (green_cutline) {
								glow += max(0.f, 1.f - abs(1.f -y)) * 4.f;
							}

							if (y>1.f) {
								y = 1.f - (y - 1.f) * 0.05f;
							}

							if (green_grid) {
								glow += max(0.f, y / max(1.f, dist));
							}
						}
					}
					if (mask_to_alpha) {
						weight *= mask;
					}
				}
			}

			if (glow > 0.f) {
				float line;
				line  = max(0.f, cosf(pos.y * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 2.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 4.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 8.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.y * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.x * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				line += max(0.f, cosf(pos.z * 16.f * 3.141592653589793f * 16.f) - 0.975f);
				if (grid_mode) {
					glow = /*glow*glow*0.75f + */ glow * line * 15.f;
					rgb.y = glow;
					rgb.z = glow * 0.5f;
					rgb.x = glow * 0.25f;
				} else {
					glow = glow * glow * 0.25f + glow * line * 15.f;
					rgb.y += glow;
					rgb.z += glow * 0.5f;
					rgb.x += glow * 0.25f;
				}
			}
		} // glow

		if (render_mode == ERenderMode::Normals) {
			// Network input contains the gradient of the network output w.r.t. input.
			// So to compute density gradients, we need to apply the chain rule.
			// The normal is then in the opposite direction of the density gradient (i.e. the direction of decreasing density)
			vec3 normal = -network_to_density_derivative(float(local_network_output[3]), density_activation) * warped_pos;
			rgb = normalize(normal);
		} else if (render_mode == ERenderMode::Positions) {
			if (show_accel >= 0) {
				uint32_t mip = max(show_accel, mip_from_pos(pos));
				uint32_t res = NERF_GRIDSIZE() >> mip;
				int ix = pos.x * res;
				int iy = pos.y * res;
				int iz = pos.z * res;
				default_rng_t rng(ix + iy * 232323 + iz * 727272);
				rgb.x = 1.f - mip * (1.f / (NERF_CASCADES() - 1));
				rgb.y = rng.next_float();
				rgb.z = rng.next_float();
			} else {
				rgb = (pos - vec3(0.5f)) / 2.0f + vec3(0.5f);
			}
		} else if (render_mode == ERenderMode::EncodingVis) {
			rgb = warped_pos;
		} else if (render_mode == ERenderMode::Depth) {
			rgb = vec3(dot(cam_fwd, pos - origin) * depth_scale);
		} else if (render_mode == ERenderMode::AO) {
			rgb = vec3(alpha);
		}

		local_rgba += vec4(rgb * weight, weight);
		if (weight > payload.max_weight) {
			payload.max_weight = weight;
			local_depth = dot(cam_fwd, pos - camera_matrix[3]);
		}

		if (use_coslam_sdf){
			if (found_surface && (cur_depth>surface_d+truncation_distance)) {
				local_rgba /= local_rgba.a;
				break;
			}
		} else {
			if (local_rgba.a > (1.0f - min_transmittance)) {
				local_rgba /= local_rgba.a;
				break;
			}
		}
	}

	if (j < n_steps) {
		payload.alive = false;
		payload.n_steps = j + current_step;
	}

	rgba[i] = local_rgba;
	depth[i] = local_depth;
}

static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;

inline __device__ vec2 sample_cdf_2d(vec2 sample, uint32_t img, const ivec2& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, float* __restrict__ pdf) {
	if (sample.x < UNIFORM_SAMPLING_FRACTION) {
		sample.x /= UNIFORM_SAMPLING_FRACTION;
		return sample;
	}

	sample.x = (sample.x - UNIFORM_SAMPLING_FRACTION) / (1.0f - UNIFORM_SAMPLING_FRACTION);

	cdf_y += img * res.y;

	// First select row according to cdf_y
	uint32_t y = binary_search(sample.y, cdf_y, res.y);
	float prev = y > 0 ? cdf_y[y-1] : 0.0f;
	float pmf_y = cdf_y[y] - prev;
	sample.y = (sample.y - prev) / pmf_y;

	cdf_x_cond_y += img * res.y * res.x + y * res.x;

	// Then, select col according to x
	uint32_t x = binary_search(sample.x, cdf_x_cond_y, res.x);
	prev = x > 0 ? cdf_x_cond_y[x-1] : 0.0f;
	float pmf_x = cdf_x_cond_y[x] - prev;
	sample.x = (sample.x - prev) / pmf_x;

	if (pdf) {
		*pdf = pmf_x * pmf_y * compMul(res);
	}

	return {((float)x + sample.x) / (float)res.x, ((float)y + sample.y) / (float)res.y};
}

inline __device__ float pdf_2d(vec2 sample, uint32_t img, const ivec2& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y) {
	ivec2 p = clamp(ivec2(sample * vec2(res)), ivec2(0), res - ivec2(1));

	cdf_y += img * res.y;
	cdf_x_cond_y += img * res.y * res.x + p.y * res.x;

	float pmf_y = cdf_y[p.y];
	if (p.y > 0) {
		pmf_y -= cdf_y[p.y-1];
	}

	float pmf_x = cdf_x_cond_y[p.x];
	if (p.x > 0) {
		pmf_x -= cdf_x_cond_y[p.x-1];
	}

	// Probability mass of picking the pixel
	float pmf = pmf_x * pmf_y;

	// To convert to probability density, divide by area of pixel
	return UNIFORM_SAMPLING_FRACTION + pmf * compMul(res) * (1.0f - UNIFORM_SAMPLING_FRACTION);
}

inline __device__ vec2 nerf_random_image_pos_training(default_rng_t& rng, const ivec2& resolution, bool snap_to_pixel_centers, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, const ivec2& cdf_res, uint32_t img, float* __restrict__ pdf = nullptr) {
	vec2 uv = random_val_2d(rng);

	if (cdf_x_cond_y) {
		uv = sample_cdf_2d(uv, img, cdf_res, cdf_x_cond_y, cdf_y, pdf);
	} else if (pdf) {
		*pdf = 1.0f;
	}

	if (snap_to_pixel_centers) {
		uv = (vec2(clamp(ivec2(uv * vec2(resolution)), ivec2(0), resolution - ivec2(1))) + vec2(0.5f)) / vec2(resolution);
	}

	return uv;
}

inline __device__ uint32_t image_idx(uint32_t base_idx, uint32_t n_rays, uint32_t n_rays_total, uint32_t n_training_images, const float* __restrict__ cdf = nullptr, float* __restrict__ pdf = nullptr) {
	if (cdf) {
		float sample = ld_random_val(base_idx/* + n_rays_total*/, 0xdeadbeef);
		// float sample = random_val(base_idx/* + n_rays_total*/);
		uint32_t img = binary_search(sample, cdf, n_training_images);

		if (pdf) {
			float prev = img > 0 ? cdf[img-1] : 0.0f;
			*pdf = (cdf[img] - prev) * n_training_images;
		}

		return img;
	}

	// return ((base_idx/* + n_rays_total*/) * 56924617 + 96925573) % n_training_images;

	// Neighboring threads in the warp process the same image. Increases locality.
	if (pdf) {
		*pdf = 1.0f;
	}
	return (((base_idx/* + n_rays_total*/) * n_training_images) / n_rays) % n_training_images;
}

__device__ LossAndGradient loss_and_gradient(const vec3& target, const vec3& prediction, ELossType loss_type) {
	switch (loss_type) {
		case ELossType::RelativeL2:  return relative_l2_loss(target, prediction); break;
		case ELossType::L1:          return l1_loss(target, prediction); break;
		case ELossType::Mape:        return mape_loss(target, prediction); break;
		case ELossType::Smape:       return smape_loss(target, prediction); break;
		// Note: we divide the huber loss by a factor of 5 such that its L2 region near zero
		// matches with the L2 loss and error numbers become more comparable. This allows reading
		// off dB numbers of ~converged models and treating them as approximate PSNR to compare
		// with other NeRF methods. Self-normalizing optimizers such as Adam are agnostic to such
		// constant factors; optimization is therefore unaffected.
		case ELossType::Huber:       return huber_loss(target, prediction, 0.1f) / 5.0f; break;
		case ELossType::LogL1:       return log_l1_loss(target, prediction); break;
		default: case ELossType::L2: return l2_loss(target, prediction); break;
	}
}


__global__ void generate_training_samples_nerf(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const uint32_t n_rays_total,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	uint32_t max_mip,
	bool max_level_rand_training,
	float* __restrict__ max_level_ptr,
	bool snap_to_pixel_centers,
	bool train_envmap,
	float cone_angle_constant,
	Buffer2DView<const vec2> distortion,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 cdf_res,
	const float* __restrict__ extra_dims_gpu,
	uint32_t n_extra_dims
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	uint32_t img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
	ivec2 resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(uv, resolution, 0);
	if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f) {
		return;
	}

	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	float motionblur_time = random_val(rng);

	const vec2 focal_length = metadata[img].focal_length;
	const vec2 principal_point = metadata[img].principal_point;
	const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
	const Lens lens = metadata[img].lens;

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);

	Ray ray_unnormalized;
	const Ray* rays_in_unnormalized = metadata[img].rays;
	if (rays_in_unnormalized) {
		// Rays have been explicitly supplied. Read them.
		ray_unnormalized = rays_in_unnormalized[pix_idx];

		/* DEBUG - compare the stored rays to the computed ones
		const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, 0.f);
		Ray ray2;
		ray2.o = xform[3];
		ray2.d = f_theta_distortion(uv, principal_point, lens);
		ray2.d = (xform.block<3, 3>(0, 0) * ray2.d).normalized();
		if (i==1000) {
			printf("\n%d uv %0.3f,%0.3f pixel %0.2f,%0.2f transform from [%0.5f %0.5f %0.5f] to [%0.5f %0.5f %0.5f]\n"
				" origin    [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
				" direction [%0.5f %0.5f %0.5f] vs [%0.5f %0.5f %0.5f]\n"
			, img,uv.x, uv.y, uv.x*resolution.x, uv.y*resolution.y,
				training_xforms[img].start[3].x,training_xforms[img].start[3].y,training_xforms[img].start[3].z,
				training_xforms[img].end[3].x,training_xforms[img].end[3].y,training_xforms[img].end[3].z,
				ray_unnormalized.o.x,ray_unnormalized.o.y,ray_unnormalized.o.z,
				ray2.o.x,ray2.o.y,ray2.o.z,
				ray_unnormalized.d.x,ray_unnormalized.d.y,ray_unnormalized.d.z,
				ray2.d.x,ray2.d.y,ray2.d.z);
		}
		*/
	} else {
		ray_unnormalized = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, lens, distortion);
		if (!ray_unnormalized.is_valid()) {
			ray_unnormalized = {xform[3], xform[2]};
		}
	}

	vec3 ray_d_normalized = normalize(ray_unnormalized.d);

	vec2 tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
	float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x = fmaxf(tminmax.x, 0.0f);

	float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
	vec3 idir = vec3(1.0f) / ray_d_normalized;

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;
	vec3 pos;

	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			++j;
			t += dt;
		} else {
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
		}
	}
	if (j == 0 && !train_envmap) {
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
	rays_out_unnormalized[ray_idx] = ray_unnormalized;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	vec3 warped_dir = warp_direction(ray_d_normalized);
	t=startt;
	j=0;
	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
			++j;
			t += dt;
		} else {
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
		}
	}

	if (max_level_rand_training) {
		max_level_ptr += base;
		for (j = 0; j < numsteps; ++j) {
			max_level_ptr[j] = max_level;
		}
	}
}


__global__ void compute_loss_kernel_train_nerf(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const uint32_t max_samples_compacted,
	const uint32_t* __restrict__ rays_counter,
	float loss_scale,
	int padded_output_width,
	Buffer2DView<const vec4> envmap,
	float* __restrict__ envmap_gradient,
	const ivec2 envmap_resolution,
	ELossType envmap_loss_type,
	vec3 background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	bool max_level_rand_training,
	float* __restrict__ max_level_compacted_ptr,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	bool snap_to_pixel_centers,
	float* __restrict__ error_map,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res,
	const ivec2 error_map_cdf_res,
	const float* __restrict__ sharpness_data,
	ivec2 sharpness_resolution,
	float* __restrict__ sharpness_grid,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	uint32_t max_mip,
	const vec3* __restrict__ exposure,
	vec3* __restrict__ exposure_gradient,
	float depth_supervision_lambda,
	float near_distance
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float EPSILON = 1e-4f;

	vec3 rgb_ray = vec3(0.0f);
	vec3 hitpoint = vec3(0.0f);

	float depth_ray = 0.f;
	uint32_t compacted_numsteps = 0;
	vec3 ray_o = rays_in_unnormalized[i].o;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T < EPSILON) {
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = distance(pos, ray_o);
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
	uint32_t ray_idx = ray_indices_in[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	float img_pdf = 1.0f;
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
	ivec2 resolution = metadata[img].resolution;

	float uv_pdf = 1.0f;
	vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &uv_pdf);
	float max_level = max_level_rand_training ? (random_val(rng) * 2.0f) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level
	rng.advance(1); // motionblur_time

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	vec4 envmap_value;
	vec3 dir;
	if (envmap) {
		dir = normalize(rays_in_unnormalized[i].d);
		envmap_value = read_envmap(envmap, dir);
		background_color = envmap_value.rgb + background_color * (1.0f - envmap_value.a);
	}

	vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
	// vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
	// vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
	vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

	vec3 rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.rgb + (1.0f - texsamp.a) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.a > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

	// Step again, this time computing loss
	network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_in[i*2+0] = compacted_numsteps;
	numsteps_in[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	max_level_compacted_ptr += compacted_base;
	coords_out += compacted_base;

	dloss_doutput += compacted_base * padded_output_width;

	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
	lg.loss /= img_pdf * uv_pdf;

	float target_depth = length(rays_in_unnormalized[i].d) * ((depth_supervision_lambda > 0.0f && metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);
	LossAndGradient lg_depth = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
	float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;

	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * uv_pdf;

	float mean_loss = compAdd(lg.loss) / 3.0f;
	if (loss_output) {
		loss_output[i] = mean_loss / (float)n_rays;
	}

	if (error_map) {
		const vec2 pos = clamp(uv * vec2(error_map_res) - vec2(0.5f), vec2(0.0f), vec2(error_map_res) - vec2(1.0f + 1e-4f));
		const ivec2 pos_int = pos;
		const vec2 weight = pos - vec2(pos_int);

		ivec2 idx = clamp(pos_int, ivec2(0), resolution - ivec2(2));

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[img * compMul(error_map_res) + y * error_map_res.x + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), ivec2(0), sharpness_resolution - ivec2(1));
			float sharp = sharpness_data[img * compMul(sharpness_resolution) + sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x,   idx.y,   (1 - weight.x) * (1 - weight.y) * mean_loss);
		deposit_val(idx.x+1, idx.y,        weight.x  * (1 - weight.y) * mean_loss);
		deposit_val(idx.x,   idx.y+1, (1 - weight.x) *      weight.y  * mean_loss);
		deposit_val(idx.x+1, idx.y+1,      weight.x  *      weight.y  * mean_loss);
	}

	loss_scale /= n_rays;

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
	vec3 rgb_ray2 = { 0.f,0.f,0.f };
	float depth_ray2 = 0.f;
	T = 1.f;
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {
		if (max_level_rand_training) {
			max_level_compacted_ptr[j] = max_level;
		}
		// Compact network inputs
		NerfCoordinate* coord_out = coords_out(j);
		const NerfCoordinate* coord_in = coords_in(j);
		coord_out->copy(*coord_in, coords_out.stride_in_bytes);

		const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
		float depth = distance(pos, ray_o);

		float dt = unwarp_dt(coord_in->dt);
		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray2 += weight * rgb;
		depth_ray2 += weight * depth;
		T *= (1.f - alpha);

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const vec3 suffix = rgb_ray - rgb_ray2;
		const vec3 dloss_by_drgb = weight * lg.gradient;

		tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		const float depth_suffix = depth_ray - depth_ray2;
		const float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);

		float dloss_by_dmlp = density_derivative * (
			dt * (dot(lg.gradient, T * rgb - suffix) + depth_supervision)
		);

		//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
		//dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

		local_dL_doutput[3] =
			loss_scale * dloss_by_dmlp +
			(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
			(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
			;

		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += padded_output_width;
	}

	if (exposure_gradient) {
		// Assume symmetric loss
		vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

		if (!train_in_linear_colors) {
			dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
		}

		// 2^exposure * log(2)
		vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
		atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
		atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
		atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
	}

	if (compacted_numsteps == numsteps && envmap_gradient) {
		vec3 loss_gradient = lg.gradient;
		if (envmap_loss_type != loss_type) {
			loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
		}

		vec3 dloss_by_dbackground = T * loss_gradient;
		if (!train_in_linear_colors) {
			dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
		}

		tcnn::vector_t<tcnn::network_precision_t, 4> dL_denvmap;
		dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
		dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
		dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;


		float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

		// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
		dL_denvmap[3] = (tcnn::network_precision_t)0;

		deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
	}
}


__global__ void compute_cam_gradient_train_nerf(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	const TrainingXForm* training_xforms,
	bool snap_to_pixel_centers,
	vec3* cam_pos_gradient,
	vec3* cam_rot_gradient,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords,
	PitchedPtr<NerfCoordinate> coords_gradient,
	float* __restrict__ distortion_gradient,
	float* __restrict__ distortion_gradient_weight,
	const ivec2 distortion_resolution,
	vec2* cam_focal_length_gradient,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}

	uint32_t base = numsteps_in[i*2+1];
	coords += base;
	coords_gradient += base;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);
	ivec2 resolution = metadata[img].resolution;

	const mat4x3& xform = training_xforms[img].start;

	Ray ray = rays_in_unnormalized[i];
	ray.d = normalize(ray.d);
	Ray ray_gradient = { vec3(0.0f), vec3(0.0f) };

	// Compute ray gradient
	for (uint32_t j = 0; j < numsteps; ++j) {
		const vec3 warped_pos = coords(j)->pos.p;
		const vec3 pos_gradient = coords_gradient(j)->pos.p * warp_position_derivative(warped_pos, aabb);
		ray_gradient.o += pos_gradient;
		const vec3 pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = distance(pos, ray.o);
		const vec3 dir_gradient = coords_gradient(j)->dir.d * warp_direction_derivative(coords(j)->dir.d);
		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
	float uv_pdf = 1.0f;

	vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, &uv_pdf);

	if (distortion_gradient) {
		// Projection of the raydir gradient onto the plane normal to raydir,
		// because that's the only degree of motion that the raydir has.
		vec3 orthogonal_ray_gradient = ray_gradient.d - ray.d * dot(ray_gradient.d, ray.d);

		// Rotate ray gradient to obtain image plane gradient.
		// This has the effect of projecting the (already projected) ray gradient from the
		// tangent plane of the sphere onto the image plane (which is correct!).
		vec3 image_plane_gradient = inverse(mat3(xform)) * orthogonal_ray_gradient;

		// Splat the resulting 2D image plane gradient into the distortion params
		deposit_image_gradient(image_plane_gradient.xy() / uv_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, uv);
	}

	if (cam_pos_gradient) {
		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / uv_pdf);
		}
	}

	if (cam_rot_gradient) {
		// Rotation is averaged in log-space (i.e. by averaging angle-axes).
		// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
		// orthogonal, leading to the angle_axis magnitude to equal the magnitude
		// of ray_gradient.d.
		vec3 angle_axis = cross(ray.d, ray_gradient.d);

		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / uv_pdf);
		}
	}
}

__global__ void compute_extra_dims_gradient_train_nerf(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	const uint32_t* __restrict__ rays_counter,
	float* extra_dims_gradient,
	uint32_t n_extra_dims,
	const uint32_t n_training_images,
	const uint32_t* __restrict__ ray_indices_in,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords_gradient,
	const float* __restrict__ cdf_img
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}
	uint32_t base = numsteps_in[i*2+1];
	coords_gradient += base;
	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);

	extra_dims_gradient += n_extra_dims * img;

	for (uint32_t j = 0; j < numsteps; ++j) {
		const float *src = coords_gradient(j)->get_extra_dims();
		for (uint32_t k = 0; k < n_extra_dims; ++k) {
			atomicAdd(&extra_dims_gradient[k], src[k]);
		}
	}
}




__global__ void shade_kernel_nerf(
	const uint32_t n_elements,
	vec4* __restrict__ rgba,
	float* __restrict__ depth,
	NerfPayload* __restrict__ payloads,
	ERenderMode render_mode,
	bool train_in_linear_colors,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	const bool use_sdf_in_nerf
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;
	NerfPayload& payload = payloads[i];

	vec4 tmp = rgba[i];
	if (use_sdf_in_nerf){
		tmp /= tmp.a;
	}

	//DEBUG
	//if (tmp.a > 1.1){printf(" %f |", tmp.a);}

	if (render_mode == ERenderMode::Normals) {
		vec3 n = normalize(tmp.xyz());
		tmp.rgb = (0.5f * n + vec3(0.5f)) * tmp.a;
	} else if (render_mode == ERenderMode::Cost) {
		float col = (float)payload.n_steps / 128;
		tmp = {col, col, col, 1.0f};
	}

	if (!train_in_linear_colors && (render_mode == ERenderMode::Shade || render_mode == ERenderMode::Slice)) {
		// Accumulate in linear colors
		tmp.rgb = srgb_to_linear(tmp.rgb);
	}

	frame_buffer[payload.idx] = tmp + frame_buffer[payload.idx] * (1.0f - tmp.a);
	if (render_mode != ERenderMode::Slice && tmp.a > 0.2f) {
		depth_buffer[payload.idx] = depth[i];
	}
}

__global__ void compact_kernel_nerf(
	const uint32_t n_elements,
	vec4* src_rgba, float* src_depth, NerfPayload* src_payloads,
	vec4* dst_rgba, float* dst_depth, NerfPayload* dst_payloads,
	vec4* dst_final_rgba, float* dst_final_depth, NerfPayload* dst_final_payloads,
	uint32_t* counter, uint32_t* finalCounter
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	NerfPayload& src_payload = src_payloads[i];

	if (src_payload.alive) {
		uint32_t idx = atomicAdd(counter, 1);
		dst_payloads[idx] = src_payload;
		dst_rgba[idx] = src_rgba[i];
		dst_depth[idx] = src_depth[i];
	} else if (src_rgba[i].a > 0.001f) {
		uint32_t idx = atomicAdd(finalCounter, 1);
		dst_final_payloads[idx] = src_payload;
		dst_final_rgba[idx] = src_rgba[i];
		dst_final_depth[idx] = src_depth[i];
	}
}

__global__ void init_rays_with_payload_kernel_nerf(
	uint32_t sample_index,
	NerfPayload* __restrict__ payloads,
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera_matrix0,
	mat4x3 camera_matrix1,
	vec4 rolling_shutter,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	BoundingBox render_aabb,
	mat3 render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	Foveation foveation,
	Lens lens,
	Buffer2DView<const vec4> envmap,
	vec4* __restrict__ frame_buffer,
	float* __restrict__ depth_buffer,
	Buffer2DView<const uint8_t> hidden_area_mask,
	Buffer2DView<const vec2> distortion,
	ERenderMode render_mode,
	const bool use_sdf_in_nerf
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	if (plane_z < 0) {
		aperture_size = 0.0;
	}

	vec2 pixel_offset = ld_random_pixel_offset(snap_to_pixel_centers ? 0 : sample_index);
	vec2 uv = vec2{(float)x + pixel_offset.x, (float)y + pixel_offset.y} / vec2(resolution);
	float ray_time = rolling_shutter.x + rolling_shutter.y * uv.x + rolling_shutter.z * uv.y + rolling_shutter.w * ld_random_val(sample_index, idx * 72239731);
	Ray ray = uv_to_ray(
		sample_index,
		uv,
		resolution,
		focal_length,
		camera_matrix0 * ray_time + camera_matrix1 * (1.f - ray_time),
		screen_center,
		parallax_shift,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		hidden_area_mask,
		lens,
		distortion
	);

	NerfPayload& payload = payloads[idx];
	payload.max_weight = 0.0f;

	depth_buffer[idx] = MAX_DEPTH();

	if (!ray.is_valid()) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (plane_z < 0) {
		float n = length(ray.d);
		payload.origin = ray.o;
		payload.dir = (1.0f/n) * ray.d;
		payload.t = -plane_z*n;
		payload.idx = idx;
		payload.n_steps = 0;
		payload.alive = false;
		depth_buffer[idx] = -plane_z;
		return;
	}

	ray.d = normalize(ray.d);

	if (envmap) {
		frame_buffer[idx] = read_envmap(envmap, ray.d);
	}

	float t = fmaxf(render_aabb.ray_intersect(render_aabb_to_local * ray.o, render_aabb_to_local * ray.d).x, 0.0f) + 1e-6f;

	if (!render_aabb.contains(render_aabb_to_local * ray(t))) {
		payload.origin = ray.o;
		payload.alive = false;
		return;
	}

	if (render_mode == ERenderMode::Distortion) {
		vec2 offset = vec2(0.0f);
		if (distortion) {
			offset += distortion.at_lerp(vec2{(float)x + 0.5f, (float)y + 0.5f} / vec2(resolution));
		}

		frame_buffer[idx].rgb() = to_rgb(offset * 50.0f);
		frame_buffer[idx].a = 1.0f;
		depth_buffer[idx] = 1.0f;
		payload.origin = ray(MAX_DEPTH());
		payload.alive = false;
		return;
	}

	payload.origin = ray.o;
	payload.dir = ray.d;
	payload.t = t;
	payload.idx = idx;
	payload.n_steps = 0;
	payload.alive = true;

	if (use_sdf_in_nerf) {
		payload.zmin = -1.0f;
		payload.found_surface = false;
		payload.prev_sdf = 1.0f;
		payload.norm_weights = 0.0f;
	}
}

static constexpr float MIN_PDF = 0.01f;

__global__ void construct_cdf_2d(
	uint32_t n_images,
	uint32_t height,
	uint32_t width,
	const float* __restrict__ data,
	float* __restrict__ cdf_x_cond_y,
	float* __restrict__ cdf_y
) {
	const uint32_t y = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t img = threadIdx.y + blockIdx.y * blockDim.y;
	if (y >= height || img >= n_images) return;

	const uint32_t offset_xy = img * height * width + y * width;
	data += offset_xy;
	cdf_x_cond_y += offset_xy;

	float cum = 0;
	for (uint32_t x = 0; x < width; ++x) {
		cum += data[x] + 1e-10f;
		cdf_x_cond_y[x] = cum;
	}

	cdf_y[img * height + y] = cum;
	float norm = __frcp_rn(cum);

	for (uint32_t x = 0; x < width; ++x) {
		cdf_x_cond_y[x] = (1.0f - MIN_PDF) * cdf_x_cond_y[x] * norm + MIN_PDF * (float)(x+1) / (float)width;
	}
}

__global__ void construct_cdf_1d(
	uint32_t n_images,
	uint32_t height,
	float* __restrict__ cdf_y,
	float* __restrict__ cdf_img
) {
	const uint32_t img = threadIdx.x + blockIdx.x * blockDim.x;
	if (img >= n_images) return;

	cdf_y += img * height;

	float cum = 0;
	for (uint32_t y = 0; y < height; ++y) {
		cum += cdf_y[y];
		cdf_y[y] = cum;
	}

	cdf_img[img] = cum;

	float norm = __frcp_rn(cum);
	for (uint32_t y = 0; y < height; ++y) {
		cdf_y[y] = (1.0f - MIN_PDF) * cdf_y[y] * norm + MIN_PDF * (float)(y+1) / (float)height;
	}
}

__global__ void safe_divide(const uint32_t num_elements, float* __restrict__ inout, const float* __restrict__ divisor) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= num_elements) return;

	float local_divisor = divisor[i];
	inout[i] = local_divisor > 0.0f ? (inout[i] / local_divisor) : 0.0f;
}

void Testbed::NerfTracer::init_rays_from_camera(
	uint32_t sample_index,
	uint32_t padded_output_width,
	uint32_t n_extra_dims,
	const ivec2& resolution,
	const vec2& focal_length,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
	const vec2& screen_center,
	const vec3& parallax_shift,
	bool snap_to_pixel_centers,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	float near_distance,
	float plane_z,
	float aperture_size,
	const Foveation& foveation,
	const Lens& lens,
	const Buffer2DView<const vec4>& envmap,
	const Buffer2DView<const vec2>& distortion,
	vec4* frame_buffer,
	float* depth_buffer,
	const Buffer2DView<const uint8_t>& hidden_area_mask,
	const uint8_t* grid,
	int show_accel,
	uint32_t max_mip,
	float cone_angle_constant,
	ERenderMode render_mode,
	cudaStream_t stream,
	bool use_sdf_in_nerf
) {
	// Make sure we have enough memory reserved to render at the requested resolution
	size_t n_pixels = (size_t)resolution.x * resolution.y;
	enlarge(n_pixels, padded_output_width, n_extra_dims, stream);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)resolution.x, threads.x), div_round_up((uint32_t)resolution.y, threads.y), 1 };
	init_rays_with_payload_kernel_nerf<<<blocks, threads, 0, stream>>>(
		sample_index,
		m_rays[0].payload,
		resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		render_aabb,
		render_aabb_to_local,
		near_distance,
		plane_z,
		aperture_size,
		foveation,
		lens,
		envmap,
		frame_buffer,
		depth_buffer,
		hidden_area_mask,
		distortion,
		render_mode,
		use_sdf_in_nerf
	);

	m_n_rays_initialized = resolution.x * resolution.y;

	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].rgba, 0, m_n_rays_initialized * sizeof(vec4), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(m_rays[0].depth, 0, m_n_rays_initialized * sizeof(float), stream));

	linear_kernel(advance_pos_nerf_kernel, 0, stream,
		m_n_rays_initialized,
		render_aabb,
		render_aabb_to_local,
		camera_matrix1[2],
		focal_length,
		sample_index,
		m_rays[0].payload,
		grid,
		(show_accel >= 0) ? show_accel : 0,
		max_mip,
		cone_angle_constant
	);
}

uint32_t Testbed::NerfTracer::trace(
	NerfNetwork<network_precision_t>& network,
	const BoundingBox& render_aabb,
	const mat3& render_aabb_to_local,
	const BoundingBox& train_aabb,
	const vec2& focal_length,
	float cone_angle_constant,
	const uint8_t* grid,
	ERenderMode render_mode,
	const mat4x3 &camera_matrix,
	float depth_scale,
	int visualized_layer,
	int visualized_dim,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	int show_accel,
	uint32_t max_mip,
	float min_transmittance,
	float glow_y_cutoff,
	int glow_mode,
	const float* extra_dims_gpu,
	cudaStream_t stream,
	//SDF related vars
    const bool use_sdf_in_nerf,
    const bool use_stylesdf_in_nerf,
    const bool use_volsdf_in_nerf,
    const bool use_coslam_sdf_in_nerf,
    float truncation_distance,
    float sdf_beta,
	//Custom SLAIM sampling related vars
	const bool use_custom_ray_marching,
	const float dt_for_regular_sampling
) {
	if (m_n_rays_initialized == 0) {
		return 0;
	}

	CUDA_CHECK_THROW(cudaMemsetAsync(m_hit_counter, 0, sizeof(uint32_t), stream));

	uint32_t n_alive = m_n_rays_initialized;
	// m_n_rays_initialized = 0;

	uint32_t i = 1;
	uint32_t double_buffer_index = 0;
	while (i < MARCH_ITER) {
		RaysNerfSoa& rays_current = m_rays[(double_buffer_index + 1) % 2];
		RaysNerfSoa& rays_tmp = m_rays[double_buffer_index % 2];
		++double_buffer_index;

		// Compact rays that did not diverge yet
		{
			CUDA_CHECK_THROW(cudaMemsetAsync(m_alive_counter, 0, sizeof(uint32_t), stream));
			linear_kernel(compact_kernel_nerf, 0, stream,
				n_alive,
				rays_tmp.rgba, rays_tmp.depth, rays_tmp.payload,
				rays_current.rgba, rays_current.depth, rays_current.payload,
				m_rays_hit.rgba, m_rays_hit.depth, m_rays_hit.payload,
				m_alive_counter, m_hit_counter
			);
			CUDA_CHECK_THROW(cudaMemcpyAsync(&n_alive, m_alive_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
		}

		if (n_alive == 0) {
			break;
		}

		// Want a large number of queries to saturate the GPU and to ensure compaction doesn't happen toooo frequently.
		uint32_t target_n_queries = 2 * 1024 * 1024;
		uint32_t n_steps_between_compaction = tcnn::clamp(target_n_queries / n_alive, (uint32_t)MIN_STEPS_INBETWEEN_COMPACTION, (uint32_t)MAX_STEPS_INBETWEEN_COMPACTION);

		uint32_t extra_stride = network.n_extra_dims() * sizeof(float);
		PitchedPtr<NerfCoordinate> input_data((NerfCoordinate*)m_network_input, 1, 0, extra_stride);

			linear_kernel(generate_next_nerf_network_inputs, 0, stream,
				n_alive,
				render_aabb,
				render_aabb_to_local,
				train_aabb,
				focal_length,
				camera_matrix[2],
				rays_current.payload,
				input_data,
				n_steps_between_compaction,
				grid,
				(show_accel>=0) ? show_accel : 0,
				max_mip,
				cone_angle_constant,
				extra_dims_gpu,
				//Custom Ray marching
				use_custom_ray_marching,
	 			dt_for_regular_sampling
			);

		uint32_t n_elements = next_multiple(n_alive * n_steps_between_compaction, tcnn::batch_size_granularity);
		GPUMatrix<float> positions_matrix((float*)m_network_input, (sizeof(NerfCoordinate) + extra_stride) / sizeof(float), n_elements);
		GPUMatrix<network_precision_t, RM> rgbsigma_matrix((network_precision_t*)m_network_output, network.padded_output_width(), n_elements);
		network.inference_mixed_precision(stream, positions_matrix, rgbsigma_matrix);

		if (render_mode == ERenderMode::Normals) {
			network.input_gradient(stream, 3, positions_matrix, positions_matrix);
		} else if (render_mode == ERenderMode::EncodingVis) {
			network.visualize_activation(stream, visualized_layer, visualized_dim, positions_matrix, positions_matrix);
		}

        if (use_sdf_in_nerf) {
		    linear_kernel(composite_kernel_nerf_sdf, 0, stream,
		    	n_alive,
		    	n_elements,
		    	i,
		    	train_aabb,
		    	glow_y_cutoff,
		    	glow_mode,
		    	camera_matrix,
		    	focal_length,
		    	depth_scale,
		    	rays_current.rgba,
		    	rays_current.depth,
		    	rays_current.payload,
		    	input_data,
		    	m_network_output,
		    	network.padded_output_width(),
		    	n_steps_between_compaction,
		    	render_mode,
		    	nullptr,
		    	rgb_activation,
		    	density_activation,
		    	show_accel,
		    	min_transmittance,
		    	truncation_distance
		    );
        } else {
		    linear_kernel(composite_kernel_nerf, 0, stream,
		    	n_alive,
		    	n_elements,
		    	i,
		    	train_aabb,
		    	glow_y_cutoff,
		    	glow_mode,
		    	camera_matrix,
		    	focal_length,
		    	depth_scale,
		    	rays_current.rgba,
		    	rays_current.depth,
		    	rays_current.payload,
		    	input_data,
		    	m_network_output,
		    	network.padded_output_width(),
		    	n_steps_between_compaction,
		    	render_mode,
		    	grid,
		    	rgb_activation,
		    	density_activation,
		    	show_accel,
		    	min_transmittance,
    			use_stylesdf_in_nerf,
    			use_volsdf_in_nerf,
    			use_coslam_sdf_in_nerf,
				sdf_beta,
				truncation_distance
		    );
        }

		i += n_steps_between_compaction;
	}

	uint32_t n_hit;
	CUDA_CHECK_THROW(cudaMemcpyAsync(&n_hit, m_hit_counter, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	return n_hit;
}

void Testbed::NerfTracer::enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream) {
	n_elements = next_multiple(n_elements, size_t(tcnn::batch_size_granularity));
	size_t num_floats = sizeof(NerfCoordinate) / sizeof(float) + n_extra_dims;
	auto scratch = allocate_workspace_and_distribute<
		vec4, float, NerfPayload, // m_rays[0]
		vec4, float, NerfPayload, // m_rays[1]
		vec4, float, NerfPayload, // m_rays_hit

		network_precision_t,
		float,
		uint32_t,
		uint32_t
	>(
		stream, &m_scratch_alloc,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements, n_elements, n_elements,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * padded_output_width,
		n_elements * MAX_STEPS_INBETWEEN_COMPACTION * num_floats,
		32, // 2 full cache lines to ensure no overlap
		32  // 2 full cache lines to ensure no overlap
	);

	m_rays[0].set(std::get<0>(scratch), std::get<1>(scratch), std::get<2>(scratch), n_elements);
	m_rays[1].set(std::get<3>(scratch), std::get<4>(scratch), std::get<5>(scratch), n_elements);
	m_rays_hit.set(std::get<6>(scratch), std::get<7>(scratch), std::get<8>(scratch), n_elements);

	m_network_output = std::get<9>(scratch);
	m_network_input = std::get<10>(scratch);

	m_hit_counter = std::get<11>(scratch);
	m_alive_counter = std::get<12>(scratch);
}

void Testbed::Nerf::Training::reset_extra_dims(default_rng_t& rng) {
	uint32_t n_extra_dims = dataset.n_extra_dims();
	std::vector<float> extra_dims_cpu(n_extra_dims * (dataset.n_images + 1)); // n_images + 1 since we use an extra 'slot' for the inference latent code
	float* dst = extra_dims_cpu.data();
	extra_dims_opt = std::vector<VarAdamOptimizer>(dataset.n_images, VarAdamOptimizer(n_extra_dims, 1e-4f));
	for (uint32_t i = 0; i < dataset.n_images; ++i) {
		vec3 light_dir = warp_direction(normalize(dataset.metadata[i].light_dir));
		extra_dims_opt[i].reset_state();
		std::vector<float>& optimzer_value = extra_dims_opt[i].variable();
		for (uint32_t j = 0; j < n_extra_dims; ++j) {
			if (dataset.has_light_dirs && j < 3) {
				dst[j] = light_dir[j];
			} else {
				dst[j] = random_val(rng) * 2.0f - 1.0f;
			}
			optimzer_value[j] = dst[j];
		}
		dst += n_extra_dims;
	}
	extra_dims_gpu.resize_and_copy_from_host(extra_dims_cpu);
}

void Testbed::Nerf::Training::update_extra_dims() {
	uint32_t n_extra_dims = dataset.n_extra_dims();
	std::vector<float> extra_dims_cpu(extra_dims_gpu.size());
	for (uint32_t i = 0; i < extra_dims_opt.size(); ++i) {
		const std::vector<float>& value = extra_dims_opt[i].variable();
		for (uint32_t j = 0; j < n_extra_dims; ++j) {
			extra_dims_cpu[i * n_extra_dims + j] = value[j];
		}
	}

	CUDA_CHECK_THROW(cudaMemcpyAsync(extra_dims_gpu.data(), extra_dims_cpu.data(), extra_dims_opt.size() * n_extra_dims * sizeof(float), cudaMemcpyHostToDevice));
}

const float* Testbed::get_inference_extra_dims(cudaStream_t stream) const {
	if (m_nerf_network->n_extra_dims() == 0) {
		return nullptr;
	}
	const float* extra_dims_src = m_nerf.training.extra_dims_gpu.data() + m_nerf.extra_dim_idx_for_inference * m_nerf.training.dataset.n_extra_dims();
	if (!m_nerf.training.dataset.has_light_dirs) {
		return extra_dims_src;
	}

	// the dataset has light directions, so we must construct a temporary buffer and fill it as requested.
	// we use an extra 'slot' that was pre-allocated for us at the end of the extra_dims array.
	size_t size = m_nerf_network->n_extra_dims() * sizeof(float);
	float* dims_gpu = m_nerf.training.extra_dims_gpu.data() + m_nerf.training.dataset.n_images * m_nerf.training.dataset.n_extra_dims();
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, extra_dims_src, size, cudaMemcpyDeviceToDevice, stream));
	vec3 light_dir = warp_direction(normalize(m_nerf.light_dir));
	CUDA_CHECK_THROW(cudaMemcpyAsync(dims_gpu, &light_dir, min(size, sizeof(vec3)), cudaMemcpyHostToDevice, stream));
	return dims_gpu;
}

void Testbed::render_nerf(
	cudaStream_t stream,
	const CudaRenderBufferView& render_buffer,
	NerfNetwork<precision_t>& nerf_network,
	const uint8_t* density_grid_bitfield,
	const vec2& focal_length,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec4& rolling_shutter,
	const vec2& screen_center,
	const Foveation& foveation,
	int visualized_dimension
) {
	float plane_z = m_slice_plane_z + m_scale;
	if (m_render_mode == ERenderMode::Slice) {
		plane_z = -plane_z;
	}

	ERenderMode render_mode = visualized_dimension > -1 ? ERenderMode::EncodingVis : m_render_mode;

	const float* extra_dims_gpu = get_inference_extra_dims(stream);

	NerfTracer tracer;

	// Our motion vector code can't undo grid distortions -- so don't render grid distortion if DLSS is enabled
	auto grid_distortion = m_nerf.render_with_lens_distortion && !m_dlss ? m_distortion.inference_view() : Buffer2DView<const vec2>{};
	Lens lens = m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{};

	auto resolution = render_buffer.resolution;

	// very hacky solution
	// in SDF mode densities grids don't make sense anymore -> don't use them
	tracer.init_rays_from_camera(
		render_buffer.spp,
		nerf_network.padded_output_width(),
		nerf_network.n_extra_dims(),
		render_buffer.resolution,
		focal_length,
		camera_matrix0,
		camera_matrix1,
		rolling_shutter,
		screen_center,
		m_parallax_shift,
		m_snap_to_pixel_centers,
		m_render_aabb,
		m_render_aabb_to_local,
		m_render_near_distance,
		plane_z,
		m_aperture_size,
		foveation,
		lens,
		m_envmap.inference_view(),
		grid_distortion,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		render_buffer.hidden_area_mask ? render_buffer.hidden_area_mask->const_view() : Buffer2DView<const uint8_t>{},
		((!m_use_sdf_in_nerf) && m_use_density_in_nerf_sampling) ? density_grid_bitfield: nullptr,
		m_nerf.show_accel,
		m_nerf.max_cascade,
		m_nerf.cone_angle_constant,
		render_mode,
		stream,
		(m_use_sdf_in_nerf||m_use_coslam_sdf_in_nerf)
	);

	uint32_t n_hit;
	if (m_render_mode == ERenderMode::Slice) {
		n_hit = tracer.n_rays_initialized();
	} else {
		float depth_scale = 1.0f / m_nerf.training.dataset.scale;
		n_hit = tracer.trace(
			nerf_network,
			m_render_aabb,
			m_render_aabb_to_local,
			m_aabb,
			focal_length,
			m_nerf.cone_angle_constant,
			//density_grid_bitfield,
			((!m_use_sdf_in_nerf) && m_use_density_in_nerf_sampling) ? density_grid_bitfield: nullptr,
			render_mode,
			camera_matrix1,
			depth_scale,
			m_visualized_layer,
			visualized_dimension,
			m_nerf.rgb_activation,
			m_nerf.density_activation,
			m_nerf.show_accel,
			m_nerf.max_cascade,
			m_nerf.render_min_transmittance,
			m_nerf.glow_y_cutoff,
			m_nerf.glow_mode,
			extra_dims_gpu,
			stream,
			//SDF
            m_use_sdf_in_nerf,
            m_use_stylesdf_in_nerf,
            m_use_volsdf_in_nerf,
            m_use_coslam_sdf_in_nerf,
		    m_nerf.training.truncation_distance,
			m_nerf.training.volsdf_beta,
			//Custom Ray marching,
			m_use_custom_ray_marching,
			m_nerf.training.dt_for_regular_sampling
		);
	}
	RaysNerfSoa& rays_hit = m_render_mode == ERenderMode::Slice ? tracer.rays_init() : tracer.rays_hit();

	if (m_render_mode == ERenderMode::Slice) {
		// Store colors in the normal buffer
		uint32_t n_elements = next_multiple(n_hit, tcnn::batch_size_granularity);
		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + nerf_network.n_extra_dims();
		const uint32_t extra_stride = nerf_network.n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

		GPUMatrix<float> positions_matrix{floats_per_coord, n_elements, stream};
		GPUMatrix<float> rgbsigma_matrix{4, n_elements, stream};

		linear_kernel(generate_nerf_network_inputs_at_current_position, 0, stream, n_hit, m_aabb, rays_hit.payload, PitchedPtr<NerfCoordinate>((NerfCoordinate*)positions_matrix.data(), 1, 0, extra_stride), extra_dims_gpu );

		if (visualized_dimension == -1) {
			nerf_network.inference(stream, positions_matrix, rgbsigma_matrix);
			linear_kernel(compute_nerf_rgba_kernel, 0, stream, n_hit, (vec4*)rgbsigma_matrix.data(), m_nerf.rgb_activation, m_nerf.density_activation, 0.01f, false);
		} else {
			nerf_network.visualize_activation(stream, m_visualized_layer, visualized_dimension, positions_matrix, rgbsigma_matrix);
		}

		linear_kernel(shade_kernel_nerf, 0, stream,
			n_hit,
			(vec4*)rgbsigma_matrix.data(),
			nullptr,
			rays_hit.payload,
			m_render_mode,
			m_nerf.training.linear_colors,
			render_buffer.frame_buffer,
			render_buffer.depth_buffer,
			m_use_sdf_in_nerf || m_use_coslam_sdf_in_nerf
		);
		return;
	}

	linear_kernel(shade_kernel_nerf, 0, stream,
		n_hit,
		rays_hit.rgba,
		rays_hit.depth,
		rays_hit.payload,
		m_render_mode,
		m_nerf.training.linear_colors,
		render_buffer.frame_buffer,
		render_buffer.depth_buffer,
		m_use_sdf_in_nerf || m_use_coslam_sdf_in_nerf
	);

	if (render_mode == ERenderMode::Cost) {
		std::vector<NerfPayload> payloads_final_cpu(n_hit);
		CUDA_CHECK_THROW(cudaMemcpyAsync(payloads_final_cpu.data(), rays_hit.payload, n_hit * sizeof(NerfPayload), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

		size_t total_n_steps = 0;
		for (uint32_t i = 0; i < n_hit; ++i) {
			total_n_steps += payloads_final_cpu[i].n_steps;
		}
		tlog::info() << "Total steps per hit= " << total_n_steps << "/" << n_hit << " = " << ((float)total_n_steps/(float)n_hit);
	}
}

void Testbed::Nerf::Training::set_camera_intrinsics(int frame_idx, float fx, float fy, float cx, float cy, float k1, float k2, float p1, float p2, float k3, float k4, bool is_fisheye) {
	if (frame_idx < 0 || frame_idx >= dataset.n_images) {
		return;
	}
	if (fx <= 0.f) fx = fy;
	if (fy <= 0.f) fy = fx;
	auto& m = dataset.metadata[frame_idx];
	if (cx < 0.f) cx = -cx; else cx = cx / m.resolution.x;
	if (cy < 0.f) cy = -cy; else cy = cy / m.resolution.y;
	m.lens = { ELensMode::Perspective };
	if (k1 || k2 || k3 || k4 || p1 || p2) {
		if (is_fisheye) {
			m.lens = { ELensMode::OpenCVFisheye, k1, k2, k3, k4 };
		} else {
			m.lens = { ELensMode::OpenCV, k1, k2, p1, p2 };
		}
	}

	m.principal_point = { cx, cy };
	m.focal_length = { fx, fy };
	dataset.update_metadata(frame_idx, frame_idx + 1);
}

void Testbed::Nerf::Training::set_camera_extrinsics_rolling_shutter(int frame_idx, mat4x3 camera_to_world_start, mat4x3 camera_to_world_end, const vec4& rolling_shutter, bool convert_to_ngp) {
	if (frame_idx < 0 || frame_idx >= dataset.n_images) {
		return;
	}

	if (convert_to_ngp) {
		camera_to_world_start = dataset.nerf_matrix_to_ngp(camera_to_world_start);
		camera_to_world_end = dataset.nerf_matrix_to_ngp(camera_to_world_end);
	}

	dataset.xforms[frame_idx].start = camera_to_world_start;
	dataset.xforms[frame_idx].end = camera_to_world_end;
	dataset.metadata[frame_idx].rolling_shutter = rolling_shutter;
	dataset.update_metadata(frame_idx, frame_idx + 1);

	cam_rot_offset[frame_idx].reset_state();
	cam_pos_offset[frame_idx].reset_state();
	cam_exposure[frame_idx].reset_state();
	update_transforms(frame_idx, frame_idx + 1);
}

void Testbed::Nerf::Training::set_camera_extrinsics(int frame_idx, mat4x3 camera_to_world, bool convert_to_ngp) {
	set_camera_extrinsics_rolling_shutter(frame_idx, camera_to_world, camera_to_world, vec4(0.0f), convert_to_ngp);
}

void Testbed::Nerf::Training::reset_camera_extrinsics() {
	for (auto&& opt : cam_rot_offset) {
		opt.reset_state();
	}

	for (auto&& opt : cam_pos_offset) {
		opt.reset_state();
	}

	for (auto&& opt : cam_exposure) {
		opt.reset_state();
	}
}

void Testbed::Nerf::Training::export_camera_extrinsics(const fs::path& path, bool export_extrinsics_in_quat_format) {
	tlog::info() << "Saving a total of " << n_images_for_training << " poses to " << path.str();
	nlohmann::json trajectory;
	for(int i = 0; i < n_images_for_training; ++i) {
		nlohmann::json frame{{"id", i}};

		const mat4x3 p_nerf = get_camera_extrinsics(i);
		if (export_extrinsics_in_quat_format) {
			// Assume 30 fps
			frame["time"] =  i*0.033f;
			// Convert the pose from NeRF to Quaternion format.
			const mat3 conv_coords_l{
				 0.f,   0.f,  -1.f,
				 1.f,   0.f,   0.f,
				 0.f,  -1.f,   0.f,
			};
			const mat4 conv_coords_r{
				1.f,  0.f,  0.f,  0.f,
				0.f, -1.f,  0.f,  0.f,
				0.f,  0.f, -1.f,  0.f,
				0.f,  0.f,  0.f,  1.f,
			};
			const mat4x3 p_quat = conv_coords_l * p_nerf * conv_coords_r;

			const quat rot_q = mat3(p_quat);
			frame["q"] = rot_q;
			frame["t"] = p_quat[3];
		} else {
			frame["transform_matrix"] = p_nerf;
		}

		trajectory.emplace_back(frame);
	}

	std::ofstream file{native_string(path)};
	file << std::setw(2) << trajectory << std::endl;
}

mat4x3 Testbed::Nerf::Training::get_camera_extrinsics(int frame_idx) {
	if (frame_idx < 0 || frame_idx >= dataset.n_images) {
		return mat4x3(1.0f);
	}
	return dataset.ngp_matrix_to_nerf(transforms[frame_idx].start);
}

void Testbed::Nerf::Training::update_transforms(int first, int last) {
	if (last < 0) {
		last = dataset.n_images;
	}

	if (last > dataset.n_images) {
		last = dataset.n_images;
	}

	int n = last - first;
	if (n <= 0) {
		return;
	}

	if (transforms.size() < last) {
		transforms.resize(last);
	}

	for (uint32_t i = 0; i < n; ++i) {
		auto xform = dataset.xforms[i + first];
		float det_start = determinant(mat3(xform.start));
		float det_end = determinant(mat3(xform.end));
		if (distance(det_start, 1.0f) > 0.01f || distance(det_end, 1.0f) > 0.01f) {
			tlog::warning() << "Rotation of camera matrix in frame " << i + first << " has a scaling component (determinant!=1).";
			tlog::warning() << "Normalizing the matrix. This hints at an issue in your data generation pipeline and should be fixed.";

			xform.start[0] /= std::cbrt(det_start); xform.start[1] /= std::cbrt(det_start); xform.start[2] /= std::cbrt(det_start);
			xform.end[0]   /= std::cbrt(det_end);   xform.end[1]   /= std::cbrt(det_end);   xform.end[2]   /= std::cbrt(det_end);
			dataset.xforms[i + first] = xform;
		}

		mat3 rot = rotmat(cam_rot_offset[i + first].variable());
		auto rot_start = rot * mat3(xform.start);
		auto rot_end = rot * mat3(xform.end);
		xform.start = mat4x3(rot_start[0], rot_start[1], rot_start[2], xform.start[3]);
		xform.end = mat4x3(rot_end[0], rot_end[1], rot_end[2], xform.end[3]);

		xform.start[3] += cam_pos_offset[i + first].variable();
		xform.end[3] += cam_pos_offset[i + first].variable();
		transforms[i + first] = xform;
	}

	transforms_gpu.enlarge(last);
	CUDA_CHECK_THROW(cudaMemcpy(transforms_gpu.data() + first, transforms.data() + first, n * sizeof(TrainingXForm), cudaMemcpyHostToDevice));
}

void Testbed::create_empty_nerf_dataset(size_t n_images, int aabb_scale, bool is_hdr) {
	m_data_path = {};
	set_mode(ETestbedMode::Nerf);
	m_nerf.training.dataset = ngp::create_empty_nerf_dataset(n_images, aabb_scale, is_hdr);
	load_nerf(m_data_path);
	m_nerf.training.n_images_for_training = 0;
	m_training_data_available = true;
}

void Testbed::load_nerf_post() { // moved the second half of load_nerf here
	m_nerf.rgb_activation = m_nerf.training.dataset.is_hdr ? ENerfActivation::Exponential : ENerfActivation::Logistic;

	m_nerf.training.n_images_for_training = (int)m_nerf.training.dataset.n_images;

	m_nerf.training.dataset.update_metadata();

	m_nerf.training.cam_pos_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_pos_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_pos_gradient);

	m_nerf.training.cam_exposure.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-3f));
	m_nerf.training.cam_pos_offset.resize(m_nerf.training.dataset.n_images, AdamOptimizer<vec3>(1e-4f));
	m_nerf.training.cam_rot_offset.resize(m_nerf.training.dataset.n_images, RotationAdamOptimizer(1e-4f));
	m_nerf.training.cam_focal_length_offset = AdamOptimizer<vec2>(1e-5f);

	m_nerf.training.cam_rot_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_rot_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_rot_gradient);

	m_nerf.training.cam_exposure_gradient.resize(m_nerf.training.dataset.n_images, vec3(0.0f));
	m_nerf.training.cam_exposure_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);
	m_nerf.training.cam_exposure_gradient_gpu.resize_and_copy_from_host(m_nerf.training.cam_exposure_gradient);

	m_nerf.training.cam_focal_length_gradient = vec2(0.0f);
	m_nerf.training.cam_focal_length_gradient_gpu.resize_and_copy_from_host(&m_nerf.training.cam_focal_length_gradient, 1);

	m_nerf.training.reset_extra_dims(m_rng);
	m_nerf.training.optimize_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0;

	if (m_nerf.training.dataset.has_rays) {
		m_nerf.training.near_distance = 0.0f;
	}

	// Perturbation of the training cameras -- for debugging the online extrinsics learning code
	// float perturb_amount = 0.0f;
	// if (perturb_amount > 0.f) {
	// 	for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
	// 		vec3 rot = random_val_3d(m_rng) * perturb_amount;
	// 		float angle = rot.norm();
	// 		rot /= angle;
	// 		auto trans = random_val_3d(m_rng);
	// 		m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].start.block<3,3>(0,0);
	// 		m_nerf.training.dataset.xforms[i].start[3] += trans * perturb_amount;
	// 		m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0) = AngleAxisf(angle, rot).matrix() * m_nerf.training.dataset.xforms[i].end.block<3,3>(0,0);
	// 		m_nerf.training.dataset.xforms[i].end[3] += trans * perturb_amount;
	// 	}
	// }

	m_nerf.training.update_transforms();

	if (!m_nerf.training.dataset.metadata.empty()) {
		m_nerf.render_lens = m_nerf.training.dataset.metadata[0].lens;
		m_screen_center = vec2(1.f) - m_nerf.training.dataset.metadata[0].principal_point;
	}

	if (!is_pot(m_nerf.training.dataset.aabb_scale)) {
		throw std::runtime_error{fmt::format("NeRF dataset's `aabb_scale` must be a power of two, but is {}.", m_nerf.training.dataset.aabb_scale)};
	}

	int max_aabb_scale = 1 << (NERF_CASCADES()-1);
	if (m_nerf.training.dataset.aabb_scale > max_aabb_scale) {
		throw std::runtime_error{fmt::format(
			"NeRF dataset must have `aabb_scale <= {}`, but is {}. "
			"You can increase this limit by factors of 2 by incrementing `NERF_CASCADES()` and re-compiling.",
			max_aabb_scale, m_nerf.training.dataset.aabb_scale
		)};
	}

	m_aabb = BoundingBox{vec3(0.5f), vec3(0.5f)};
	m_aabb.inflate(0.5f * std::min(1 << (NERF_CASCADES()-1), m_nerf.training.dataset.aabb_scale));
	m_raw_aabb = m_aabb;
	m_render_aabb = m_aabb;
	m_render_aabb_to_local = m_nerf.training.dataset.render_aabb_to_local;
	if (!m_nerf.training.dataset.render_aabb.is_empty()) {
		m_render_aabb = m_nerf.training.dataset.render_aabb.intersection(m_aabb);
	}

	m_nerf.max_cascade = 0;
	while ((1 << m_nerf.max_cascade) < m_nerf.training.dataset.aabb_scale) {
		++m_nerf.max_cascade;
	}

	// Perform fixed-size stepping in unit-cube scenes (like original NeRF) and exponential
	// stepping in larger scenes.
	m_nerf.cone_angle_constant = m_nerf.training.dataset.aabb_scale <= 1 ? 0.0f : (1.0f / 256.0f);

	m_up_dir = m_nerf.training.dataset.up;
}

void Testbed::load_nerf(const fs::path& data_path) {
	if (!data_path.empty()) {
		std::vector<fs::path> json_paths;
		if (data_path.is_directory()) {
			for (const auto& path : fs::directory{data_path}) {
				if (path.is_file() && equals_case_insensitive(path.extension(), "json")) {
					json_paths.emplace_back(path);
				}
			}
		} else if (equals_case_insensitive(data_path.extension(), "json")) {
			json_paths.emplace_back(data_path);
		} else {
			throw std::runtime_error{"NeRF data path must either be a json file or a directory containing json files."};
		}

		const auto prev_aabb_scale = m_nerf.training.dataset.aabb_scale;

		if(m_keep_data_on_cpu){
			m_nerf.training.dataset = ngp::load_nerf_cpu(json_paths, m_nerf.sharpen);
		} else {
			m_nerf.training.dataset = ngp::load_nerf(json_paths, m_nerf.sharpen);
		}

		// Check if the NeRF network has been previously configured.
		// If it has not, don't reset it.
		if (m_nerf.training.dataset.aabb_scale != prev_aabb_scale && m_nerf_network) {
			// The AABB scale affects network size indirectly. If it changed after loading,
			// we need to reset the previously configured network to keep a consistent internal state.
			reset_network();
		}
	}

	load_nerf_post();
}

void Testbed::update_density_grid_nerf(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);

	m_nerf.density_grid.resize(n_elements);

	const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		NerfPosition,       // positions at which the NN will be queried for density evaluation
		uint32_t,           // indices of corresponding density grid cells
		float,              // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
		network_precision_t // output of the MLP before being converted to densities.
	>(stream, &alloc, n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

	NerfPosition* density_grid_positions = std::get<0>(scratch);
	uint32_t* density_grid_indices = std::get<1>(scratch);
	float* density_grid_tmp = std::get<2>(scratch);
	network_precision_t* mlp_out = std::get<3>(scratch);

	if (m_training_step == 0 || m_nerf.training.n_images_for_training != m_nerf.training.n_images_for_training_prev) {
		m_nerf.training.n_images_for_training_prev = m_nerf.training.n_images_for_training;
		if (m_training_step == 0) {
			m_nerf.density_grid_ema_step = 0;
		}
		// Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
		if (!m_nerf.training.dataset.has_rays) {
			linear_kernel(mark_untrained_density_grid, 0, stream, n_elements, m_nerf.density_grid.data(),
				m_nerf.training.n_images_for_training,
				m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
				m_training_step == 0
			);
		} else {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
		}
	}

	uint32_t n_steps = 1;
	for (uint32_t i = 0; i < n_steps; ++i) {
		CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float)*n_elements, stream));

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_uniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions,
			density_grid_indices,
			m_nerf.max_cascade+1,
			-0.01f
		);
		m_nerf.training.density_grid_rng.advance();

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_nonuniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions+n_uniform_density_grid_samples,
			density_grid_indices+n_uniform_density_grid_samples,
			m_nerf.max_cascade+1,
			NERF_MIN_OPTICAL_THICKNESS()
		);
		m_nerf.training.density_grid_rng.advance();

		// Evaluate density at the spawned locations in batches.
		// Otherwise, we can exhaust the maximum index range of cutlass
		size_t batch_size = NERF_GRID_N_CELLS() * 2;

		for (size_t i = 0; i < n_density_grid_samples; i += batch_size) {
			batch_size = std::min(batch_size, n_density_grid_samples - i);

			GPUMatrix<network_precision_t, RM> density_matrix(mlp_out + i, padded_output_width, batch_size);
			GPUMatrix<float> density_grid_position_matrix((float*)(density_grid_positions + i), sizeof(NerfPosition)/sizeof(float), batch_size);
			m_nerf_network->density(stream, density_grid_position_matrix, density_matrix, false);
		}

		linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream, n_density_grid_samples, density_grid_indices, mlp_out, density_grid_tmp, m_nerf.rgb_activation, m_nerf.density_activation, m_use_volsdf_in_nerf, m_use_stylesdf_in_nerf, m_use_coslam_sdf_in_nerf, m_nerf.training.volsdf_beta);
		linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, decay, m_nerf.density_grid_ema_step, m_nerf.density_grid.data(), density_grid_tmp);

		++m_nerf.density_grid_ema_step;
	}

	update_density_grid_mean_and_bitfield(stream);
}

void Testbed::update_density_grid_mean_and_bitfield_slam(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS();

	size_t size_including_mips = grid_mip_offset(NERF_CASCADES())/8;
	m_nerf.density_grid_bitfield.enlarge(size_including_mips);
	m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
	reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

	linear_kernel(grid_to_bitfield_slam, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data(), m_nerf.density_grid_unvisible_regions_bool.data());

	for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
		linear_kernel(bitfield_max_pool, 0, stream, n_elements/64, m_nerf.get_density_grid_bitfield_mip(level-1), m_nerf.get_density_grid_bitfield_mip(level));
	}

	set_all_devices_dirty();
}


void Testbed::update_density_grid_mean_and_bitfield(cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS();

	size_t size_including_mips = grid_mip_offset(NERF_CASCADES())/8;
	m_nerf.density_grid_bitfield.enlarge(size_including_mips);
	m_nerf.density_grid_mean.enlarge(reduce_sum_workspace_size(n_elements));

	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_mean.data(), 0, sizeof(float), stream));
	reduce_sum(m_nerf.density_grid.data(), [n_elements] __device__ (float val) { return fmaxf(val, 0.f) / (n_elements); }, m_nerf.density_grid_mean.data(), n_elements, stream);

	linear_kernel(grid_to_bitfield, 0, stream, n_elements/8 * NERF_CASCADES(), n_elements/8 * (m_nerf.max_cascade + 1), m_nerf.density_grid.data(), m_nerf.density_grid_bitfield.data(), m_nerf.density_grid_mean.data());

	for (uint32_t level = 1; level < NERF_CASCADES(); ++level) {
		linear_kernel(bitfield_max_pool, 0, stream, n_elements/64, m_nerf.get_density_grid_bitfield_mip(level-1), m_nerf.get_density_grid_bitfield_mip(level));
	}

	set_all_devices_dirty();
}

__global__ void mark_density_grid_in_sphere_empty_kernel(const uint32_t n_elements, float* density_grid, vec3 pos, float radius) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	// Random position within that cellq
	uint32_t level = i / NERF_GRID_N_CELLS();
	uint32_t pos_idx = i % NERF_GRID_N_CELLS();

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	float cell_radius = scalbnf(SQRT3(), level) / NERF_GRIDSIZE();
	vec3 cell_pos = ((vec3{(float)x+0.5f, (float)y+0.5f, (float)z+0.5f}) / (float)NERF_GRIDSIZE() - vec3(0.5f)) * scalbnf(1.0f, level) + vec3(0.5f);

	// Disable if the cell touches the sphere (conservatively, by bounding the cell with a sphere)
	if (distance(pos, cell_pos) < radius + cell_radius) {
		density_grid[i] = -1.0f;
	}
}

void Testbed::mark_density_grid_in_sphere_empty(const vec3& pos, float radius, cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);
	if (m_nerf.density_grid.size() != n_elements) {
		return;
	}

	linear_kernel(mark_density_grid_in_sphere_empty_kernel, 0, stream, n_elements, m_nerf.density_grid.data(), pos, radius);

	update_density_grid_mean_and_bitfield(stream);
}

void Testbed::NerfCounters::prepare_for_training_steps(cudaStream_t stream) {
	numsteps_counter.enlarge(1);
	numsteps_counter_compacted.enlarge(1);
	loss.enlarge(rays_per_batch);
	CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
	CUDA_CHECK_THROW(cudaMemsetAsync(numsteps_counter_compacted.data(), 0, sizeof(uint32_t), stream)); // clear the counter in the first slot
	CUDA_CHECK_THROW(cudaMemsetAsync(loss.data(), 0, sizeof(float)*rays_per_batch, stream));
}

float Testbed::NerfCounters::update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	std::vector<uint32_t> counter_cpu(1);
	std::vector<uint32_t> compacted_counter_cpu(1);
	numsteps_counter.copy_to_host(counter_cpu);
	numsteps_counter_compacted.copy_to_host(compacted_counter_cpu);
	measured_batch_size = 0;
	measured_batch_size_before_compaction = 0;

	if (counter_cpu[0] == 0 || compacted_counter_cpu[0] == 0) {
		return 0.f;
	}

	measured_batch_size_before_compaction = counter_cpu[0];
	measured_batch_size = compacted_counter_cpu[0];

	float loss_scalar = 0.0;
	if (get_loss_scalar) {
		loss_scalar = reduce_sum(loss.data(), rays_per_batch, stream) * (float)measured_batch_size / (float)target_batch_size;
	}

	rays_per_batch = (uint32_t)((float)rays_per_batch * (float)target_batch_size / (float)measured_batch_size);
	rays_per_batch = std::min(next_multiple(rays_per_batch, tcnn::batch_size_granularity), 1u << 18);

	return loss_scalar;
}

void Testbed::train_nerf(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	if (m_nerf.training.n_images_for_training == 0) {
		return;
	}

	if (m_nerf.training.include_sharpness_in_error) {
		size_t n_cells = NERF_GRID_N_CELLS() * NERF_CASCADES();
		if (m_nerf.training.sharpness_grid.size() < n_cells) {
			m_nerf.training.sharpness_grid.enlarge(NERF_GRID_N_CELLS() * NERF_CASCADES());
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		}

		if (m_training_step == 0) {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		} else {
			linear_kernel(decay_sharpness_grid_nerf, 0, stream, m_nerf.training.sharpness_grid.size(), 0.95f, m_nerf.training.sharpness_grid.data());
		}
	}
	m_nerf.training.counters_rgb.prepare_for_training_steps(stream);

	if (m_nerf.training.n_steps_since_cam_update == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(), 0, m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradient_weights(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(), 0, m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), stream));
	}

	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
	uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
	if (train_extra_dims) {
		uint32_t n = n_extra_dims * m_nerf.training.n_images_for_training;
		m_nerf.training.extra_dims_gradient_gpu.enlarge(n);
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.extra_dims_gradient_gpu.data(), 0, m_nerf.training.extra_dims_gradient_gpu.get_bytes(), stream));
	}

	if (m_nerf.training.n_steps_since_error_map_update == 0 && !m_nerf.training.dataset.metadata.empty()) {
		uint32_t n_samples_per_image = (m_nerf.training.n_steps_between_error_map_updates * m_nerf.training.counters_rgb.rays_per_batch) / m_nerf.training.dataset.n_images;
		ivec2 res = m_nerf.training.dataset.metadata[0].resolution;
		m_nerf.training.error_map.resolution = min(ivec2((int)(std::sqrt(std::sqrt((float)n_samples_per_image)) * 3.5f)), res);
		m_nerf.training.error_map.data.resize(compMul(m_nerf.training.error_map.resolution) * m_nerf.training.dataset.n_images);
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.data.data(), 0, m_nerf.training.error_map.data.get_bytes(), stream));
	}

	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
	if (envmap_gradient) {
		CUDA_CHECK_THROW(cudaMemsetAsync(envmap_gradient, 0, sizeof(float)*m_envmap.envmap->n_params(), stream));
	}


	train_nerf_step(target_batch_size, m_nerf.training.counters_rgb, stream);


	m_trainer->optimizer_step(stream, LOSS_SCALE);

	++m_training_step;

	if (envmap_gradient) {
		m_envmap.trainer->optimizer_step(stream, LOSS_SCALE);
	}

	float loss_scalar = m_nerf.training.counters_rgb.update_after_training(target_batch_size, get_loss_scalar, stream);
	bool zero_records = m_nerf.training.counters_rgb.measured_batch_size == 0;
	if (get_loss_scalar) {
		m_loss_scalar.update(loss_scalar);
	}

	if (zero_records) {
		m_loss_scalar.set(0.f);
		tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
		m_train = false;
	}

	// Compute CDFs from the error map
	m_nerf.training.n_steps_since_error_map_update += 1;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;
	if (accumulate_error && m_nerf.training.n_steps_since_error_map_update >= m_nerf.training.n_steps_between_error_map_updates) {
		m_nerf.training.error_map.cdf_resolution = m_nerf.training.error_map.resolution;
		m_nerf.training.error_map.cdf_x_cond_y.resize(compMul(m_nerf.training.error_map.cdf_resolution) * m_nerf.training.dataset.n_images);
		m_nerf.training.error_map.cdf_y.resize(m_nerf.training.error_map.cdf_resolution.y * m_nerf.training.dataset.n_images);
		m_nerf.training.error_map.cdf_img.resize(m_nerf.training.dataset.n_images);

		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_x_cond_y.data(), 0, m_nerf.training.error_map.cdf_x_cond_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_y.data(), 0, m_nerf.training.error_map.cdf_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_img.data(), 0, m_nerf.training.error_map.cdf_img.get_bytes(), stream));

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)m_nerf.training.error_map.cdf_resolution.y, threads.x), div_round_up((uint32_t)m_nerf.training.dataset.n_images, threads.y), 1 };
		construct_cdf_2d<<<blocks, threads, 0, stream>>>(
			m_nerf.training.dataset.n_images, m_nerf.training.error_map.cdf_resolution.y, m_nerf.training.error_map.cdf_resolution.x,
			m_nerf.training.error_map.data.data(),
			m_nerf.training.error_map.cdf_x_cond_y.data(),
			m_nerf.training.error_map.cdf_y.data()
		);
		linear_kernel(construct_cdf_1d, 0, stream,
			m_nerf.training.dataset.n_images,
			m_nerf.training.error_map.cdf_resolution.y,
			m_nerf.training.error_map.cdf_y.data(),
			m_nerf.training.error_map.cdf_img.data()
		);

		// Compute image CDF on the CPU. It's single-threaded anyway. No use parallelizing.
		m_nerf.training.error_map.pmf_img_cpu.resize(m_nerf.training.error_map.cdf_img.size());
		m_nerf.training.error_map.cdf_img.copy_to_host(m_nerf.training.error_map.pmf_img_cpu);
		std::vector<float> cdf_img_cpu = m_nerf.training.error_map.pmf_img_cpu; // Copy unnormalized PDF into CDF buffer
		float cum = 0;
		for (float& f : cdf_img_cpu) {
			cum += f;
			f = cum;
		}
		float norm = 1.0f / cum;
		for (size_t i = 0; i < cdf_img_cpu.size(); ++i) {
			constexpr float MIN_PMF = 0.1f;
			m_nerf.training.error_map.pmf_img_cpu[i] = (1.0f - MIN_PMF) * m_nerf.training.error_map.pmf_img_cpu[i] * norm + MIN_PMF / (float)m_nerf.training.dataset.n_images;
			cdf_img_cpu[i] = (1.0f - MIN_PMF) * cdf_img_cpu[i] * norm + MIN_PMF * (float)(i+1) / (float)m_nerf.training.dataset.n_images;
		}
		m_nerf.training.error_map.cdf_img.copy_from_host(cdf_img_cpu);

		// Reset counters and decrease update rate.
		m_nerf.training.n_steps_since_error_map_update = 0;
		m_nerf.training.n_rays_since_error_map_update = 0;
		m_nerf.training.error_map.is_cdf_valid = true;

		m_nerf.training.n_steps_between_error_map_updates = (uint32_t)(m_nerf.training.n_steps_between_error_map_updates * 1.5f);
	}

	// Get extrinsics gradients
	m_nerf.training.n_steps_since_cam_update += 1;

	if (train_extra_dims) {
		std::vector<float> extra_dims_gradient(m_nerf.training.extra_dims_gradient_gpu.size());
		m_nerf.training.extra_dims_gradient_gpu.copy_to_host(extra_dims_gradient);

		// Optimization step
		for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
			std::vector<float> gradient(n_extra_dims);
			for (uint32_t j = 0; j < n_extra_dims; ++j) {
				gradient[j] = extra_dims_gradient[i * n_extra_dims + j] / LOSS_SCALE;
			}

			//float l2_reg = 1e-4f;
			//gradient += m_nerf.training.extra_dims_opt[i].variable() * l2_reg;

			m_nerf.training.extra_dims_opt[i].set_learning_rate(m_optimizer->learning_rate());
			m_nerf.training.extra_dims_opt[i].step(gradient);
		}

		m_nerf.training.update_extra_dims();
	}

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length || m_nerf.training.optimize_exposure;
	if (train_camera && m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates) {
		float per_camera_loss_scale = (float)m_nerf.training.n_images_for_training / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;

		if (m_nerf.training.optimize_extrinsics) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// Optimization step
			for (uint32_t k = 0; k < m_nerf.training.idx_images_for_training_extrinsics.size(); ++k) {
				uint32_t i = m_nerf.training.idx_images_for_training_extrinsics[k];

				vec3 pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
				vec3 rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

				float l2_reg = m_nerf.training.extrinsic_l2_reg;
				pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
				rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

				m_nerf.training.cam_pos_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_pos_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));
				m_nerf.training.cam_rot_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_rot_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));

				m_nerf.training.cam_pos_offset[i].step(pos_gradient);
				m_nerf.training.cam_rot_offset[i].step(rot_gradient);
			}

			m_nerf.training.update_transforms();
		}

		if (m_nerf.training.optimize_distortion) {
			linear_kernel(safe_divide, 0, stream,
				m_distortion.map->n_params(),
				m_distortion.map->gradients(),
				m_distortion.map->gradient_weights()
			);
			m_distortion.trainer->optimizer_step(stream, LOSS_SCALE*(float)m_nerf.training.n_steps_between_cam_updates);
		}

		if (m_nerf.training.optimize_focal_length) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(&m_nerf.training.cam_focal_length_gradient, m_nerf.training.cam_focal_length_gradient_gpu.data(), m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			vec2 focal_length_gradient = m_nerf.training.cam_focal_length_gradient * per_camera_loss_scale;
			float l2_reg = m_nerf.training.intrinsic_l2_reg;
			focal_length_gradient += m_nerf.training.cam_focal_length_offset.variable() * l2_reg;
			m_nerf.training.cam_focal_length_offset.set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.cam_focal_length_offset.step() / 128)),m_optimizer->learning_rate() / 1000.0f));
			m_nerf.training.cam_focal_length_offset.step(focal_length_gradient);
			m_nerf.training.dataset.update_metadata();
		}

		if (m_nerf.training.optimize_exposure) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gradient.data(), m_nerf.training.cam_exposure_gradient_gpu.data(), m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			vec3 mean_exposure = vec3(0.0f);

			// Optimization step
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				vec3 gradient = m_nerf.training.cam_exposure_gradient[i] * per_camera_loss_scale;

				float l2_reg = m_nerf.training.exposure_l2_reg;
				gradient += m_nerf.training.cam_exposure[i].variable() * l2_reg;

				m_nerf.training.cam_exposure[i].set_learning_rate(m_optimizer->learning_rate());
				m_nerf.training.cam_exposure[i].step(gradient);

				mean_exposure += m_nerf.training.cam_exposure[i].variable();
			}

			mean_exposure /= m_nerf.training.n_images_for_training;

			// Renormalize
			std::vector<vec3> cam_exposures(m_nerf.training.n_images_for_training);
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				cam_exposures[i] = m_nerf.training.cam_exposure[i].variable() -= mean_exposure;
			}

			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gpu.data(), cam_exposures.data(), m_nerf.training.n_images_for_training * sizeof(vec3), cudaMemcpyHostToDevice, stream));
		}

		m_nerf.training.n_steps_since_cam_update = 0;
	}
}

void Testbed::train_nerf_step(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

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
		uint32_t // ray_counter
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
		1
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

	uint32_t max_inference;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference = max_samples;
	} else {
		max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	}

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_training_step == 0) {
		counters.n_rays_total = 0;
	}

	uint32_t n_rays_total = counters.n_rays_total;
	counters.n_rays_total += counters.rays_per_batch;
	m_nerf.training.n_rays_since_error_map_update += counters.rays_per_batch;

	// If we have an envmap, prepare its gradient buffer
	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;

	bool sample_focal_plane_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_focal_plane_proportional_to_error;
	bool sample_image_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
	bool include_sharpness_in_error = m_nerf.training.include_sharpness_in_error;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

		linear_kernel(generate_training_samples_nerf, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			max_inference,
			n_rays_total,
			m_rng,
			ray_counter,
			counters.numsteps_counter.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			m_nerf.training.n_images_for_training,
			m_nerf.training.dataset.metadata_gpu.data(),
			m_nerf.training.transforms_gpu.data(),
			m_nerf.density_grid_bitfield.data(),
			m_nerf.max_cascade,
			m_max_level_rand_training,
			max_level,
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.train_envmap,
			m_nerf.cone_angle_constant,
			m_distortion.view(),
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			m_nerf.training.extra_dims_gpu.data(),
			m_nerf_network->n_extra_dims()
		);


	//DEBUG
	//DEBUG
	uint32_t m_track_pose_nerf_num_rays_in_tracking_step = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&m_track_pose_nerf_num_rays_in_tracking_step,std::get<10>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	m_ray_counter = m_track_pose_nerf_num_rays_in_tracking_step;
	m_rays_per_batch = counters.rays_per_batch;
	//DEBUG
	//DEBUG


		if (hg_enc) {
			hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
		}

		GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
		GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
		m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

		if (hg_enc) {
			hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
		}

		linear_kernel(compute_loss_kernel_train_nerf, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			n_rays_total,
			m_rng,
			target_batch_size,
			ray_counter,
			LOSS_SCALE,
			padded_output_width,
			m_envmap.view(),
			envmap_gradient,
			m_envmap.resolution,
			m_envmap.loss_type,
			m_background_color.rgb,
			m_color_space,
			m_nerf.training.random_bg_color,
			m_nerf.training.linear_colors,
			m_nerf.training.n_images_for_training,
			m_nerf.training.dataset.metadata_gpu.data(),
			mlp_out,
			counters.numsteps_counter_compacted.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
			dloss_dmlp_out,
			m_nerf.training.loss_type,
			m_nerf.training.depth_loss_type,
			counters.loss.data(),
			m_max_level_rand_training,
			max_level_compacted,
			m_nerf.rgb_activation,
			m_nerf.density_activation,
			m_nerf.training.snap_to_pixel_centers,
			accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.resolution,
			m_nerf.training.error_map.cdf_resolution,
			include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
			m_nerf.training.dataset.sharpness_resolution,
			m_nerf.training.sharpness_grid.data(),
			m_nerf.density_grid.data(),
			m_nerf.density_grid_mean.data(),
			m_nerf.max_cascade,
			m_nerf.training.cam_exposure_gpu.data(),
			m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
			m_nerf.training.depth_supervision_lambda,
			m_nerf.training.near_distance
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

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length;
	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
	bool prepare_input_gradients = train_camera || train_extra_dims;
	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);

	{
		auto ctx = m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, prepare_input_gradients);
		m_network->backward(stream, *ctx, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, EGradientMode::Overwrite);
	}

	if (train_extra_dims) {
		// Compute extra-dim gradients
		linear_kernel(compute_extra_dims_gradient_train_nerf, 0, stream,
			counters.rays_per_batch,
			n_rays_total,
			ray_counter,
			m_nerf.training.extra_dims_gradient_gpu.data(),
			m_nerf.training.dataset.n_extra_dims(),
			m_nerf.training.n_images_for_training,
			ray_indices,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr
		);
	}

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf, 0, stream,
			counters.rays_per_batch,
			n_rays_total,
			m_rng,
			m_aabb,
			ray_counter,
			m_nerf.training.transforms_gpu.data(),
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_pos_gradient_gpu.data() : nullptr,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_rot_gradient_gpu.data() : nullptr,
			m_nerf.training.n_images_for_training,
			m_nerf.training.dataset.metadata_gpu.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
			m_distortion.resolution,
			m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution
		);
	}

	m_rng.advance();

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}
}


void Testbed::training_prep_nerf(uint32_t batch_size, cudaStream_t stream) {
	if (m_nerf.training.n_images_for_training == 0) {
		return;
	}

	float alpha = m_nerf.training.density_grid_decay;
	uint32_t n_cascades = m_nerf.max_cascade+1;

	if (m_training_step < 256) {
		update_density_grid_nerf(alpha, NERF_GRID_N_CELLS() * n_cascades, 0, stream);
	} else {
		update_density_grid_nerf(alpha, NERF_GRID_N_CELLS() / 4 * n_cascades, NERF_GRID_N_CELLS() / 4 * n_cascades, stream);
	}
}

void Testbed::optimise_mesh_step(uint32_t n_steps) {
	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	if (!n_verts) {
		return;
	}

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);
	GPUMemory<float> coords(n_verts * floats_per_coord);
	GPUMemory<network_precision_t> mlp_out(n_verts * padded_output_width);

	GPUMatrix<float> positions_matrix((float*)coords.data(), floats_per_coord, n_verts);
	GPUMatrix<network_precision_t, RM> density_matrix(mlp_out.data(), padded_output_width, n_verts);

	const float* extra_dims_gpu = get_inference_extra_dims(m_stream.get());

	for (uint32_t i = 0; i < n_steps; ++i) {
		linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_stream.get(),
			n_verts,
			m_aabb,
			m_mesh.verts.data(),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords.data(), 1, 0, extra_stride),
			extra_dims_gpu
		);

		// For each optimizer step, we need the density at the given pos...
		m_nerf_network->density(m_stream.get(), positions_matrix, density_matrix);
		// ...as well as the input gradient w.r.t. density, which we will store in the nerf coords.
		m_nerf_network->input_gradient(m_stream.get(), 3, positions_matrix, positions_matrix);
		// and the 1ring centroid for laplacian smoothing
		compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);

		// With these, we can compute a gradient that points towards the threshold-crossing of density...
		compute_mesh_opt_gradients(
			m_mesh.thresh,
			m_mesh.verts,
			m_mesh.vert_normals,
			m_mesh.verts_smoothed,
			mlp_out.data(),
			floats_per_coord,
			(const float*)coords.data(),
			m_mesh.verts_gradient,
			m_mesh.smooth_amount,
			m_mesh.density_amount,
			m_mesh.inflate_amount
		);

		// ...that we can pass to the optimizer.
		m_mesh.verts_optimizer->step(m_stream.get(), 1.0f, (float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
	}
}

void Testbed::compute_mesh_vertex_colors() {
	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	if (!n_verts) {
		return;
	}

	m_mesh.vert_colors.resize(n_verts);
	m_mesh.vert_colors.memset(0);

	if (m_testbed_mode == ETestbedMode::Nerf) {
		const float* extra_dims_gpu = get_inference_extra_dims(m_stream.get());

		const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
		const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);
		GPUMemory<float> coords(n_verts * floats_per_coord);
		GPUMemory<float> mlp_out(n_verts * 4);

		GPUMatrix<float> positions_matrix((float*)coords.data(), floats_per_coord, n_verts);
		GPUMatrix<float> color_matrix(mlp_out.data(), 4, n_verts);
		linear_kernel(generate_nerf_network_inputs_from_positions, 0, m_stream.get(), n_verts, m_aabb, m_mesh.verts.data(), PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords.data(), 1, 0, extra_stride), extra_dims_gpu);
		m_network->inference(m_stream.get(), positions_matrix, color_matrix);
		linear_kernel(extract_srgb_with_activation, 0, m_stream.get(), n_verts * 3, 3, mlp_out.data(), (float*)m_mesh.vert_colors.data(), m_nerf.rgb_activation, m_nerf.training.linear_colors);
	}
}

GPUMemory<float> Testbed::get_density_on_grid(ivec3 res3d, const BoundingBox& aabb, const mat3& render_aabb_to_local) {
	const uint32_t n_elements = (res3d.x*res3d.y*res3d.z);
	GPUMemory<float> density(n_elements);

	const uint32_t batch_size = std::min(n_elements, 1u<<20);
	bool nerf_mode = m_testbed_mode == ETestbedMode::Nerf;

	const uint32_t padded_output_width = nerf_mode ? m_nerf_network->padded_density_output_width() : m_network->padded_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		NerfPosition,
		network_precision_t
	>(m_stream.get(), &alloc, n_elements, batch_size * padded_output_width);

	NerfPosition* positions = std::get<0>(scratch);
	network_precision_t* mlp_out = std::get<1>(scratch);

	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res3d.x, threads.x), div_round_up((uint32_t)res3d.y, threads.y), div_round_up((uint32_t)res3d.z, threads.z) };

	BoundingBox unit_cube = BoundingBox{vec3(0.0f), vec3(1.0f)};
	generate_grid_samples_nerf_uniform<<<blocks, threads, 0, m_stream.get()>>>(res3d, m_nerf.density_grid_ema_step, aabb, render_aabb_to_local, nerf_mode ? m_aabb : unit_cube , positions);

	// Only process 1m elements at a time
	for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
		uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

		GPUMatrix<network_precision_t, RM> density_matrix(mlp_out, padded_output_width, local_batch_size);

		GPUMatrix<float> positions_matrix((float*)(positions + offset), sizeof(NerfPosition)/sizeof(float), local_batch_size);
		if (nerf_mode) {
			m_nerf_network->density(m_stream.get(), positions_matrix, density_matrix);
		} else {
			m_network->inference_mixed_precision(m_stream.get(), positions_matrix, density_matrix);
		}
		linear_kernel(grid_samples_half_to_float, 0, m_stream.get(),
			local_batch_size,
			m_aabb,
			density.data() + offset , //+ axis_step * n_elements,
			mlp_out,
			m_nerf.density_activation,
			positions + offset,
			nerf_mode ? m_nerf.density_grid.data() : nullptr,
			m_nerf.max_cascade,
			(m_use_volsdf_in_nerf||m_use_stylesdf_in_nerf||m_use_coslam_sdf_in_nerf)
		);
	}

	return density;
}

GPUMemory<vec4> Testbed::get_rgba_on_grid(ivec3 res3d, vec3 ray_dir, bool voxel_centers, float depth, bool density_as_alpha) {
	const uint32_t n_elements = (res3d.x*res3d.y*res3d.z);
	GPUMemory<vec4> rgba(n_elements);

	const float* extra_dims_gpu = get_inference_extra_dims(m_stream.get());

	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float);

	GPUMemory<float> positions(n_elements * floats_per_coord);

	const uint32_t batch_size = std::min(n_elements, 1u<<20);

	// generate inputs
	const dim3 threads = { 16, 8, 1 };
	const dim3 blocks = { div_round_up((uint32_t)res3d.x, threads.x), div_round_up((uint32_t)res3d.y, threads.y), div_round_up((uint32_t)res3d.z, threads.z) };
	generate_grid_samples_nerf_uniform_dir<<<blocks, threads, 0, m_stream.get()>>>(
		res3d,
		m_nerf.density_grid_ema_step,
		m_render_aabb,
		m_render_aabb_to_local,
		m_aabb,
		ray_dir,
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)positions.data(), 1, 0, extra_stride),
		extra_dims_gpu,
		voxel_centers
	);

	// Only process 1m elements at a time
	for (uint32_t offset = 0; offset < n_elements; offset += batch_size) {
		uint32_t local_batch_size = std::min(n_elements - offset, batch_size);

		// run network
		GPUMatrix<float> positions_matrix((float*)(positions.data() + offset * floats_per_coord), floats_per_coord, local_batch_size);
		GPUMatrix<float> rgbsigma_matrix((float*)(rgba.data() + offset), 4, local_batch_size);
		m_network->inference(m_stream.get(), positions_matrix, rgbsigma_matrix);

		// convert network output to RGBA (in place)
		linear_kernel(compute_nerf_rgba_kernel, 0, m_stream.get(), local_batch_size, rgba.data() + offset, m_nerf.rgb_activation, m_nerf.density_activation, depth, density_as_alpha);
	}
	return rgba;
}

int Testbed::marching_cubes(ivec3 res3d, const BoundingBox& aabb, const mat3& render_aabb_to_local, float thresh) {
	res3d.x = next_multiple((unsigned int)res3d.x, 16u);
	res3d.y = next_multiple((unsigned int)res3d.y, 16u);
	res3d.z = next_multiple((unsigned int)res3d.z, 16u);

	if (thresh == std::numeric_limits<float>::max()) {
		thresh = m_mesh.thresh;
	}

	GPUMemory<float> density = get_density_on_grid(res3d, aabb, render_aabb_to_local);
	marching_cubes_gpu(m_stream.get(), aabb, render_aabb_to_local, res3d, thresh, density, m_mesh.verts, m_mesh.indices);

	uint32_t n_verts = (uint32_t)m_mesh.verts.size();
	m_mesh.verts_gradient.resize(n_verts);

	m_mesh.trainable_verts = std::make_shared<TrainableBuffer<3, 1, float>>(std::array<int, 1>{{(int)n_verts}});
	m_mesh.verts_gradient.copy_from_device(m_mesh.verts); // Make sure the vertices don't get destroyed in the initialization

	pcg32 rnd{m_seed};
	m_mesh.trainable_verts->initialize_params(rnd, (float*)m_mesh.verts.data());
	m_mesh.trainable_verts->set_params((float*)m_mesh.verts.data(), (float*)m_mesh.verts.data(), (float*)m_mesh.verts_gradient.data());
	m_mesh.verts.copy_from_device(m_mesh.verts_gradient);

	m_mesh.verts_optimizer.reset(create_optimizer<float>({
		{"otype", "Adam"},
		{"learning_rate", 1e-4},
		{"beta1", 0.9f},
		{"beta2", 0.99f},
	}));

	m_mesh.verts_optimizer->allocate(m_mesh.trainable_verts);

	compute_mesh_1ring(m_mesh.verts, m_mesh.indices, m_mesh.verts_smoothed, m_mesh.vert_normals);
	compute_mesh_vertex_colors();

	return (int)(m_mesh.indices.size()/3);
}

uint8_t* Testbed::Nerf::get_density_grid_bitfield_mip(uint32_t mip) {
	return density_grid_bitfield.data() + grid_mip_offset(mip)/8;
}

int Testbed::find_best_training_view(int default_view) {
	int bestimage = default_view;
	float bestscore = 1000.f;
	for (int i = 0; i < m_nerf.training.n_images_for_training; ++i) {
		float score = distance(m_nerf.training.transforms[i].start[3], m_camera[3]);
		score += 0.25f * distance(m_nerf.training.transforms[i].start[2], m_camera[2]);
		if (score < bestscore) {
			bestscore = score;
			bestimage = i;
		}
	}
	return bestimage;
}
































/*

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################




			SLAM implementation




##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################

*/


inline __device__ vec2 nerf_random_image_pos_training_with_margin(
	default_rng_t& rng, const ivec2& resolution, bool snap_to_pixel_centers,
	const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y,
	const ivec2& cdf_res, uint32_t img, const ivec2& margin,
	float* __restrict__ pdf = nullptr
	) {
	vec2 uv = random_val_2d(rng);

	if (cdf_x_cond_y) {
		uv = sample_cdf_2d(uv, img, cdf_res, cdf_x_cond_y, cdf_y, pdf);
	} else if (pdf) {
		*pdf = 1.0f;
	}

	vec2 m = vec2(margin) / vec2(resolution);
	uv = uv * (1.f - 2.f*m) + m;

	if (snap_to_pixel_centers) {
		uv = (vec2(clamp(ivec2(uv * vec2(resolution)), ivec2(0), resolution - ivec2(1))) + vec2(0.5f)) / vec2(resolution);
	}

	return uv;
}


__global__ void generate_training_samples_nerf_slam(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const uint32_t n_rays_total,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ ray_counter_depth,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	uint32_t max_mip,
	bool max_level_rand_training,
	float* __restrict__ max_level_ptr,
	bool snap_to_pixel_centers,
	bool train_envmap,
	float cone_angle_constant,
	Buffer2DView<const vec2> distortion,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 cdf_res,
	const float* __restrict__ extra_dims_gpu,
	uint32_t n_extra_dims,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const ivec2 sample_away_from_border_margin,
	const float max_grid_level_factor,
	const bool use_depth_guided_sampling,
	const float truncation_distance_for_depth_guided_sampling,
	const float dt_for_depth_guided_sampling,
	const bool use_custom_ray_marching,
	const float dt_for_regular_sampling,
	const uint32_t n_samples_for_regular_sampling,
	float* __restrict__ xy_coords_sample, //DEBUG
	uint32_t* __restrict__ img_id_sample,  //DEBUG
	float* __restrict__ sample_z_vals //DEBUG
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	uint32_t local_img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
	uint32_t img = idx_images_for_mapping[local_img];
	ivec2 resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	vec2 uv = nerf_random_image_pos_training_with_margin(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img, sample_away_from_border_margin);
	//vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(uv, resolution, 0);
	if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f) {
		return;
	}

	float max_level = max_level_rand_training ? (random_val(rng) * max_grid_level_factor) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	float motionblur_time = random_val(rng);

	const vec2 focal_length = metadata[img].focal_length;
	const vec2 principal_point = metadata[img].principal_point;
	const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
	const Lens lens = metadata[img].lens;

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);
	vec3 cam_fwd = xform[2];

	float target_z = metadata[img].depth  ? read_depth(uv, resolution, metadata[img].depth) : -1.0;

	Ray ray_unnormalized;
	const Ray* rays_in_unnormalized = metadata[img].rays;
	if (rays_in_unnormalized) {
		// Rays have been explicitly supplied. Read them.
		ray_unnormalized = rays_in_unnormalized[pix_idx];
	} else {
		ray_unnormalized = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, lens, distortion);
		if (!ray_unnormalized.is_valid()) {
			ray_unnormalized = {xform[3], xform[2]};
		}
	}

	float target_depth = length(ray_unnormalized.d) * target_z;

	vec3 ray_d_normalized = normalize(ray_unnormalized.d);

	vec2 tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
	float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x = fmaxf(tminmax.x, 0.0f);

	float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
	vec3 idir = vec3(1.0f) / ray_d_normalized;

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;
	vec3 pos;
	vec3 ray_o = ray_unnormalized.o;
	float cam_fwd_factor = dot(cam_fwd, ray_d_normalized);

	uint32_t max_samples_per_ray;
	if (use_custom_ray_marching) {
		max_samples_per_ray = n_samples_for_regular_sampling;
	} else {
		max_samples_per_ray = NERF_STEPS(); 
	}

	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < max_samples_per_ray) {
		float dt;
		if (use_custom_ray_marching){ //ie sampling on planes VS spheres
			float cur_z = dot(cam_fwd, pos - ray_o);
			//TODO add perturb
			dt = calc_dt_slaim(
				cur_z, target_z, dt_for_regular_sampling, 
				dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling,
				cam_fwd_factor, use_depth_guided_sampling
			);
		} else {
			if (use_depth_guided_sampling) {
				float cur_depth = distance(pos, ray_o);
				dt = calc_dt_with_depth_guided_sampling(
					t, cone_angle,
					cur_depth, target_depth,
					dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling
				);
			} else {
				dt = calc_dt(t, cone_angle);
			}
		}
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid) {
			if (density_grid_occupied_at(pos, density_grid, mip)) {
				++j;
				t += dt;
			} else {
				t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
			}
		} else {
			++j;
			t += dt;
		}
	}

	if (j == 0 && !train_envmap) {
		return;
	}

	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	if (base + numsteps > max_samples) {
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	// get depth_ray_counters
	if (target_depth > 0.) {
		uint32_t ray_idx_depth = atomicAdd(ray_counter_depth, 1);
	}

	ray_indices_out[ray_idx] = i;
	rays_out_unnormalized[ray_idx] = ray_unnormalized;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	if (img_id_sample) {img_id_sample[ray_idx] = img;}
	if (xy_coords_sample) {xy_coords_sample[ray_idx*2+0] = uv.x; xy_coords_sample[ray_idx*2+1] = uv.y;}

	vec3 warped_dir = warp_direction(ray_d_normalized);
	t=startt;
	j=0;
	
	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
		float dt;
		if (use_custom_ray_marching){
			float cur_z = dot(cam_fwd, pos - ray_o);
			//TODO add perturb
			dt = calc_dt_slaim(
				cur_z, target_z, dt_for_regular_sampling, 
				dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling,
				cam_fwd_factor, use_depth_guided_sampling
			);
		} else {
			if (use_depth_guided_sampling) {
				float cur_depth = distance(pos, ray_o);
				dt = calc_dt_with_depth_guided_sampling(
					t, cone_angle,
					cur_depth, target_depth,
					dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling
				);
			} else {
				dt = calc_dt(t, cone_angle);
			}
		}
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid){
			if (density_grid_occupied_at(pos, density_grid, mip)) {
				coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
				if (sample_z_vals) {
					float cur_z = dot(cam_fwd, pos - ray_o);
					sample_z_vals[base+j] = cur_z;
				}
				++j;
				t += dt;
			} else {
				t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
			}
		} else {
			coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
			if (sample_z_vals) {
				float cur_z = dot(cam_fwd, pos - ray_o);
				sample_z_vals[base+j] = cur_z;
			}
			++j;
			t += dt;
		}
	}

	if (max_level_rand_training) {
		max_level_ptr += base;
		for (j = 0; j < numsteps; ++j) {
			max_level_ptr[j] = max_level;
		}
	}
}

__global__ void compute_loss_kernel_train_nerf_slam(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const uint32_t max_samples_compacted,
	const uint32_t* __restrict__ rays_counter,
	const uint32_t* __restrict__ rays_counter_depth,
	float loss_scale,
	int padded_output_width,
	Buffer2DView<const vec4> envmap,
	float* __restrict__ envmap_gradient,
	const ivec2 envmap_resolution,
	ELossType envmap_loss_type,
	vec3 background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	bool max_level_rand_training,
	float* __restrict__ max_level_compacted_ptr,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	bool snap_to_pixel_centers,
	float* __restrict__ error_map,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res,
	const ivec2 error_map_cdf_res,
	const float* __restrict__ sharpness_data,
	ivec2 sharpness_resolution,
	float* __restrict__ sharpness_grid,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	uint32_t max_mip,
	const vec3* __restrict__ exposure,
	vec3* __restrict__ exposure_gradient,
	float depth_supervision_lambda,
	float rgb_supervision_lambda,
	float near_distance,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const ivec2 sample_away_from_border_margin,
	const bool use_depth_variance_in_loss,
	const float max_grid_level_factor,
	const bool add_free_space_loss,
	const float free_space_supervision_lambda,
	const float free_space_supervision_distance,
	float* __restrict__ sample_outputs, //DEBUG
	const bool use_custom_ray_marching,
	float* __restrict__ rays_depth_variance,
	const bool add_DSnerf_loss,
	const float DS_nerf_supervision_lambda,
	const float DS_nerf_supervision_depth_sigma,
	//SDF
	const bool use_stylesdf,
	const bool use_volsdf,
	const bool use_coslam_sdf,
	const float truncation_distance,
	const float beta,
	const bool add_sdf_loss,
	const float sdf_supervision_lambda,
	const bool add_sdf_free_space_loss,
	const float sdf_free_space_supervision_lambda
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	float img_pdf = 1.0f;
	uint32_t local_img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
	uint32_t img = idx_images_for_mapping[local_img];
	ivec2 resolution = metadata[img].resolution;

	float uv_pdf = 1.0f;
	vec2 uv = nerf_random_image_pos_training_with_margin(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, sample_away_from_border_margin, &uv_pdf);
	//vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &uv_pdf);
	float max_level = max_level_rand_training ? (random_val(rng) * max_grid_level_factor) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	//rng.advance(1); // motionblur_time
	float motionblur_time = random_val(rng);

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);
	vec3 cam_fwd = xform[2];

	float target_z = metadata[img].depth ? read_depth(uv, resolution, metadata[img].depth) : -1.0f;
	float target_d = length(rays_in_unnormalized[i].d) * target_z;

	float T = 1.f;

	//coslam SDF stuff
	bool found_surface = false;	
	float surface_d = 0.f;	
	float weight_sum = 0.f;

	float EPSILON = 1e-4f;

	vec3 rgb_ray = vec3(0.0f);
	vec3 hitpoint = vec3(0.0f);

	float depth_ray = 0.f;
	uint32_t compacted_numsteps = 0;
	vec3 ray_o = rays_in_unnormalized[i].o;

	//precompute min depth for potential early stop
	float min_depth_for_sampling=-1.f;
	if (target_d>0.){
		if (add_free_space_loss) {
			min_depth_for_sampling = max(min_depth_for_sampling, target_d-free_space_supervision_distance); //ensures we sample enough points for the FS loss
		}
		if (add_DSnerf_loss) {
			min_depth_for_sampling = max(min_depth_for_sampling, target_d+3*DS_nerf_supervision_depth_sigma); //ensures we sample enough points for the DS loss
		}
		if (add_sdf_loss) {
			min_depth_for_sampling = max(min_depth_for_sampling, target_d+truncation_distance); //ensures we sample enough points for the sdf loss
		}
		if (add_sdf_free_space_loss) {
			min_depth_for_sampling = max(min_depth_for_sampling, target_d-truncation_distance); //ensures we sample enough points for the sdf FS loss
		}
	}
	
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {

		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = distance(pos, ray_o);
		float cur_z = dot(cam_fwd, pos - ray_o);
		float weight;

		if (use_coslam_sdf) {
			float sdf = network_to_density(float(local_network_output[3]), density_activation);
			if (sdf < 0. && !found_surface) {
				found_surface=true;
				surface_d=cur_depth;
			}
			if (found_surface && (cur_depth >(surface_d+truncation_distance))) {
				weight = 0.f;
				//Need to keep sampling for loss supervison (SDF)
				if (min_depth_for_sampling > 0.0) {
					if (cur_depth > min_depth_for_sampling) { //ensures we sample enough points for the corresponding losses
						break;
					}
				} else {break;}
			} else {
				weight = coslam_sdf_to_weight(sdf, beta);
				weight_sum += weight;
			}
		} else {
			float density = network_to_density(float(local_network_output[3]), density_activation);
			if (use_volsdf){
				density = vol_sdf_to_density(density, beta);
			} else if(use_stylesdf){
				density = style_sdf_to_density(density, beta);
			}

			if (T < EPSILON) {
				if (min_depth_for_sampling > 0.0) {
					if (cur_depth > min_depth_for_sampling) { //ensures we sample enough points for the corresponding losses
						break;
					}
				} else {
					break;
				}
			}

			const float alpha = 1.f - __expf(-density * dt);
			weight = alpha * T;
			T *= (1.f - alpha);
		}
		
		rgb_ray += weight * rgb;
		hitpoint += weight * pos;
		depth_ray += weight * cur_depth;

		network_output += padded_output_width;
		coords_in += 1;
	}
	if (use_coslam_sdf) {
		weight_sum += 1e-6f;
		rgb_ray /= weight_sum;
		depth_ray /= weight_sum;
		hitpoint /= weight_sum;
		T = 0.;
	} else {
		hitpoint /= (1.0f - T);
	}

	// compute depth variance
	float depth_var = 0.f;
	if (use_depth_variance_in_loss) {
		//rewind pointers
		network_output -= compacted_numsteps * padded_output_width;
		coords_in -= compacted_numsteps;
		T = 1.f;
		for (uint32_t j=0; j < compacted_numsteps; ++j) {
			const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 	       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
			const float dt = unwarp_dt(coords_in.ptr->dt);
			float cur_depth = distance(pos, ray_o);
 	 	    float tmp = (cur_depth - depth_ray) * (cur_depth - depth_ray);
			float weight;

			if (use_coslam_sdf) {
				float sdf = network_to_density(float(local_network_output[3]), density_activation);
				if (found_surface && (cur_depth >(surface_d+truncation_distance))) {
					weight = 0.f; //still need to go thru the entire pointer (then easier to rewind)
				} else {
					weight = coslam_sdf_to_weight(sdf, beta) / weight_sum;
				}
			} else {
				float density = network_to_density(float(local_network_output[3]), density_activation);
				if (use_volsdf){
					density = vol_sdf_to_density(density, beta);
				} else if (use_stylesdf){
					density = style_sdf_to_density(density, beta);
				}
				const float alpha = 1.f - __expf(-density * dt);
				weight = alpha * T;
				T *= (1.f - alpha);
			}

 	       	depth_var += weight * tmp;

			network_output += padded_output_width;
			coords_in += 1;
		}
	}
	if (use_coslam_sdf) {
		T=0.f;
	}

	if (rays_depth_variance) {
		rays_depth_variance[i] = depth_var;
	}

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	vec4 envmap_value;
	vec3 dir;
	if (envmap) {
		dir = normalize(rays_in_unnormalized[i].d);
		envmap_value = read_envmap(envmap, dir);
		background_color = envmap_value.rgb + background_color * (1.0f - envmap_value.a);
	}

	vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
	// vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
	// vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
	vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

	vec3 rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.rgb + (1.0f - texsamp.a) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.a > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

	// Step again, this time computing loss
	network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_in[i*2+0] = compacted_numsteps;
	numsteps_in[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	max_level_compacted_ptr += compacted_base;
	coords_out += compacted_base;

	dloss_doutput += compacted_base * padded_output_width;

	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
	lg = lg * rgb_supervision_lambda;
	lg.loss /= img_pdf * uv_pdf;

	float ray_dir_norm = length(rays_in_unnormalized[i].d);
	float target_depth = length(rays_in_unnormalized[i].d) * (((depth_supervision_lambda > 0.0f || add_DSnerf_loss || add_free_space_loss || add_sdf_loss || add_sdf_free_space_loss) && metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);
	LossAndGradient lg_depth;
	if (use_custom_ray_marching) {
		lg_depth = loss_and_gradient(vec3(target_depth / ray_dir_norm), vec3(depth_ray / ray_dir_norm), depth_loss_type);
	} else {
		lg_depth = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
	}
	if (use_depth_variance_in_loss) {
        if (depth_var < 1e-6){
            depth_var = 1e-6;
        }
		float depth_std = sqrt(depth_var);
		if (use_custom_ray_marching) {
			depth_std /= ray_dir_norm;
		}
		lg_depth = lg_depth / depth_std;
	}
	float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;
	float depth_loss_value    = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.loss.x : 0;

	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * uv_pdf;

	//FREE-SPACE-LOSS-CODE
	// /!\ compacted_numsteps might be smaller due to memory limits -> shouldn't have much of an imapact.
	float free_space_loss = 0.;
	uint32_t n_steps_for_free_space_loss = 0;
	if (add_free_space_loss) {
		if (target_depth > 0.0f){
			uint32_t tmp_j=0;
			for (; tmp_j < compacted_numsteps; ++tmp_j) {
				const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
				const float dt = unwarp_dt(coords_in.ptr->dt);
 		       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
				float cur_depth = distance(pos, ray_o);
				
				if (cur_depth > (target_depth - free_space_supervision_distance)) {
					break;
				}

				float density = network_to_density(float(local_network_output[3]), density_activation);
				if (use_volsdf){
					density = vol_sdf_to_density(density, beta);
				} else if (use_stylesdf){
					density = style_sdf_to_density(density, beta);
				}

				float alpha = 1.f - __expf(-density * dt);

				free_space_loss +=  alpha * alpha;

				network_output += padded_output_width;
				coords_in += 1;
			}
			if (tmp_j > 0) {
				n_steps_for_free_space_loss = tmp_j;
				network_output -= padded_output_width * tmp_j; // rewind the pointer
				coords_in -= tmp_j;
				free_space_loss /= (float)tmp_j;
			}
		}
	}

	//	
	//	
	//	
	//	
	//DS-NERF-LOSS-CODE
	float ds_nerf_loss = 0.;
	uint32_t n_steps_for_ds_nerf_loss = 0;
	if (add_DSnerf_loss) {
		if (target_depth > 0.0f){
			uint32_t tmp_j=0;
			T = 1.f;
			for (; tmp_j < compacted_numsteps; ++tmp_j) {
				const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
				const float dt = unwarp_dt(coords_in.ptr->dt);
 		       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
				const float cur_depth = distance(pos, ray_o);
				float density = network_to_density(float(local_network_output[3]), density_activation);
				if (use_volsdf){
					density = vol_sdf_to_density(density, beta);
				} else if (use_stylesdf){
					density = style_sdf_to_density(density, beta);
				}
				const float alpha = 1.f - __expf(-density * dt);

				const float h = alpha * T;
			
				ds_nerf_loss -= __logf(h+EPSILON) * __expf( -( (cur_depth-target_depth)*(cur_depth-target_depth) ) / (2*DS_nerf_supervision_depth_sigma*DS_nerf_supervision_depth_sigma) ) * dt;
				
				T *= (1.f - alpha);

				network_output += padded_output_width;
				coords_in += 1;
			}
			if (tmp_j > 0) {
				n_steps_for_ds_nerf_loss = tmp_j;
				network_output -= padded_output_width * tmp_j; // rewind the pointer
				coords_in -= tmp_j;
				//ds_nerf_loss /= (float)tmp_j;
			}
		}
	}

	//
	//
	//
	// SDF LOSS
	float sdf_loss = 0.;
	uint32_t n_steps_for_sdf_loss = 0;
	float sdf_weight = 0.;
	if (add_sdf_loss) {
		if (target_depth > 0.0f){
			uint32_t tmp_j=0;
			for (; tmp_j < compacted_numsteps; ++tmp_j) {
				const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 		       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
				const float cur_depth = distance(pos, ray_o);
				
				if (cur_depth > (target_depth + truncation_distance)) {
					break;
				}
				
				if (cur_depth >= (target_depth - truncation_distance)) {
					float sdf = network_to_density(float(local_network_output[3]), density_activation);
					sdf_loss += (cur_depth + sdf * truncation_distance - target_depth) * (cur_depth + sdf * truncation_distance - target_depth);
					n_steps_for_sdf_loss+=1;
				}

				network_output += padded_output_width;
				coords_in += 1;
			}
			if (tmp_j > 0) {
				network_output -= padded_output_width * tmp_j; // rewind the pointer
				coords_in -= tmp_j;
				if (n_steps_for_sdf_loss > 0) {
					sdf_loss /= (float)n_steps_for_sdf_loss;
					sdf_weight = 1. - (float)n_steps_for_sdf_loss / (float)compacted_numsteps;
					//sdf_loss *= sdf_weight; //may or may not be needed
				}
			}
		}
	}


	//
	//
	//
	// SDF Free Space loss
	float sdf_free_space_loss = 0.;
	uint32_t n_steps_for_sdf_free_space_loss = 0;
	float sdf_fs_weight = 0.;
	if (add_sdf_free_space_loss) {
		if (target_depth > 0.0f){
			uint32_t tmp_j=0;
			for (; tmp_j < compacted_numsteps; ++tmp_j) {
				const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 		       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
				const float cur_depth = distance(pos, ray_o);
				
				if (cur_depth > (target_depth - truncation_distance)) {
					break;
				}
				
				float sdf = network_to_density(float(local_network_output[3]), density_activation);
				sdf_free_space_loss += (sdf - 1.) * (sdf - 1.);
				n_steps_for_sdf_free_space_loss+=1;

				network_output += padded_output_width;
				coords_in += 1;
			}
			if (tmp_j > 0) {
				network_output -= padded_output_width * tmp_j; // rewind the pointer
				coords_in -= tmp_j;
				if (n_steps_for_sdf_free_space_loss > 0) {
					sdf_free_space_loss /= (float)n_steps_for_sdf_free_space_loss;
					sdf_fs_weight = 1.f - ((float)n_steps_for_sdf_free_space_loss / (float)compacted_numsteps);
					//sdf_free_space_loss *= sdf_fs_weight; //may or may not be needed
				}
			}
		}
	}



	uint32_t total_rays = *rays_counter;
	uint32_t total_rays_depth = *rays_counter_depth;

	float rgb_loss = compAdd(lg.loss) / 3.0f;
	float depth_loss = depth_loss_value + free_space_supervision_lambda*free_space_loss + DS_nerf_supervision_lambda*ds_nerf_loss;
	depth_loss += sdf_supervision_lambda*sdf_loss + sdf_free_space_supervision_lambda*sdf_free_space_loss;
	float mean_loss = rgb_loss;
	
	//printf(" %f, %f, %f, %f | ", rgb_loss, depth_loss_value, ds_nerf_loss, free_space_loss);

	if (loss_output) {
		loss_output[i] = rgb_loss / (float)total_rays + depth_loss / (float)total_rays_depth;
	}

	if (error_map) {
		const vec2 pos = clamp(uv * vec2(error_map_res) - vec2(0.5f), vec2(0.0f), vec2(error_map_res) - vec2(1.0f + 1e-4f));
		const ivec2 pos_int = pos;
		const vec2 weight = pos - vec2(pos_int);

		ivec2 idx = clamp(pos_int, ivec2(0), resolution - ivec2(2));

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[local_img * compMul(error_map_res) + y * error_map_res.x + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), ivec2(0), sharpness_resolution - ivec2(1));
			float sharp = sharpness_data[img * compMul(sharpness_resolution) + sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x,   idx.y,   (1 - weight.x) * (1 - weight.y) * mean_loss);
		deposit_val(idx.x+1, idx.y,        weight.x  * (1 - weight.y) * mean_loss);
		deposit_val(idx.x,   idx.y+1, (1 - weight.x) *      weight.y  * mean_loss);
		deposit_val(idx.x+1, idx.y+1,      weight.x  *      weight.y  * mean_loss);
	}

	//loss_scale /= n_rays;
	float loss_scale_rgb = loss_scale / (float)total_rays;
	float loss_scale_depth = 0.0f;
	if (total_rays_depth > 0) {
		loss_scale_depth = loss_scale / (float)total_rays_depth;
	}

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
	vec3 rgb_ray2 = { 0.f,0.f,0.f };
	float depth_ray2 = 0.f;
	T = 1.f;
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {
		if (max_level_rand_training) {
			max_level_compacted_ptr[j] = max_level;
		}
		// Compact network inputs
		NerfCoordinate* coord_out = coords_out(j);
		const NerfCoordinate* coord_in = coords_in(j);
		coord_out->copy(*coord_in, coords_out.stride_in_bytes);

		const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
		float depth = distance(pos, ray_o);
		float cur_z = dot(cam_fwd, pos - ray_o);

		float dt = unwarp_dt(coord_in->dt);
		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);

		float weight;
		float h;
		float sdf;
		float alpha;
		float density;

		if (use_coslam_sdf) {
			sdf = network_to_density(float(local_network_output[3]), density_activation);
			if (found_surface && (depth >(surface_d+truncation_distance))) {
				weight = 0.f; //still need to go thru the entire pointer for loss and graidents
			} else {
				weight = coslam_sdf_to_weight(sdf, beta) / weight_sum;
			}
		} else {
			density = network_to_density(float(local_network_output[3]), density_activation);
			sdf = density;
			if (use_volsdf){
				density = vol_sdf_to_density(sdf, beta);
			} else if (use_stylesdf) {
				density = style_sdf_to_density(sdf, beta);
			}
			alpha = 1.f - __expf(-density * dt);
			weight = alpha * T;
			h = alpha * T + EPSILON;
			T *= (1.f - alpha);
		}
		
		rgb_ray2 += weight * rgb;
		depth_ray2 += weight * depth;

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const vec3 suffix = rgb_ray - rgb_ray2;
		const vec3 dloss_by_drgb = weight * lg.gradient;

		tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = (loss_scale_rgb / 3.0) * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = (loss_scale_rgb / 3.0) * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = (loss_scale_rgb / 3.0) * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		float dloss_by_dmlp;
		float density_derivative;
		float sdf_derivative;
		if (use_coslam_sdf) {
			if (found_surface && (depth >(surface_d+truncation_distance))) {
				dloss_by_dmlp = 0.f;
			} else {
				float depth_supervision = loss_scale_depth * depth_loss_gradient * (depth - depth_ray) / weight_sum;
				float rgb_supervision = loss_scale_rgb / 3.0 * dot(lg.gradient, (rgb - rgb_ray) / weight_sum);
				sdf_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
				float weight_derivative = coslam_sdf_to_weight_derivative(sdf, beta);
				dloss_by_dmlp = weight_derivative * sdf_derivative * (rgb_supervision + depth_supervision);
			}
		} else {
			density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
			sdf_derivative=density_derivative;
			if (use_volsdf){
				density_derivative *= vol_sdf_to_density_derivative(sdf, beta);
			} else if (use_stylesdf) {
				density_derivative *= style_sdf_to_density_derivative(sdf, beta);
			}

			const float depth_suffix = depth_ray - depth_ray2;
			float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);
			if (use_custom_ray_marching) {
				depth_supervision /= ray_dir_norm;
			}

			dloss_by_dmlp = density_derivative * (
				dt * ( loss_scale_rgb / 3.0 * dot(lg.gradient, T * rgb - suffix) + loss_scale_depth * depth_supervision)
			);
		}

		//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
		//dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

		//adding free space supervision using depth
		if (add_free_space_loss) {
			if (target_depth > 0.f){
				if (n_steps_for_free_space_loss>0){
					if (depth <= (target_depth - free_space_supervision_distance)) {
						dloss_by_dmlp += loss_scale_depth * density_derivative * free_space_supervision_lambda * 2 * alpha * dt * __expf(-density*dt) / (float)n_steps_for_free_space_loss;
					}
				}
			}
		}

		if (add_DSnerf_loss) {
			if (target_depth > 0.f){
				if (n_steps_for_ds_nerf_loss>0){
					float tmp_ds_grad = -dt*dt*T/h*__expf(-((depth-target_depth)*(depth-target_depth))/(2*DS_nerf_supervision_depth_sigma*DS_nerf_supervision_depth_sigma));
					for (uint32_t k = j+1; k < compacted_numsteps; ++k) {
						const NerfCoordinate* tmp_coord_in = coords_in(k);
						const vec3 tmp_pos = unwarp_position(tmp_coord_in->pos.p, aabb);
						float tmp_depth = distance(tmp_pos, ray_o);
						float tmp_dt = unwarp_dt(tmp_coord_in->dt);
						tmp_ds_grad += dt * tmp_dt * __expf(-((tmp_depth-target_depth)*(tmp_depth-target_depth))/(2*DS_nerf_supervision_depth_sigma*DS_nerf_supervision_depth_sigma));
					}
					//dloss_by_dmlp += loss_scale_depth * density_derivative * DS_nerf_supervision_lambda * tmp_ds_grad / (float)n_steps_for_ds_nerf_loss;
					dloss_by_dmlp += loss_scale_depth * density_derivative * DS_nerf_supervision_lambda * tmp_ds_grad;
				}
			}
		}

		//adding SDF losses
		if (add_sdf_loss) {
			if (n_steps_for_sdf_loss > 0) {
				if (target_depth > 0.f){
					if ((depth >= (target_depth - truncation_distance))&&(depth <= (target_depth + truncation_distance))) {
						float dloss_sdf_by_dsdf = 2*(depth + sdf * truncation_distance - target_depth)*truncation_distance;
						//dloss_by_dmlp += loss_scale_depth * sdf_derivative * sdf_supervision_lambda * dloss_sdf_by_dsdf / (float)n_steps_for_sdf_loss * sdf_weight;
						dloss_by_dmlp += loss_scale_depth * sdf_derivative * sdf_supervision_lambda * dloss_sdf_by_dsdf / (float)n_steps_for_sdf_loss;
					}
				}
			}
		}

		//adding SDF Free Space losses
		if (add_sdf_free_space_loss) {
			if (n_steps_for_sdf_free_space_loss > 0) {
				if (target_depth > 0.f){
					if (depth < (target_depth - truncation_distance)) {
						float dloss_sdf_fs_by_dsdf = 2*(sdf - 1);
						//dloss_by_dmlp += loss_scale_depth * sdf_derivative * sdf_free_space_supervision_lambda * dloss_sdf_fs_by_dsdf / (float)n_steps_for_sdf_free_space_loss * sdf_fs_weight;
						dloss_by_dmlp += loss_scale_depth * sdf_derivative * sdf_free_space_supervision_lambda * dloss_sdf_fs_by_dsdf / (float)n_steps_for_sdf_free_space_loss;
					}
				}
			}
		}

		local_dL_doutput[3] =
			//loss_scale * dloss_by_dmlp +
			dloss_by_dmlp +
			(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
			(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
			;

		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += padded_output_width;

		if (sample_outputs) { //For debug purposes
			sample_outputs[(compacted_base+j)*4+0] = density;
			sample_outputs[(compacted_base+j)*4+1] = alpha;
			sample_outputs[(compacted_base+j)*4+2] = weight;
			sample_outputs[(compacted_base+j)*4+3] = T;
		}
	}

	if (exposure_gradient) {
		// Assume symmetric loss
		vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

		if (!train_in_linear_colors) {
			dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
		}

		// 2^exposure * log(2)
		vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
		atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
		atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
		atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
	}

	if (compacted_numsteps == numsteps && envmap_gradient) {
		vec3 loss_gradient = lg.gradient;
		if (envmap_loss_type != loss_type) {
			loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
		}

		vec3 dloss_by_dbackground = T * loss_gradient;
		if (!train_in_linear_colors) {
			dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
		}

		tcnn::vector_t<tcnn::network_precision_t, 4> dL_denvmap;
		dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
		dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
		dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;


		float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

		// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
		dL_denvmap[3] = (tcnn::network_precision_t)0;

		deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
	}
}







__global__ void compute_loss_kernel_train_nerf_sdf_slam(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const uint32_t max_samples_compacted,
	const uint32_t* __restrict__ rays_counter,
	const uint32_t* __restrict__ rays_counter_depth,
	float loss_scale,
	int padded_output_width,
	Buffer2DView<const vec4> envmap,
	float* __restrict__ envmap_gradient,
	const ivec2 envmap_resolution,
	ELossType envmap_loss_type,
	vec3 background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	bool max_level_rand_training,
	float* __restrict__ max_level_compacted_ptr,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	bool snap_to_pixel_centers,
	float* __restrict__ error_map,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res,
	const ivec2 error_map_cdf_res,
	const float* __restrict__ sharpness_data,
	ivec2 sharpness_resolution,
	float* __restrict__ sharpness_grid,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	uint32_t max_mip,
	const vec3* __restrict__ exposure,
	vec3* __restrict__ exposure_gradient,
	float depth_supervision_lambda,
	float near_distance,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const ivec2 sample_away_from_border_margin,
	const bool use_depth_variance_in_loss,
	const float max_grid_level_factor,
	const bool add_free_space_loss,
	const float free_space_supervision_lambda,
	const float free_space_supervision_distance,
    const float truncation_distance,
	const bool add_sdf_loss,
	const float sdf_supervision_lambda
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	// some variables needed to early stop.
    bool found_surface = false;
    float prev_sdf = 1.0f;
    float zmin;
    float norm_weights = 0.f;

	vec3 rgb_ray = vec3(0.0f);
	vec3 hitpoint = vec3(0.0f);
	float depth_ray = 0.f;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	float img_pdf = 1.0f;
	uint32_t local_img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
	uint32_t img = idx_images_for_mapping[local_img];
	ivec2 resolution = metadata[img].resolution;

	float uv_pdf = 1.0f;
	vec2 uv = nerf_random_image_pos_training_with_margin(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, sample_away_from_border_margin, &uv_pdf);
	//vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, &uv_pdf);
	float max_level = max_level_rand_training ? (random_val(rng) * max_grid_level_factor) : 1.0f; // Multiply by 2 to ensure 50% of training is at max level

	//rng.advance(1); // motionblur_time
	float motionblur_time = random_val(rng);

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);
	vec3 cam_fwd = xform[2];

	float target_z = metadata[img].depth ? read_depth(uv, resolution, metadata[img].depth) : -1.0f;

	uint32_t compacted_numsteps = 0;
	vec3 ray_o = rays_in_unnormalized[i].o;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {

        const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		float cur_depth = distance(pos, ray_o);
		float cur_z = dot(cam_fwd, pos - ray_o);

        float sdf = network_to_density(float(local_network_output[3]), density_activation); // uses "density" as sdf here.

        prev_sdf *= sdf;

		float weight = sigmoid(sdf / truncation_distance) * sigmoid(-sdf / truncation_distance);

		// hacky way to find the compacted min number of steps
		// need to find the first surface + need samples around target_depth(same as target_z) if needed for SDF and free space supervision
        if (compacted_numsteps > 0){
            if (prev_sdf < 0. & !found_surface) { // change of sdf sign indicates that we found a surface positive near us and neg inside the surface.
                found_surface = true;
                zmin = cur_z;
            }
            if (found_surface) {
		        if (cur_z > (zmin + truncation_distance)) {
					weight = 0.;
					if (target_z > 0.) {
						if (cur_z > (target_z + truncation_distance)) {
							break;
						}
					} else {
		    	    	break;
					}
		        }
            }
        }

        norm_weights += weight;

		rgb_ray += weight * rgb;
		hitpoint += weight * pos;
		depth_ray += weight * cur_depth;

		network_output += padded_output_width;
		coords_in += 1;

		prev_sdf = sdf;
	}
	norm_weights += 1.e-8;

	hitpoint /= norm_weights;
    rgb_ray /= norm_weights;
    depth_ray /= norm_weights;

	// compute depth variance
	float depth_var = 0.f;
	if (use_depth_variance_in_loss) {
		//rewind pointers
		network_output -= compacted_numsteps * padded_output_width;
		coords_in -= compacted_numsteps;
		for (uint32_t j=0; j < compacted_numsteps; ++j) {
			const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 	       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
			float cur_depth = distance(pos, ray_o);
			float cur_z = dot(cam_fwd, pos - ray_o);

			float sdf = network_to_density(float(local_network_output[3]), density_activation);

 	       	float tmp = (cur_depth - depth_ray) * (cur_depth - depth_ray);

            float weight = sigmoid(sdf / truncation_distance) * sigmoid(-sdf / truncation_distance);

			if ( (found_surface) && (cur_z > (zmin + truncation_distance)) ) {
				weight = 0.;
			}

 	       	depth_var += weight * tmp;

			network_output += padded_output_width;
			coords_in += 1;
		}
	    depth_var /= norm_weights;
	}

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	vec4 envmap_value;
	vec3 dir;
	if (envmap) {
		dir = normalize(rays_in_unnormalized[i].d);
		envmap_value = read_envmap(envmap, dir);
		background_color = envmap_value.rgb + background_color * (1.0f - envmap_value.a);
	}

	vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
	// vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
	// vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
	vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

	vec3 rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.rgb + (1.0f - texsamp.a) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.a > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		//rgb_ray += (1 / norm_weights) * background_color; --> turning this OFF for SLAM experiments
	}

	// Step again, this time computing loss
	network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_in[i*2+0] = compacted_numsteps;
	numsteps_in[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	max_level_compacted_ptr += compacted_base;
	coords_out += compacted_base;

	dloss_doutput += compacted_base * padded_output_width;

	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
	lg.loss /= img_pdf * uv_pdf;

	float target_depth = length(rays_in_unnormalized[i].d) * ((depth_supervision_lambda > 0.0f && metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);
	LossAndGradient lg_depth = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
	if (use_depth_variance_in_loss) {
        if (depth_var < 1e-6){
            depth_var = 1e-6;
        }
		lg_depth = lg_depth / sqrt(depth_var);
	}
	float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;
	float depth_loss_value    = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.loss.x : 0;

	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * uv_pdf;

	//FREE-SPACE-LOSS-CODE
	float free_space_loss = 0.;
	float free_space_weight = 0.;
	uint32_t n_steps_for_free_space_loss = 0;
	if (add_free_space_loss) {
		if (target_z > 0.0f){
			uint32_t tmp_j=0;
			for (; tmp_j < compacted_numsteps; ++tmp_j) {
				const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 		       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
				float cur_depth = distance(pos, ray_o);
				float cur_z = dot(cam_fwd, pos - ray_o);

				if (cur_z > (target_z - truncation_distance)) {
					break;
				}

			    float sdf = network_to_density(float(local_network_output[3]), density_activation);

				//free_space_loss += (sdf-truncation_distance) * (sdf-truncation_distance);
				free_space_loss += (sdf-1) * (sdf-1); // sdf is normalized [-1,1]

				network_output += padded_output_width;
				coords_in += 1;
			}
			if (tmp_j > 0) {
				n_steps_for_free_space_loss = tmp_j;
				network_output -= padded_output_width * tmp_j; // rewind the pointer
				coords_in -= tmp_j;
				free_space_loss /= (float)numsteps;
				free_space_weight = 1. - (float)n_steps_for_free_space_loss / (float)numsteps;
			}
		}
	}

	//SDF LOSS
	float sdf_loss = 0.;
	float sdf_weight = 0.;
	uint32_t n_steps_for_sdf_loss = 0;
	if (add_sdf_loss) {
		if (target_z > 0.0f){
			uint32_t tmp_j=0;
			for (; tmp_j < compacted_numsteps; ++tmp_j) {
				const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 		       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
				float cur_depth = distance(pos, ray_o);
				float cur_z = dot(cam_fwd, pos - ray_o);

				if (cur_z > (target_z + truncation_distance)) {
					break;
				}

				if (cur_z > (target_z - truncation_distance)) {

					float sdf = network_to_density(float(local_network_output[3]), density_activation);

					sdf_loss += (sdf*truncation_distance - (target_z - cur_z)) * (sdf*truncation_distance - (target_z - cur_z));

					n_steps_for_sdf_loss += 1;
				}

				network_output += padded_output_width;
				coords_in += 1;
			}
			if (tmp_j > 0) {
				network_output -= padded_output_width * tmp_j; // rewind the pointer
				coords_in -= tmp_j;
				if (n_steps_for_sdf_loss > 0) {
					sdf_loss /= (float)numsteps;
					sdf_weight = 1. - (float)n_steps_for_sdf_loss / (float)numsteps;
				}
			}
		}
	}

	uint32_t total_rays = *rays_counter;
	uint32_t total_rays_depth = *rays_counter_depth;

	float rgb_loss = compAdd(lg.loss) / 3.0f;
	float depth_loss = depth_loss_value;
	depth_loss += free_space_supervision_lambda*free_space_loss*free_space_weight;
	depth_loss += sdf_supervision_lambda*sdf_loss*sdf_weight;

	float mean_loss = rgb_loss; //error map only on RGB loss

	if (loss_output) {
		loss_output[i] = rgb_loss / (float)total_rays + depth_loss / (float)total_rays_depth;
	}

	if (error_map) {
		const vec2 pos = clamp(uv * vec2(error_map_res) - vec2(0.5f), vec2(0.0f), vec2(error_map_res) - vec2(1.0f + 1e-4f));
		const ivec2 pos_int = pos;
		const vec2 weight = pos - vec2(pos_int);

		ivec2 idx = clamp(pos_int, ivec2(0), resolution - ivec2(2));

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[local_img * compMul(error_map_res) + y * error_map_res.x + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), ivec2(0), sharpness_resolution - ivec2(1));
			float sharp = sharpness_data[img * compMul(sharpness_resolution) + sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x,   idx.y,   (1 - weight.x) * (1 - weight.y) * mean_loss);
		deposit_val(idx.x+1, idx.y,        weight.x  * (1 - weight.y) * mean_loss);
		deposit_val(idx.x,   idx.y+1, (1 - weight.x) *      weight.y  * mean_loss);
		deposit_val(idx.x+1, idx.y+1,      weight.x  *      weight.y  * mean_loss);
	}

	//loss_scale /= n_rays;
	float loss_scale_rgb = loss_scale / (float)total_rays;
	float loss_scale_depth = 0.0f;
	if (total_rays_depth > 0) {
		loss_scale_depth = loss_scale / (float)total_rays_depth;
	}

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;

	// NOTE: no density in SDF
	//const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;
	const float output_l1_reg_density = 1e-6f;

	// now do it again computing gradients
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {
		if (max_level_rand_training) {
			max_level_compacted_ptr[j] = max_level;
		}
		// Compact network inputs
		NerfCoordinate* coord_out = coords_out(j);
		const NerfCoordinate* coord_in = coords_in(j);
		coord_out->copy(*coord_in, coords_out.stride_in_bytes);

		const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
		float depth = distance(pos, ray_o);
		float cur_z = dot(cam_fwd, pos - ray_o);

        const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
        const float sdf = network_to_density(float(local_network_output[3]), density_activation);

        float weight = sigmoid(sdf / truncation_distance) * sigmoid(-sdf / truncation_distance);

		if ((found_surface) && (cur_z > (zmin + truncation_distance))) {
			weight = 0.;
		}

		const vec3 dloss_by_drgb = weight * lg.gradient / norm_weights;

		tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = (loss_scale_rgb/3.0f) * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = (loss_scale_rgb/3.0f) * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = (loss_scale_rgb/3.0f) * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);

		const float depth_supervision = depth_loss_gradient * ( depth - depth_ray);
		const float rgb_supervision = dot(lg.gradient, rgb - rgb_ray);

		float dloss_by_dmlp = density_derivative * (1/truncation_distance) * (1 / norm_weights) * weight
							* (sigmoid(-sdf / truncation_distance) - sigmoid(sdf / truncation_distance)) *
							 (loss_scale_rgb / 3.0 * rgb_supervision + loss_scale_depth * depth_supervision);

		//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
		//dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

		//adding free space supervision using depth
		if (add_free_space_loss) {
			if (target_z > 0.0f){
				if (n_steps_for_free_space_loss>0){
					if (cur_z < (target_z - truncation_distance)) {
						dloss_by_dmlp += loss_scale_depth * density_derivative * free_space_supervision_lambda * free_space_weight * 2 * (sdf - 1) / (float)numsteps;
					}
				}
			}
		}
		//adding SDF supervision using depth
		if (add_sdf_loss) {
			if (target_z > 0.0f){
				if (n_steps_for_sdf_loss > 0) {
					if ((cur_z >= (target_z - truncation_distance)) &
						(cur_z <= (target_z + truncation_distance))) {
							dloss_by_dmlp += loss_scale_depth * density_derivative * sdf_supervision_lambda * sdf_weight * 2 * (sdf*truncation_distance - (target_z - cur_z)) / (float)numsteps;
					}
				}
			}
		}

		local_dL_doutput[3] =
			//loss_scale * dloss_by_dmlp +
			dloss_by_dmlp +
			(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
			(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
			;

		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += padded_output_width;
	}

	if (exposure_gradient) {
		// Assume symmetric loss
		vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

		if (!train_in_linear_colors) {
			dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
		}

		// 2^exposure * log(2)
		vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
		atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
		atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
		atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
	}

	if (compacted_numsteps == numsteps && envmap_gradient) {
		vec3 loss_gradient = lg.gradient;
		if (envmap_loss_type != loss_type) {
			loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
		}

		vec3 dloss_by_dbackground = (1-norm_weights) * loss_gradient;
		if (!train_in_linear_colors) {
			dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
		}

		tcnn::vector_t<tcnn::network_precision_t, 4> dL_denvmap;
		dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
		dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
		dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;


		float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

		// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
		dL_denvmap[3] = (tcnn::network_precision_t)0;

		deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
	}
}








__global__ void compute_extra_dims_gradient_train_nerf_slam(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	const uint32_t* __restrict__ rays_counter,
	float* extra_dims_gradient,
	uint32_t n_extra_dims,
	const uint32_t n_training_images,
	const uint32_t* __restrict__ ray_indices_in,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords_gradient,
	const float* __restrict__ cdf_img,
	const uint32_t* __restrict__ idx_images_for_mapping
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}
	uint32_t base = numsteps_in[i*2+1];
	coords_gradient += base;
	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t local_img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);
	uint32_t img = idx_images_for_mapping[local_img];

	extra_dims_gradient += n_extra_dims * img;

	for (uint32_t j = 0; j < numsteps; ++j) {
		const float *src = coords_gradient(j)->get_extra_dims();
		for (uint32_t k = 0; k < n_extra_dims; ++k) {
			atomicAdd(&extra_dims_gradient[k], src[k]);
		}
	}
}



__global__ void compute_cam_gradient_train_nerf_slam(
	const uint32_t n_rays,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	const TrainingXForm* training_xforms,
	bool snap_to_pixel_centers,
	vec3* cam_pos_gradient,
	vec3* cam_rot_gradient,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords,
	PitchedPtr<NerfCoordinate> coords_gradient,
	float* __restrict__ distortion_gradient,
	float* __restrict__ distortion_gradient_weight,
	const ivec2 distortion_resolution,
	vec2* cam_focal_length_gradient,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const ivec2 sample_away_from_border_margin,
	const bool is_dir_encoding_empty
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}
	
	uint32_t base = numsteps_in[i*2+1];
	coords += base;
	coords_gradient += base;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	uint32_t local_img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img);
	uint32_t img = idx_images_for_mapping[local_img];
	ivec2 resolution = metadata[img].resolution;

	const mat4x3& xform = training_xforms[img].start;

	Ray ray = rays_in_unnormalized[i];
	ray.d = normalize(ray.d);
	Ray ray_gradient = { vec3(0.0f), vec3(0.0f) };

	// Compute ray gradient
	for (uint32_t j = 0; j < numsteps; ++j) {
		const vec3 warped_pos = coords(j)->pos.p;
		vec3 pos_gradient = coords_gradient(j)->pos.p * warp_position_derivative(warped_pos, aabb);

		// in case gradient are nan at this point
        if ( std::isnan(pos_gradient.x) or std::isnan(pos_gradient.y) or std::isnan(pos_gradient.z)
          or std::isinf(pos_gradient.x) or std::isinf(pos_gradient.y) or std::isinf(pos_gradient.z) ) {
                pos_gradient.x = 0.f;
                pos_gradient.y = 0.f;
                pos_gradient.z = 0.f;
        }

		ray_gradient.o += pos_gradient;
		const vec3 pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = distance(pos, ray.o);
		vec3 dir_gradient = coords_gradient(j)->dir.d * warp_direction_derivative(coords(j)->dir.d);

		if (is_dir_encoding_empty) {
			dir_gradient = {0.,0.,0.};
		}

		// in case gradient are nan at this point
        if ( std::isnan(dir_gradient.x) or std::isnan(dir_gradient.y) or std::isnan(dir_gradient.z)
          or std::isinf(dir_gradient.x) or std::isinf(dir_gradient.y) or std::isinf(dir_gradient.z) ) {
                dir_gradient.x = 0.f;
                dir_gradient.y = 0.f;
                dir_gradient.z = 0.f;
        }

		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());
	float uv_pdf = 1.0f;

	vec2 uv = nerf_random_image_pos_training_with_margin(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, sample_away_from_border_margin, &uv_pdf);
	//vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_res, img, &uv_pdf);

	if (distortion_gradient) {
		// Projection of the raydir gradient onto the plane normal to raydir,
		// because that's the only degree of motion that the raydir has.
		vec3 orthogonal_ray_gradient = ray_gradient.d - ray.d * dot(ray_gradient.d, ray.d);

		// Rotate ray gradient to obtain image plane gradient.
		// This has the effect of projecting the (already projected) ray gradient from the
		// tangent plane of the sphere onto the image plane (which is correct!).
		vec3 image_plane_gradient = inverse(mat3(xform)) * orthogonal_ray_gradient;

		// Splat the resulting 2D image plane gradient into the distortion params
		deposit_image_gradient(image_plane_gradient.xy() / uv_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, uv);
	}

	if (cam_pos_gradient) {
		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / uv_pdf);
		}
	}

	if (cam_rot_gradient) {
		// Rotation is averaged in log-space (i.e. by averaging angle-axes).
		// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
		// orthogonal, leading to the angle_axis magnitude to equal the magnitude
		// of ray_gradient.d.
		vec3 angle_axis = cross(ray.d, ray_gradient.d);

		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / uv_pdf);
		}
	}
}










void Testbed::train_nerf_slam(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {

	//adding safeguards if when using SDF
	if (m_use_volsdf_in_nerf||m_use_stylesdf_in_nerf||m_use_coslam_sdf_in_nerf||m_use_sdf_in_nerf){
		assert(m_nerf.density_activation==ENerfActivation::None);
	}
	if (m_add_sdf_loss || m_add_sdf_free_space_loss){
		assert(m_use_volsdf_in_nerf||m_use_stylesdf_in_nerf||m_use_coslam_sdf_in_nerf||m_use_sdf_in_nerf);
	}
	if (m_use_coslam_sdf_in_nerf){
		assert(!m_add_DSnerf_loss && !m_add_free_space_loss);
	}


	if (m_nerf.training.n_images_for_training == 0) {
		return;
	}

	if (m_nerf.training.include_sharpness_in_error) {
		size_t n_cells = NERF_GRID_N_CELLS() * NERF_CASCADES();
		if (m_nerf.training.sharpness_grid.size() < n_cells) {
			m_nerf.training.sharpness_grid.enlarge(NERF_GRID_N_CELLS() * NERF_CASCADES());
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		}

		if (m_training_step == 0) {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		} else {
			linear_kernel(decay_sharpness_grid_nerf, 0, stream, m_nerf.training.sharpness_grid.size(), 0.95f, m_nerf.training.sharpness_grid.data());
		}
	}

	if (m_nerf.training.m_set_fix_num_rays_to_sample) {
    	m_nerf.training.counters_rgb.rays_per_batch = m_nerf.training.m_target_num_rays;
	}
	m_nerf.training.counters_rgb.prepare_for_training_steps(stream);

	if (m_nerf.training.n_steps_since_cam_update == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(), 0, m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradient_weights(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(), 0, m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), stream));
	}

	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
	uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
	if (train_extra_dims) {
		uint32_t n = n_extra_dims * m_nerf.training.n_images_for_training;
		m_nerf.training.extra_dims_gradient_gpu.enlarge(n);
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.extra_dims_gradient_gpu.data(), 0, m_nerf.training.extra_dims_gradient_gpu.get_bytes(), stream));
	}

	if (m_nerf.training.n_steps_since_error_map_update == 0 && !m_nerf.training.dataset.metadata.empty()) {
		uint32_t n_samples_per_image = (m_nerf.training.n_steps_between_error_map_updates * m_nerf.training.counters_rgb.rays_per_batch) / m_nerf.training.idx_images_for_mapping.size();
		ivec2 res = m_nerf.training.dataset.metadata[0].resolution;
		m_nerf.training.error_map.resolution = min(ivec2((int)(std::sqrt(std::sqrt((float)n_samples_per_image)) * 3.5f)), res);
		m_nerf.training.error_map.data.resize(compMul(m_nerf.training.error_map.resolution) * m_nerf.training.idx_images_for_mapping.size());
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.data.data(), 0, m_nerf.training.error_map.data.get_bytes(), stream));
	}

	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
	if (envmap_gradient) {
		CUDA_CHECK_THROW(cudaMemsetAsync(envmap_gradient, 0, sizeof(float)*m_envmap.envmap->n_params(), stream));
	}

	//NOTE: send mapping image indices to device
	m_nerf.training.idx_images_for_mapping_gpu.enlarge(m_nerf.training.idx_images_for_mapping.size());
	CUDA_CHECK_THROW(
       cudaMemcpy(
          m_nerf.training.idx_images_for_mapping_gpu.data(),
          m_nerf.training.idx_images_for_mapping.data(),
          m_nerf.training.idx_images_for_mapping.size() * sizeof(uint32_t),
          cudaMemcpyHostToDevice
       )
    );

	train_nerf_slam_step(target_batch_size, m_nerf.training.counters_rgb, stream);

	m_trainer->optimizer_step(stream, LOSS_SCALE);

	++m_training_step;

	if (envmap_gradient) {
		m_envmap.trainer->optimizer_step(stream, LOSS_SCALE);
	}

	float loss_scalar = m_nerf.training.counters_rgb.update_after_training(target_batch_size, get_loss_scalar, stream);
	bool zero_records = m_nerf.training.counters_rgb.measured_batch_size == 0;
	if (get_loss_scalar) {
		m_loss_scalar.update(loss_scalar);
	}

	if (zero_records) {
		m_loss_scalar.set(0.f);
		tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
		m_train = false;
	}

	// Compute CDFs from the error map
	m_nerf.training.n_steps_since_error_map_update += 1;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;
	if (accumulate_error && m_nerf.training.n_steps_since_error_map_update >= m_nerf.training.n_steps_between_error_map_updates) {
		m_nerf.training.error_map.cdf_resolution = m_nerf.training.error_map.resolution;
		m_nerf.training.error_map.cdf_x_cond_y.resize(compMul(m_nerf.training.error_map.cdf_resolution) * m_nerf.training.idx_images_for_mapping.size());
		m_nerf.training.error_map.cdf_y.resize(m_nerf.training.error_map.cdf_resolution.y * m_nerf.training.idx_images_for_mapping.size());
		m_nerf.training.error_map.cdf_img.resize(m_nerf.training.idx_images_for_mapping.size());

		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_x_cond_y.data(), 0, m_nerf.training.error_map.cdf_x_cond_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_y.data(), 0, m_nerf.training.error_map.cdf_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_img.data(), 0, m_nerf.training.error_map.cdf_img.get_bytes(), stream));

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)m_nerf.training.error_map.cdf_resolution.y, threads.x), div_round_up((uint32_t)m_nerf.training.idx_images_for_mapping.size(), threads.y), 1 };
		construct_cdf_2d<<<blocks, threads, 0, stream>>>(
			m_nerf.training.idx_images_for_mapping.size(), m_nerf.training.error_map.cdf_resolution.y, m_nerf.training.error_map.cdf_resolution.x,
			m_nerf.training.error_map.data.data(),
			m_nerf.training.error_map.cdf_x_cond_y.data(),
			m_nerf.training.error_map.cdf_y.data()
		);
		linear_kernel(construct_cdf_1d, 0, stream,
			m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.error_map.cdf_resolution.y,
			m_nerf.training.error_map.cdf_y.data(),
			m_nerf.training.error_map.cdf_img.data()
		);

		// Compute image CDF on the CPU. It's single-threaded anyway. No use parallelizing.
		m_nerf.training.error_map.pmf_img_cpu.resize(m_nerf.training.error_map.cdf_img.size());
		m_nerf.training.error_map.cdf_img.copy_to_host(m_nerf.training.error_map.pmf_img_cpu);
		std::vector<float> cdf_img_cpu = m_nerf.training.error_map.pmf_img_cpu; // Copy unnormalized PDF into CDF buffer
		float cum = 0;
		for (float& f : cdf_img_cpu) {
			cum += f;
			f = cum;
		}
		float norm = 1.0f / cum;
		for (size_t i = 0; i < cdf_img_cpu.size(); ++i) {
			constexpr float MIN_PMF = 0.1f;
			m_nerf.training.error_map.pmf_img_cpu[i] = (1.0f - MIN_PMF) * m_nerf.training.error_map.pmf_img_cpu[i] * norm + MIN_PMF / (float)m_nerf.training.idx_images_for_mapping.size();
			cdf_img_cpu[i] = (1.0f - MIN_PMF) * cdf_img_cpu[i] * norm + MIN_PMF * (float)(i+1) / (float)m_nerf.training.idx_images_for_mapping.size();
		}
		m_nerf.training.error_map.cdf_img.copy_from_host(cdf_img_cpu);

		// Reset counters and decrease update rate.
		m_nerf.training.n_steps_since_error_map_update = 0;
		m_nerf.training.n_rays_since_error_map_update = 0;
		m_nerf.training.error_map.is_cdf_valid = true;

		//NOTE: in SLAM mode the mapping frames changes over time.
		//m_nerf.training.n_steps_between_error_map_updates = (uint32_t)(m_nerf.training.n_steps_between_error_map_updates * 1.5f);
	}

	// Get extrinsics gradients
	m_nerf.training.n_steps_since_cam_update += 1;

	if (train_extra_dims) {
		std::vector<float> extra_dims_gradient(m_nerf.training.extra_dims_gradient_gpu.size());
		m_nerf.training.extra_dims_gradient_gpu.copy_to_host(extra_dims_gradient);

		// Optimization step
		for (uint32_t x = 0; x < m_nerf.training.idx_images_for_mapping.size(); ++x) {
			uint32_t i = m_nerf.training.idx_images_for_mapping[x];
			std::vector<float> gradient(n_extra_dims);
			for (uint32_t j = 0; j < n_extra_dims; ++j) {
				gradient[j] = extra_dims_gradient[i * n_extra_dims + j] / LOSS_SCALE;
			}

			//float l2_reg = 1e-4f;
			//gradient += m_nerf.training.extra_dims_opt[i].variable() * l2_reg;

			m_nerf.training.extra_dims_opt[i].set_learning_rate(m_optimizer->learning_rate());
			m_nerf.training.extra_dims_opt[i].step(gradient);
		}

		m_nerf.training.update_extra_dims();
	}

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length || m_nerf.training.optimize_exposure;
	if (train_camera && m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates) {
		//float per_camera_loss_scale = (float)m_nerf.training.idx_images_for_training_extrinsics.size() / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;
		float per_camera_loss_scale = (float)m_nerf.training.idx_images_for_mapping.size() / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;

        if (m_nerf.training.optimize_extrinsics) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// Optimization step
			for (uint32_t k = 0; k < m_nerf.training.idx_images_for_training_extrinsics.size(); ++k) {
				uint32_t i = m_nerf.training.idx_images_for_training_extrinsics[k];

				vec3 pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
				vec3 rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

				float l2_reg = m_nerf.training.extrinsic_l2_reg;
				pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
				rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

				if (m_nerf.training.m_use_pose_scheduler_in_mapping){
					m_nerf.training.cam_pos_offset[i].set_learning_rate(
						std::max(m_nerf.training.extrinsic_learning_rate_ba_pos * std::pow(m_nerf.training.m_pose_scheduler_coef, (float)(m_nerf.training.cam_pos_offset[i].step() / m_nerf.training.m_pose_scheduler_norm)),
						         m_optimizer->learning_rate()/1000.0f)
					);
					m_nerf.training.cam_rot_offset[i].set_learning_rate(
						std::max(m_nerf.training.extrinsic_learning_rate_ba_rot * std::pow(m_nerf.training.m_pose_scheduler_coef, (float)(m_nerf.training.cam_rot_offset[i].step() / m_nerf.training.m_pose_scheduler_norm)),
								 m_optimizer->learning_rate()/1000.0f)
					);
				}
				else {
					m_nerf.training.cam_pos_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_ba_pos);
					m_nerf.training.cam_rot_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_ba_rot);
				}

				if ( std::isnan(pos_gradient.x) or std::isnan(pos_gradient.y) or std::isnan(pos_gradient.z) or
				     std::isnan(rot_gradient.x) or std::isnan(rot_gradient.y) or std::isnan(rot_gradient.z) ) {
   	            	 pos_gradient.x = 0.f;
   	            	 pos_gradient.y = 0.f;
   	            	 pos_gradient.z = 0.f;
   	            	 rot_gradient.x = 0.f;
   	            	 rot_gradient.y = 0.f;
   	            	 rot_gradient.z = 0.f;
   	         	}

                if (m_nerf.training.m_use_gradient_clipping_for_extrinsics) {
   	         	    float max_step_pos = 100;
   	         	    float max_step_rot = 100;
   	         	    // pos_gradient.x = max( -max_step_pos, min(pos_gradient.x, max_step_pos) );
   	         	    // pos_gradient.y = max( -max_step_pos, min(pos_gradient.y, max_step_pos) );
   	         	    // pos_gradient.z = max( -max_step_pos, min(pos_gradient.z, max_step_pos) );

   	         	    // rot_gradient.x = max( -max_step_rot, min(rot_gradient.x, max_step_rot) );
   	         	    // rot_gradient.y = max( -max_step_rot, min(rot_gradient.y, max_step_rot) );
   	         	    // rot_gradient.z = max( -max_step_rot, min(rot_gradient.z, max_step_rot) );

				    float norm_pos = glm::length(pos_gradient);
				    float norm_rot = glm::length(rot_gradient);
				    if (norm_pos > max_step_pos){
				    		pos_gradient = max_step_pos * pos_gradient / norm_pos;
				    }
				    if (norm_rot > max_step_rot){
				    		rot_gradient = max_step_rot * rot_gradient / norm_rot;
				    }
                }

				m_nerf.training.cam_pos_offset[i].step(pos_gradient);
				m_nerf.training.cam_rot_offset[i].step(rot_gradient);

				m_nerf.training.update_transforms(i, i+1);
			}

		}

		if (m_nerf.training.optimize_distortion) {
			linear_kernel(safe_divide, 0, stream,
				m_distortion.map->n_params(),
				m_distortion.map->gradients(),
				m_distortion.map->gradient_weights()
			);
			m_distortion.trainer->optimizer_step(stream, LOSS_SCALE*(float)m_nerf.training.n_steps_between_cam_updates);
		}

		if (m_nerf.training.optimize_focal_length) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(&m_nerf.training.cam_focal_length_gradient, m_nerf.training.cam_focal_length_gradient_gpu.data(), m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			vec2 focal_length_gradient = m_nerf.training.cam_focal_length_gradient * per_camera_loss_scale;
			float l2_reg = m_nerf.training.intrinsic_l2_reg;
			focal_length_gradient += m_nerf.training.cam_focal_length_offset.variable() * l2_reg;
			m_nerf.training.cam_focal_length_offset.set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.cam_focal_length_offset.step() / 128)),m_optimizer->learning_rate() / 1000.0f));
			m_nerf.training.cam_focal_length_offset.step(focal_length_gradient);
			m_nerf.training.dataset.update_metadata();
		}

		if (m_nerf.training.optimize_exposure) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gradient.data(), m_nerf.training.cam_exposure_gradient_gpu.data(), m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			vec3 mean_exposure = vec3(0.0f);

			// Optimization step
			for (uint32_t k = 0; k < m_nerf.training.idx_images_for_training_extrinsics.size(); ++k) {
				uint32_t i = m_nerf.training.idx_images_for_training_extrinsics[k];

				vec3 gradient = m_nerf.training.cam_exposure_gradient[i] * per_camera_loss_scale;

				float l2_reg = m_nerf.training.exposure_l2_reg;
				gradient += m_nerf.training.cam_exposure[i].variable() * l2_reg;

				m_nerf.training.cam_exposure[i].set_learning_rate(m_optimizer->learning_rate());
				m_nerf.training.cam_exposure[i].step(gradient);

				mean_exposure += m_nerf.training.cam_exposure[i].variable();
			}

			mean_exposure /= m_nerf.training.idx_images_for_training_extrinsics.size();

			// Renormalize
			std::vector<vec3> cam_exposures(m_nerf.training.n_images_for_training);
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				cam_exposures[i] = m_nerf.training.cam_exposure[i].variable();
			}
			for (uint32_t k = 0; k < m_nerf.training.idx_images_for_training_extrinsics.size(); ++k) {
				uint32_t i = m_nerf.training.idx_images_for_training_extrinsics[k];
				cam_exposures[i] = m_nerf.training.cam_exposure[i].variable() -= mean_exposure;
			}

			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gpu.data(), cam_exposures.data(), m_nerf.training.n_images_for_training * sizeof(vec3), cudaMemcpyHostToDevice, stream));
		}

		m_nerf.training.n_steps_since_cam_update = 0;
	}
}

void Testbed::train_nerf_slam_step(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct
	const ivec2 sample_away_from_border_margin(m_nerf.training.m_sample_away_from_border_margin_h_mapping, m_nerf.training.m_sample_away_from_border_margin_w_mapping);

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
		uint32_t // ray_counter_depth
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
		1
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
	uint32_t* ray_counter_depth = std::get<11>(scratch);

	uint32_t max_inference=max_samples;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference;
	}
	//if (counters.measured_batch_size_before_compaction == 0) {
	//	counters.measured_batch_size_before_compaction = max_inference = max_samples;
	//} else {
	//	max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	//}

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_training_step == 0) {
		counters.n_rays_total = 0;
	}

	uint32_t n_rays_total = counters.n_rays_total;
	counters.n_rays_total += counters.rays_per_batch;
	m_nerf.training.n_rays_since_error_map_update += counters.rays_per_batch;

	// If we have an envmap, prepare its gradient buffer
	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;

	bool sample_focal_plane_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_focal_plane_proportional_to_error;
	bool sample_image_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
	bool include_sharpness_in_error = m_nerf.training.include_sharpness_in_error;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter_depth, 0, sizeof(uint32_t), stream));

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

		//DEBUG
		//DEBUG
		tcnn::GPUMemory<float> xy_coords_sample;
		tcnn::GPUMemory<uint32_t> img_id_sample;
		tcnn::GPUMemory<float> sample_z_vals;
		if (m_debug) {
			xy_coords_sample.enlarge(counters.rays_per_batch * 2);
			img_id_sample.enlarge(counters.rays_per_batch);
			sample_z_vals.enlarge(max_samples);
		}
		//DEBUG
		//DEBUG

		//NOTE: we're not using density_grid in SDF mode
		// This is a bit hacky. -> Fix density grids with SDF as TODO
		linear_kernel(generate_training_samples_nerf_slam, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			max_inference,
			n_rays_total,
			m_rng,
			ray_counter,
			ray_counter_depth,
			counters.numsteps_counter.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.dataset.metadata_gpu.data(),
			m_nerf.training.transforms_gpu.data(),
			((!m_use_sdf_in_nerf) && m_use_density_in_nerf_sampling) ? m_nerf.density_grid_bitfield.data() : nullptr, // --> DO NOT use density grid in SDF
			m_nerf.max_cascade,
			m_max_level_rand_training,
			max_level,
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.train_envmap,
			m_nerf.cone_angle_constant,
			m_distortion.view(),
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			m_nerf.training.extra_dims_gpu.data(),
			m_nerf_network->n_extra_dims(),
			m_nerf.training.idx_images_for_mapping_gpu.data(),
			sample_away_from_border_margin,
			m_max_grid_level_factor,
			m_use_depth_guided_sampling,
			m_nerf.training.truncation_distance_for_depth_guided_sampling,
			m_nerf.training.dt_for_depth_guided_sampling,
			m_use_custom_ray_marching,
			m_nerf.training.dt_for_regular_sampling,
			m_nerf.training.n_samples_for_regular_sampling,
			m_debug ? xy_coords_sample.data(): nullptr,
			m_debug ? img_id_sample.data(): nullptr,
			m_debug ? sample_z_vals.data(): nullptr
		);

	//DEBUG
	//DEBUG
	uint32_t m_track_pose_nerf_num_rays_in_tracking_step = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&m_track_pose_nerf_num_rays_in_tracking_step,std::get<10>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	m_ray_counter = m_track_pose_nerf_num_rays_in_tracking_step;
	m_rays_per_batch = counters.rays_per_batch;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&m_ray_counter_depth,std::get<11>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );

	// -- get sample coordinates for each ray
	if (m_debug) {
		m_coords_cpu.resize(max_samples * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_cpu.data(),std::get<3>(scratch), max_samples * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
		m_numsteps_cpu.resize(counters.rays_per_batch * 2);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_numsteps_cpu.data(),std::get<2>(scratch), counters.rays_per_batch * 2 * sizeof(uint32_t),cudaMemcpyDeviceToHost)
 	   );
		m_image_id_sample.resize(counters.rays_per_batch);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_image_id_sample.data(), img_id_sample.data(), counters.rays_per_batch * sizeof(uint32_t),cudaMemcpyDeviceToHost)
 	   );
		m_xy_image_pixel_indices_float.resize(counters.rays_per_batch*2);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_xy_image_pixel_indices_float.data(), xy_coords_sample.data(), counters.rays_per_batch * 2 * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
		m_sample_z_vals_cpu.resize(max_samples);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_sample_z_vals_cpu.data(),sample_z_vals.data(), max_samples * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}
	//DEBUG
	//DEBUG


		if (hg_enc) {
			hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
		}

		GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
		GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
		m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

		if (hg_enc) {
			hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
		}


		//DEBUG
		//DEBUG
		tcnn::GPUMemory<float> sample_outputs;
		if (m_debug) {
			sample_outputs.enlarge(max_samples * 4);
		}
		//DEBUG
		//DEBUG


		
		tcnn::GPUMemory<float> rays_depth_variance;
		if (!m_use_sdf_in_nerf && m_nerf.training.use_depth_var_in_tracking_loss) {
			rays_depth_variance.enlarge(counters.rays_per_batch);
		}


        if (m_use_sdf_in_nerf) {
		    linear_kernel(compute_loss_kernel_train_nerf_sdf_slam, 0, stream,
		    	counters.rays_per_batch,
		    	m_aabb,
		    	n_rays_total,
		    	m_rng,
		    	target_batch_size,
		    	ray_counter,
		    	ray_counter_depth,
		    	LOSS_SCALE,
		    	padded_output_width,
		    	m_envmap.view(),
		    	envmap_gradient,
		    	m_envmap.resolution,
		    	m_envmap.loss_type,
		    	m_background_color.rgb,
		    	m_color_space,
		    	m_nerf.training.random_bg_color,
		    	m_nerf.training.linear_colors,
		    	m_nerf.training.idx_images_for_mapping.size(),
		    	m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
		    	mlp_out,
		    	counters.numsteps_counter_compacted.data(),
		    	ray_indices,
		    	rays_unnormalized,
		    	numsteps,
		    	PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		    	PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
		    	dloss_dmlp_out,
		    	m_nerf.training.loss_type,
		    	m_nerf.training.depth_loss_type,
		    	counters.loss.data(),
		    	m_max_level_rand_training,
		    	max_level_compacted,
		    	m_nerf.rgb_activation,
		    	m_nerf.density_activation,
		    	m_nerf.training.snap_to_pixel_centers,
		    	accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
		    	sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
		    	sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
		    	sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
		    	m_nerf.training.error_map.resolution,
		    	m_nerf.training.error_map.cdf_resolution,
		    	include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
		    	m_nerf.training.dataset.sharpness_resolution,
		    	m_nerf.training.sharpness_grid.data(),
		    	m_nerf.density_grid.data(),
		    	m_nerf.density_grid_mean.data(),
		    	m_nerf.max_cascade,
		    	m_nerf.training.cam_exposure_gpu.data(),
		    	m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
		    	m_nerf.training.depth_supervision_lambda,
		    	m_nerf.training.near_distance,
		    	m_nerf.training.idx_images_for_mapping_gpu.data(),
		    	sample_away_from_border_margin,
		    	false,
		    	m_max_grid_level_factor,
		    	m_add_free_space_loss,
		    	m_nerf.training.free_space_supervision_lambda,
		    	m_nerf.training.free_space_supervision_distance,
		    	m_nerf.training.truncation_distance,
		    	m_add_sdf_loss,
		    	m_nerf.training.sdf_supervision_lambda
		    );

        } else {
		    linear_kernel(compute_loss_kernel_train_nerf_slam, 0, stream,
		    	counters.rays_per_batch,
		    	m_aabb,
		    	n_rays_total,
		    	m_rng,
		    	target_batch_size,
		    	ray_counter,
		    	ray_counter_depth,
		    	LOSS_SCALE,
		    	padded_output_width,
		    	m_envmap.view(),
		    	envmap_gradient,
		    	m_envmap.resolution,
		    	m_envmap.loss_type,
		    	m_background_color.rgb,
		    	m_color_space,
		    	m_nerf.training.random_bg_color,
		    	m_nerf.training.linear_colors,
		    	m_nerf.training.idx_images_for_mapping.size(),
		    	m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
		    	mlp_out,
		    	counters.numsteps_counter_compacted.data(),
		    	ray_indices,
		    	rays_unnormalized,
		    	numsteps,
		    	PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		    	PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
		    	dloss_dmlp_out,
		    	m_nerf.training.loss_type,
		    	m_nerf.training.depth_loss_type,
		    	counters.loss.data(),
		    	m_max_level_rand_training,
		    	max_level_compacted,
		    	m_nerf.rgb_activation,
		    	m_nerf.density_activation,
		    	m_nerf.training.snap_to_pixel_centers,
		    	accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
		    	sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
		    	sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
		    	sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
		    	m_nerf.training.error_map.resolution,
		    	m_nerf.training.error_map.cdf_resolution,
		    	include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
		    	m_nerf.training.dataset.sharpness_resolution,
		    	m_nerf.training.sharpness_grid.data(),
		    	m_nerf.density_grid.data(),
		    	m_nerf.density_grid_mean.data(),
		    	m_nerf.max_cascade,
		    	m_nerf.training.cam_exposure_gpu.data(),
		    	m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
		    	m_nerf.training.depth_supervision_lambda,
		    	m_nerf.training.rgb_supervision_lambda,
		    	m_nerf.training.near_distance,
		    	m_nerf.training.idx_images_for_mapping_gpu.data(),
		    	sample_away_from_border_margin,
		    	false,
		    	m_max_grid_level_factor,
		    	m_add_free_space_loss,
		    	m_nerf.training.free_space_supervision_lambda,
		    	m_nerf.training.free_space_supervision_distance,
				m_debug ? sample_outputs.data(): nullptr,
				m_use_custom_ray_marching,
				((!m_use_sdf_in_nerf) && m_nerf.training.use_depth_var_in_tracking_loss) ? rays_depth_variance.data() : nullptr,
		    	m_add_DSnerf_loss,
		    	m_nerf.training.DS_nerf_supervision_lambda,
		    	m_nerf.training.DS_nerf_supervision_depth_sigma,
				m_use_stylesdf_in_nerf,
				m_use_volsdf_in_nerf,
				m_use_coslam_sdf_in_nerf,
				m_nerf.training.truncation_distance,
				m_nerf.training.volsdf_beta,
		    	m_add_sdf_loss,
		    	m_nerf.training.sdf_supervision_lambda,
		    	m_add_sdf_free_space_loss,
		    	m_nerf.training.sdf_free_space_supervision_lambda
		    );
        }

	if (m_debug) {
		m_numsteps_compacted_cpu.resize(counters.rays_per_batch * 2);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_numsteps_compacted_cpu.data(),std::get<2>(scratch), counters.rays_per_batch * 2 * sizeof(uint32_t),cudaMemcpyDeviceToHost)
 	   );
		m_sample_outputs_cpu.resize(max_samples*4);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_sample_outputs_cpu.data(),sample_outputs.data(), max_samples * 4 * sizeof(float), cudaMemcpyDeviceToHost)
 	   );
	}
	//DEBUG
	//DEBUG


	fill_rollover_and_rescale<network_precision_t><<<n_blocks_linear(target_batch_size*padded_output_width), n_threads_linear, 0, stream>>>(
		target_batch_size, padded_output_width, counters.numsteps_counter_compacted.data(), dloss_dmlp_out
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size * floats_per_coord), n_threads_linear, 0, stream>>>(
		target_batch_size, floats_per_coord, counters.numsteps_counter_compacted.data(), (float*)coords_compacted
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, counters.numsteps_counter_compacted.data(), max_level_compacted
	);

	if (m_debug) {
		m_coords_compacted_filled_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_compacted_filled_cpu.data(),std::get<7>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}
	

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length;
	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
	bool prepare_input_gradients = train_camera || train_extra_dims;
	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);

	{
		auto ctx = m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, prepare_input_gradients);
		m_network->backward(stream, *ctx, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, EGradientMode::Overwrite);
	}


	if (m_debug) {
		m_coords_gradient_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_gradient_cpu.data(),std::get<8>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}
	

	if (train_extra_dims) {
		// Compute extra-dim gradients
		linear_kernel(compute_extra_dims_gradient_train_nerf_slam, 0, stream,
			counters.rays_per_batch,
			n_rays_total,
			ray_counter,
			m_nerf.training.extra_dims_gradient_gpu.data(),
			m_nerf.training.dataset.n_extra_dims(),
			m_nerf.training.idx_images_for_mapping.size(),
			ray_indices,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.idx_images_for_mapping_gpu.data()
		);
	}

	std::string dir_encoding = m_network_config["dir_encoding"].value("otype", "none");
	bool is_dir_encoding_empty = dir_encoding == "Empty";

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf_slam, 0, stream,
			counters.rays_per_batch,
			n_rays_total,
			m_rng,
			m_aabb,
			ray_counter,
			m_nerf.training.transforms_gpu.data(),
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_pos_gradient_gpu.data() : nullptr,
			m_nerf.training.optimize_extrinsics ? m_nerf.training.cam_rot_gradient_gpu.data() : nullptr,
			m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.dataset.metadata_gpu.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
			m_distortion.resolution,
			m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			m_nerf.training.idx_images_for_mapping_gpu.data(),
			sample_away_from_border_margin,
			is_dir_encoding_empty
		);
	}

	m_rng.advance();

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}
}





















/*
##############################################################################################
##############################################################################################
			Tracking
##############################################################################################
##############################################################################################
*/





void Testbed::train_nerf_slam_tracking(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	//adding safeguards if when using SDF
	if (m_use_volsdf_in_nerf||m_use_stylesdf_in_nerf||m_use_coslam_sdf_in_nerf||m_use_sdf_in_nerf){
		assert(m_nerf.density_activation==ENerfActivation::None);
	}
	if (m_add_sdf_loss_tracking || m_add_sdf_free_space_loss_tracking){
		assert(m_use_volsdf_in_nerf||m_use_stylesdf_in_nerf||m_use_coslam_sdf_in_nerf||m_use_sdf_in_nerf);
	}
	if (m_use_coslam_sdf_in_nerf){
		assert(!m_add_DSnerf_loss_tracking && !m_add_free_space_loss_tracking);
	}


	if (m_nerf.training.indice_image_for_tracking_pose == 0) {
        // no tracking for first frame.
		return;
	}

	if (m_nerf.training.m_set_fix_num_rays_to_sample) {
    	m_nerf.training.counters_rgb_tracking.rays_per_batch = m_nerf.training.m_target_num_rays_for_tracking;
	}
	m_nerf.training.counters_rgb_tracking.prepare_for_training_steps(stream);

	if (m_nerf.training.n_steps_since_cam_update == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
	}

	if (m_tracking_mode==0)	{
		train_nerf_slam_tracking_step(target_batch_size, m_nerf.training.counters_rgb_tracking, stream);
	} else if (m_tracking_mode==1) {
		train_nerf_slam_tracking_step_with_gaussian_pyramid(target_batch_size, m_nerf.training.counters_rgb_tracking, stream);
	} else if (m_tracking_mode==2) {
		train_nerf_slam_tracking_step_mgl_coarse_to_fine(target_batch_size, m_nerf.training.counters_rgb_tracking, stream);
	} else {
		throw std::runtime_error{"Invalid tracking mode."};
	}

	++m_tracking_step;

	float loss_scalar = m_nerf.training.counters_rgb_tracking.update_after_training(target_batch_size, get_loss_scalar, stream);
	bool zero_records = m_nerf.training.counters_rgb_tracking.measured_batch_size == 0;
	if (get_loss_scalar) {
		m_loss_scalar_tracking.update(loss_scalar);
	}

	if (zero_records) {
		m_loss_scalar_tracking.set(0.f);
		tlog::warning() << "Nerf training generated 0 samples. Aborting training (while tracking).";
		m_train = false;
	}
	// Get extrinsics gradients
	m_nerf.training.n_steps_since_cam_update += 1;

	if (m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates) {
		float per_camera_loss_scale = 1.0f / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;

		{
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// Optimization step
            uint32_t i = m_nerf.training.indice_image_for_tracking_pose;

			vec3 pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
			vec3 rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

			m_pos_gradient = pos_gradient;
			m_rot_gradient = rot_gradient;

			float l2_reg = m_nerf.training.extrinsic_l2_reg;
			pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
			rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

			// m_nerf.training.cam_pos_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_pos_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));
			// m_nerf.training.cam_rot_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_rot_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));
			m_nerf.training.cam_pos_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_pos);
			m_nerf.training.cam_rot_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_rot);

            if ( std::isnan(pos_gradient.x) or std::isnan(pos_gradient.y) or std::isnan(pos_gradient.z) or
			     std::isnan(rot_gradient.x) or std::isnan(rot_gradient.y) or std::isnan(rot_gradient.z) ) {
				tlog::info()<<"WARNING -> NaN values found in gradients";
                pos_gradient.x = 0.f;
                pos_gradient.y = 0.f;
                pos_gradient.z = 0.f;
                rot_gradient.x = 0.f;
                rot_gradient.y = 0.f;
                rot_gradient.z = 0.f;
            }

            if (m_nerf.training.m_use_gradient_clipping_for_extrinsics) {
                float max_step_pos = 100;
                float max_step_rot = 100;
                // pos_gradient.x = max( -max_step_pos, min(pos_gradient.x, max_step_pos) );
                // pos_gradient.y = max( -max_step_pos, min(pos_gradient.y, max_step_pos) );
                // pos_gradient.z = max( -max_step_pos, min(pos_gradient.z, max_step_pos) );

                // rot_gradient.x = max( -max_step_rot, min(rot_gradient.x, max_step_rot) );
                // rot_gradient.y = max( -max_step_rot, min(rot_gradient.y, max_step_rot) );
                // rot_gradient.z = max( -max_step_rot, min(rot_gradient.z, max_step_rot) );

			    float norm_pos = glm::length(pos_gradient);
                float norm_rot = glm::length(rot_gradient);
			    if (norm_pos > max_step_pos){
			    		pos_gradient = max_step_pos * pos_gradient / norm_pos;
			    }
			    if (norm_rot > max_step_rot){
			    		rot_gradient = max_step_rot * rot_gradient / norm_rot;
			    }
            }

			m_nerf.training.cam_pos_offset[i].step(pos_gradient);
			m_nerf.training.cam_rot_offset[i].step(rot_gradient);

			m_nerf.training.update_transforms(i, i+1);
		}
		m_nerf.training.n_steps_since_cam_update = 0;
	}
}

void Testbed::train_nerf_slam_tracking_step(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct
	const ivec2 sample_away_from_border_margin(m_nerf.training.m_sample_away_from_border_margin_h_tracking, m_nerf.training.m_sample_away_from_border_margin_w_tracking);

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
		uint32_t // ray_counter_depth
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
		1
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
	uint32_t* ray_counter_depth = std::get<11>(scratch);

	uint32_t max_inference=max_samples;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference;
	}
	//if (counters.measured_batch_size_before_compaction == 0) {
	//	counters.measured_batch_size_before_compaction = max_inference = max_samples;
	//} else {
	//	max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	//}

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_tracking_step == 0) {
		counters.n_rays_total = 0;
	}

	uint32_t n_rays_total = counters.n_rays_total;
	counters.n_rays_total += counters.rays_per_batch;

	float* envmap_gradient = nullptr;
	bool sample_focal_plane_proportional_to_error = false;
	bool sample_image_proportional_to_error = false;
	bool include_sharpness_in_error = false;
	bool accumulate_error = false;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter_depth, 0, sizeof(uint32_t), stream));

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

	std::vector<uint32_t> indice_image_for_tracking_pose = {m_nerf.training.indice_image_for_tracking_pose};
	tcnn::GPUMemory<uint32_t> indice_image_for_tracking_pose_gpu;
	indice_image_for_tracking_pose_gpu.enlarge(1);
	CUDA_CHECK_THROW(
 		cudaMemcpy(indice_image_for_tracking_pose_gpu.data(),indice_image_for_tracking_pose.data(),indice_image_for_tracking_pose.size() * sizeof(uint32_t),cudaMemcpyHostToDevice)
 	);

		linear_kernel(generate_training_samples_nerf_slam, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			max_inference,
			n_rays_total,
			m_rng,
			ray_counter,
			ray_counter_depth,
			counters.numsteps_counter.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			1,
			m_nerf.training.dataset.metadata_gpu.data(),
			m_nerf.training.transforms_gpu.data(),
			((!m_use_sdf_in_nerf) && m_use_density_in_nerf_sampling) ? m_nerf.density_grid_bitfield.data() : nullptr, // --> DO NOT use density grid in SDF
			m_nerf.max_cascade,
			m_max_level_rand_training,
			max_level,
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.train_envmap,
			m_nerf.cone_angle_constant,
			m_distortion.view(),
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			m_nerf.training.extra_dims_gpu.data(),
			m_nerf_network->n_extra_dims(),
			indice_image_for_tracking_pose_gpu.data(),
			sample_away_from_border_margin,
			m_max_grid_level_factor,
			m_use_depth_guided_sampling,
			m_nerf.training.truncation_distance_for_depth_guided_sampling,
			m_nerf.training.dt_for_depth_guided_sampling,
			m_use_custom_ray_marching,
			m_nerf.training.dt_for_regular_sampling,
			m_nerf.training.n_samples_for_regular_sampling,
			nullptr,
			nullptr,
			nullptr
		);


		//DEBUG
		//DEBUG
		uint32_t m_track_pose_nerf_num_rays_in_tracking_step = 0;
		CUDA_CHECK_THROW(
 	      cudaMemcpyAsync(&m_track_pose_nerf_num_rays_in_tracking_step,std::get<10>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
 	   	);
		m_ray_counter = m_track_pose_nerf_num_rays_in_tracking_step;
		m_rays_per_batch = counters.rays_per_batch;
		CUDA_CHECK_THROW(
 	      cudaMemcpyAsync(&m_ray_counter_depth,std::get<11>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
 	   	);


		if (hg_enc) {
			hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level : nullptr);
		}

		GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
		GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
		m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

		if (hg_enc) {
			hg_enc->set_max_level_gpu(m_max_level_rand_training ? max_level_compacted : nullptr);
		}

	
		tcnn::GPUMemory<float> rays_depth_variance;
		if (!m_use_sdf_in_nerf && m_nerf.training.use_depth_var_in_tracking_loss) {
			rays_depth_variance.enlarge(counters.rays_per_batch);
		}


        if (m_use_sdf_in_nerf) {
			linear_kernel(compute_loss_kernel_train_nerf_sdf_slam, 0, stream,
				counters.rays_per_batch,
				m_aabb,
				n_rays_total,
				m_rng,
				target_batch_size,
				ray_counter,
				ray_counter_depth,
				LOSS_SCALE,
				padded_output_width,
				m_envmap.view(),
				envmap_gradient,
				m_envmap.resolution,
				m_envmap.loss_type,
				m_background_color.rgb,
				m_color_space,
				m_nerf.training.random_bg_color,
				m_nerf.training.linear_colors,
				1,
				m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
				mlp_out,
				counters.numsteps_counter_compacted.data(),
				ray_indices,
				rays_unnormalized,
				numsteps,
				PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
				dloss_dmlp_out,
				m_nerf.training.loss_type,
				m_nerf.training.depth_loss_type,
				counters.loss.data(),
				m_max_level_rand_training,
				max_level_compacted,
				m_nerf.rgb_activation,
				m_nerf.density_activation,
				m_nerf.training.snap_to_pixel_centers,
				accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
				sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
				m_nerf.training.error_map.resolution,
				m_nerf.training.error_map.cdf_resolution,
				include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
				m_nerf.training.dataset.sharpness_resolution,
				m_nerf.training.sharpness_grid.data(),
				m_nerf.density_grid.data(),
				m_nerf.density_grid_mean.data(),
				m_nerf.max_cascade,
				m_nerf.training.cam_exposure_gpu.data(),
				m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
				m_nerf.training.depth_supervision_lambda,
				m_nerf.training.near_distance,
				indice_image_for_tracking_pose_gpu.data(),
				sample_away_from_border_margin,
				m_nerf.training.use_depth_var_in_tracking_loss,
				m_max_grid_level_factor,
		    	m_add_free_space_loss_tracking,
				m_nerf.training.free_space_supervision_lambda_tracking,
				m_nerf.training.free_space_supervision_distance,
			    m_nerf.training.truncation_distance,
		    	m_add_sdf_loss_tracking,
			    m_nerf.training.sdf_supervision_lambda
			);
		}
		else {
			linear_kernel(compute_loss_kernel_train_nerf_slam, 0, stream,
				counters.rays_per_batch,
				m_aabb,
				n_rays_total,
				m_rng,
				target_batch_size,
				ray_counter,
				ray_counter_depth,
				LOSS_SCALE,
				padded_output_width,
				m_envmap.view(),
				envmap_gradient,
				m_envmap.resolution,
				m_envmap.loss_type,
				m_background_color.rgb,
				m_color_space,
				m_nerf.training.random_bg_color,
				m_nerf.training.linear_colors,
				1,
				m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
				mlp_out,
				counters.numsteps_counter_compacted.data(),
				ray_indices,
				rays_unnormalized,
				numsteps,
				PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
				PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
				dloss_dmlp_out,
				m_nerf.training.loss_type,
				m_nerf.training.depth_loss_type,
				counters.loss.data(),
				m_max_level_rand_training,
				max_level_compacted,
				m_nerf.rgb_activation,
				m_nerf.density_activation,
				m_nerf.training.snap_to_pixel_centers,
				accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
				sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
				sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
				m_nerf.training.error_map.resolution,
				m_nerf.training.error_map.cdf_resolution,
				include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
				m_nerf.training.dataset.sharpness_resolution,
				m_nerf.training.sharpness_grid.data(),
				m_nerf.density_grid.data(),
				m_nerf.density_grid_mean.data(),
				m_nerf.max_cascade,
				m_nerf.training.cam_exposure_gpu.data(),
				m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
				m_nerf.training.depth_supervision_lambda_tracking,
				m_nerf.training.rgb_supervision_lambda_tracking,
				m_nerf.training.near_distance,
				indice_image_for_tracking_pose_gpu.data(),
				sample_away_from_border_margin,
				m_nerf.training.use_depth_var_in_tracking_loss,
				m_max_grid_level_factor,
		    	m_add_free_space_loss_tracking,
				m_nerf.training.free_space_supervision_lambda_tracking,
				m_nerf.training.free_space_supervision_distance,
				nullptr,
				m_use_custom_ray_marching,
				((!m_use_sdf_in_nerf) && m_nerf.training.use_depth_var_in_tracking_loss) ? rays_depth_variance.data() : nullptr,
		    	m_add_DSnerf_loss_tracking,
		    	m_nerf.training.DS_nerf_supervision_lambda_tracking,
		    	m_nerf.training.DS_nerf_supervision_depth_sigma,
				m_use_stylesdf_in_nerf,
				m_use_volsdf_in_nerf,
				m_use_coslam_sdf_in_nerf,
				m_nerf.training.truncation_distance,
				m_nerf.training.volsdf_beta,
		    	m_add_sdf_loss_tracking,
		    	m_nerf.training.sdf_supervision_lambda_tracking,
		    	m_add_sdf_free_space_loss_tracking,
		    	m_nerf.training.sdf_free_space_supervision_lambda_tracking
			);
	}
	
	if (m_debug) {
		m_numsteps_compacted_cpu.resize(counters.rays_per_batch * 2);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_numsteps_compacted_cpu.data(),std::get<2>(scratch), counters.rays_per_batch * 2 * sizeof(uint32_t),cudaMemcpyDeviceToHost)
 	   );
	}

	fill_rollover_and_rescale<network_precision_t><<<n_blocks_linear(target_batch_size*padded_output_width), n_threads_linear, 0, stream>>>(
		target_batch_size, padded_output_width, counters.numsteps_counter_compacted.data(), dloss_dmlp_out
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size * floats_per_coord), n_threads_linear, 0, stream>>>(
		target_batch_size, floats_per_coord, counters.numsteps_counter_compacted.data(), (float*)coords_compacted
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, counters.numsteps_counter_compacted.data(), max_level_compacted
	);

	if (m_debug) {
		m_coords_compacted_filled_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_compacted_filled_cpu.data(),std::get<7>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}

	bool train_camera = true;
	bool prepare_input_gradients = train_camera;
	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);

	{
		auto ctx = m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, prepare_input_gradients);
		m_network->backward(stream, *ctx, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, EGradientMode::Overwrite);
	}


	if (m_debug) {
		m_coords_gradient_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_gradient_cpu.data(),std::get<8>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}

	std::string dir_encoding = m_network_config["dir_encoding"].value("otype", "none");
	bool is_dir_encoding_empty = dir_encoding == "Empty";

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf_slam, 0, stream,
			counters.rays_per_batch,
			n_rays_total,
			m_rng,
			m_aabb,
			ray_counter,
			m_nerf.training.transforms_gpu.data(),
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.cam_pos_gradient_gpu.data(),
			m_nerf.training.cam_rot_gradient_gpu.data(),
			1,
			m_nerf.training.dataset.metadata_gpu.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
			m_distortion.resolution,
			m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			indice_image_for_tracking_pose_gpu.data(),
			sample_away_from_border_margin,
			is_dir_encoding_empty
		);
	}

	m_rng.advance();

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}

}















/*
##############################################################################################
##############################################################################################
			Tracking with Gaussian Pyramid
##############################################################################################
##############################################################################################
*/


__global__ void generate_training_samples_for_tracking_gp(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ ray_counter_depth,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	uint32_t max_mip,
	float cone_angle_constant,
	Buffer2DView<const vec2> distortion,
	const float* __restrict__ extra_dims_gpu,
	uint32_t n_extra_dims,
	const uint32_t indice_image_for_tracking_pose,
    const uint32_t* __restrict__ image_ids,
	int32_t* __restrict__ mapping_indices,
    const uint32_t* __restrict__ existing_ray_mapping_gpu,
    const float* __restrict__ xy_image_pixel_indices,
	float* __restrict__ sample_z_vals, //DEBUG
	const bool use_custom_ray_marching,
	const bool use_depth_guided_sampling, //only available in custom ray marching
	const float dt_for_regular_sampling,
	const float dt_for_depth_guided_sampling,
	const float truncation_distance_for_depth_guided_sampling
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	mapping_indices[i] = -1; //  default to "not existing"

    if (existing_ray_mapping_gpu[i]!=i) return;

	uint32_t img;
	if (image_ids) {
    	img = image_ids[i];
	} else {
    	img = indice_image_for_tracking_pose;
	}

	ivec2 resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());

    vec2 uv(xy_image_pixel_indices[2*i], xy_image_pixel_indices[2*i+1]);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(uv, resolution, 0);
	if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f) {
		return;
	}

	float motionblur_time = 0.0;
	const vec2 focal_length = metadata[img].focal_length;
	const vec2 principal_point = metadata[img].principal_point;
	const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
	const Lens lens = metadata[img].lens;

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);
	vec3 cam_fwd = xform[2];

	float target_z = metadata[img].depth ? read_depth(uv, resolution, metadata[img].depth) : -1.0;

	Ray ray_unnormalized;
	const Ray* rays_in_unnormalized = metadata[img].rays;
	if (rays_in_unnormalized) {
		// Rays have been explicitly supplied. Read them.
		ray_unnormalized = rays_in_unnormalized[pix_idx];
	} else {
		ray_unnormalized = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, lens, distortion);
		if (!ray_unnormalized.is_valid()) {
			ray_unnormalized = {xform[3], xform[2]};
		}
	}

	float target_depth = length(ray_unnormalized.d) * target_z;

	vec3 ray_d_normalized = normalize(ray_unnormalized.d);

	vec2 tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
	float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x = fmaxf(tminmax.x, 0.0f);

	float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
	vec3 idir = vec3(1.0f) / ray_d_normalized;

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;
	vec3 pos;
	vec3 ray_o = ray_unnormalized.o;
	float cam_fwd_factor = dot(cam_fwd, ray_d_normalized);

	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
		float dt;
		if (use_custom_ray_marching){
			float cur_z = dot(cam_fwd, pos - ray_o);
			dt = calc_dt_slaim(
				cur_z, target_z, dt_for_regular_sampling, 
				dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling,
				cam_fwd_factor, use_depth_guided_sampling
			);
		} else {
			if (use_depth_guided_sampling) {
				float cur_depth = distance(pos, ray_o);
				dt = calc_dt_with_depth_guided_sampling(
					t, cone_angle,
					cur_depth, target_depth,
					dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling
				);
			} else {
				dt = calc_dt(t, cone_angle);
			}
		}
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid) {
			if (density_grid_occupied_at(pos, density_grid, mip)) {
				++j;
				t += dt;
			} else {
				t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
			}
		} else {
			++j;
			t += dt;
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
	
	// get depth_ray_counters
	if (target_depth > 0.) {
		uint32_t ray_idx_depth = atomicAdd(ray_counter_depth, 1);
	}

	ray_indices_out[ray_idx] = i;
	mapping_indices[i] = ray_idx;
	rays_out_unnormalized[ray_idx] = ray_unnormalized;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	vec3 warped_dir = warp_direction(ray_d_normalized);
	t=startt;
	j=0;
	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
		float dt;
		if (use_custom_ray_marching){
			float cur_z = dot(cam_fwd, pos - ray_o);
			dt = calc_dt_slaim(
				cur_z, target_z, dt_for_regular_sampling, 
				dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling,
				cam_fwd_factor, use_depth_guided_sampling
			);
		} else {
			if (use_depth_guided_sampling) {
				float cur_depth = distance(pos, ray_o);
				dt = calc_dt_with_depth_guided_sampling(
					t, cone_angle,
					cur_depth, target_depth,
					dt_for_depth_guided_sampling, truncation_distance_for_depth_guided_sampling
				);
			} else {
				dt = calc_dt(t, cone_angle);
			}
		}
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid) {
			if (density_grid_occupied_at(pos, density_grid, mip)) {
				coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
				if (sample_z_vals) {
					float cur_z = dot(cam_fwd, pos - ray_o);
					sample_z_vals[base+j] = cur_z;
				}
				++j;
				t += dt;
			} else {
				t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
			}
		} else {
			coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
			if (sample_z_vals) {
				float cur_z = dot(cam_fwd, pos - ray_o);
				sample_z_vals[base+j] = cur_z;
			}
			++j;
			t += dt;
		}
	}
}



__global__ void compute_GT_and_reconstructed_rgbd_gp(
	const uint32_t n_rays,
	BoundingBox aabb,
	default_rng_t rng,
	const uint32_t target_batch_size,
	const uint32_t* __restrict__ rays_counter,
	int padded_output_width,
	Buffer2DView<const vec4> envmap,
	vec3 background_color,
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
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	const vec3* __restrict__ exposure,
	float depth_supervision_lambda,
	const uint32_t indice_image_for_tracking_pose,
    const uint32_t* __restrict__ image_ids,
    const float* __restrict__ xy_image_pixel_indices,
	float* __restrict__ ground_truth_rgbd,
	float* __restrict__ reconstructed_rgbd,
	float* __restrict__ reconstructed_depth_var,
	const bool add_DSnerf_loss,
	const float DS_nerf_supervision_depth_sigma,
	const bool add_free_space_loss,
	const float free_space_supervision_distance
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	uint32_t img;
	if (image_ids) {
		img = image_ids[ray_idx];
	} else {
		img = indice_image_for_tracking_pose;
	}

	ivec2 resolution = metadata[img].resolution;

	vec2 uv(xy_image_pixel_indices[2*ray_idx], xy_image_pixel_indices[2*ray_idx+1]);
	//rng.advance(1); // motionblur_time

	float target_depth = length(rays_in_unnormalized[i].d) * ((metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);

	float T = 1.f;

	float EPSILON = 1e-4f;

	vec3 rgb_ray = vec3(0.0f);
	vec3 hitpoint = vec3(0.0f);

	float depth_ray = 0.f;
	uint32_t compacted_numsteps = 0;
	vec3 ray_o = rays_in_unnormalized[i].o;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = distance(pos, ray_o);
		float density = network_to_density(float(local_network_output[3]), density_activation);
		
		if (T < EPSILON) {
			if (add_DSnerf_loss && (target_depth > 0.)) {
				if (cur_depth > (target_depth + 3*DS_nerf_supervision_depth_sigma)) { //ensures we sample enough points for the DS loss
					break;
				}
			} else if (add_free_space_loss && (target_depth > 0.)) {
				if (cur_depth > (target_depth - free_space_supervision_distance)) { //ensures we sample enough points for the FS loss
					break;
				}
			} else {
				break;
			}
		}

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

	// compute depth variance
	if (reconstructed_depth_var) {
		//rewind pointers
		network_output -= compacted_numsteps * padded_output_width;
		coords_in -= compacted_numsteps;
		T = 1.f;
		float depth_var = 0.f;
		for (uint32_t j=0; j < compacted_numsteps; ++j) {
			const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 	       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
			const float dt = unwarp_dt(coords_in.ptr->dt);
			float cur_depth = distance(pos, ray_o);
			float density = network_to_density(float(local_network_output[3]), density_activation);

 	       	float tmp = (cur_depth - depth_ray) * (cur_depth - depth_ray);

			const float alpha = 1.f - __expf(-density * dt);
			const float weight = alpha * T;

 	       	depth_var += weight * tmp;
			T *= (1.f - alpha);

			network_output += padded_output_width;
			coords_in += 1;
		}
    	reconstructed_depth_var[i] = depth_var;
	}

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	vec4 envmap_value;
	vec3 dir;
	if (envmap) {
		dir = normalize(rays_in_unnormalized[i].d);
		envmap_value = read_envmap(envmap, dir);
		background_color = envmap_value.rgb + background_color * (1.0f - envmap_value.a);
	}

	vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
	// vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
	// vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
	vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

	vec3 rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.rgb + (1.0f - texsamp.a) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.a > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

    ground_truth_rgbd[i*4+0] = rgbtarget.x;
    ground_truth_rgbd[i*4+1] = rgbtarget.y;
    ground_truth_rgbd[i*4+2] = rgbtarget.z;
    ground_truth_rgbd[i*4+3] = target_depth;

    reconstructed_rgbd[i*4+0] = rgb_ray.x;
    reconstructed_rgbd[i*4+1] = rgb_ray.y;
    reconstructed_rgbd[i*4+2] = rgb_ray.z;
    reconstructed_rgbd[i*4+3] = depth_ray;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(target_batch_size - min(target_batch_size, compacted_base), compacted_numsteps);
	numsteps_out[i*2+0] = compacted_numsteps;
	numsteps_out[i*2+1] = compacted_base;
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
    float* __restrict__ rec_depth_var_output,
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
	vec3 rec_rgb = vec3(0.0f);
	vec3 gt_rgb = vec3(0.0f);
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

            gt_rgb.x += kernel[cpt] * gt_rgbd_input[index_map*4+0];
            gt_rgb.y += kernel[cpt] * gt_rgbd_input[index_map*4+1];
            gt_rgb.z += kernel[cpt] * gt_rgbd_input[index_map*4+2];

            rec_rgb.x += kernel[cpt] * rec_rgbd_input[index_map*4+0];
            rec_rgb.y += kernel[cpt] * rec_rgbd_input[index_map*4+1];
            rec_rgb.z += kernel[cpt] * rec_rgbd_input[index_map*4+2];
            rec_depth   += kernel[cpt] * rec_rgbd_input[index_map*4+3];

            if (rec_depth_var_input) {
                rec_depth_var += kernel[cpt] * rec_depth_var_input[index_map];
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

    gt_rgbd_output[i*4+0] = gt_rgb.x;
    gt_rgbd_output[i*4+1] = gt_rgb.y;
    gt_rgbd_output[i*4+2] = gt_rgb.z;
    gt_rgbd_output[i*4+3] = gt_depth;

    rec_rgbd_output[i*4+0] = rec_rgb.x;
    rec_rgbd_output[i*4+1] = rec_rgb.y;
    rec_rgbd_output[i*4+2] = rec_rgb.z;
    rec_rgbd_output[i*4+3] = rec_depth;

    if (rec_depth_var_output) {
        rec_depth_var_output[i] = rec_depth_var;
    }

}


__global__ void compute_loss_gp(
	const uint32_t n_rays,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
    const bool use_depth_var_in_loss,
    const uint32_t* __restrict__ existing_ray_mapping_gpu,
    const int32_t* __restrict__ mapping_indices,
    const float* __restrict__ ground_truth_rgbd,
	const float* __restrict__ reconstructed_rgbd,
	const float* __restrict__ reconstructed_depth_var,
	float* __restrict__ losses_and_gradients,
	uint32_t* __restrict__ super_ray_counter,
	uint32_t* __restrict__ super_ray_counter_depth,
	float depth_supervision_lambda,
	float rgb_supervision_lambda
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
            return;
        } else {
            super_i = ray_idx;
        }
    }

    uint32_t num_super_rays_in_loss = atomicAdd(super_ray_counter, 1);	 // first entry in the array is a counter

	vec3 avg_rgb_ray(
        reconstructed_rgbd[super_i*4+0],
        reconstructed_rgbd[super_i*4+1],
        reconstructed_rgbd[super_i*4+2]
	);
	float avg_depth_ray = reconstructed_rgbd[super_i*4+3];

    vec3 avg_rgb_ray_target(
        ground_truth_rgbd[super_i*4+0],
        ground_truth_rgbd[super_i*4+1],
        ground_truth_rgbd[super_i*4+2]
    );
    float avg_depth_ray_target = ground_truth_rgbd[super_i*4+3];

    LossAndGradient lg = loss_and_gradient(avg_rgb_ray_target, avg_rgb_ray, loss_type);
    LossAndGradient lg_depth;

    if (avg_depth_ray_target>0) {
	    lg_depth = loss_and_gradient(vec3(avg_depth_ray_target), vec3(avg_depth_ray), depth_loss_type);

        if ((lg_depth.loss.x==0.f)) {
            lg_depth.gradient.x = 0.f;
        }
	    uint32_t num_super_rays_depth_in_loss = atomicAdd(super_ray_counter_depth, 1);	 // first entry in the array is a counter

    } else {
	    lg_depth = loss_and_gradient(vec3(0.f), vec3(0.f), depth_loss_type);
        lg_depth.gradient.x = 0.f;
    }

    if (use_depth_var_in_loss) {
	    float rec_depth_var = reconstructed_depth_var[super_i];

        if (rec_depth_var < 1e-6){
            rec_depth_var = 1e-6;
        }
		float rec_depth_std = sqrt(rec_depth_var);
		lg_depth = lg_depth / rec_depth_std;
    }


	losses_and_gradients[i*4+0] = lg.gradient.x;
    losses_and_gradients[i*4+1] = lg.gradient.y;
    losses_and_gradients[i*4+2] = lg.gradient.z;
    losses_and_gradients[i*4+3] = lg_depth.gradient.x;

	float depth_loss_value    = avg_depth_ray_target > 0.0f ? depth_supervision_lambda * lg_depth.loss.x : 0;
	float mean_loss = rgb_supervision_lambda * compAdd(lg.loss) / 3.0f;

	if (loss_output) {
		// Note that here we should devide depth_loss by the number of depth rays
		// We're making this approximation for efficiency.
		// But the gradients are properly normalized
		// /!\ This loss doesn't take into account the FS or DS loss (if ever used)
		// But the FS loss will be taken into account in gradients
		// see compute_gradient gp function
		loss_output[i] = (mean_loss + depth_loss_value) / (float)n_rays;
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

	vec3 rgb_grad = vec3(0.f);
    float depth_grad = 0;
    for (uint32_t j=0; j<output_n_rays; ++j) {

        uint32_t index = j * input_n_rays + i;

        rgb_grad.x += gradients[index] * prev_partials[j*4+0];
        rgb_grad.y += gradients[index] * prev_partials[j*4+1];
        rgb_grad.z += gradients[index] * prev_partials[j*4+2];

        depth_grad += gradients[index] * prev_partials[j*4+3];
    }
    new_partials[i*4+0] = rgb_grad.x;
    new_partials[i*4+1] = rgb_grad.y;
    new_partials[i*4+2] = rgb_grad.z;
    new_partials[i*4+3] = depth_grad;
}



__global__ void compute_gradient_gp(
	const uint32_t n_rays,
	BoundingBox aabb,
	float loss_scale,
	int padded_output_width,
	const tcnn::network_precision_t* network_output,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	uint32_t* __restrict__ numsteps_compacted,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	const float* __restrict__ mean_density_ptr,
	float depth_supervision_lambda,
	float rgb_supervision_lambda,
	float near_distance,
	const int32_t* __restrict__ mapping_indices,
	const float* __restrict__ ground_truth_rgbd,
	const float* __restrict__ reconstructed_rgbd,
    const uint32_t* __restrict__ existing_ray_mapping_gpu,
	float* __restrict__ losses_and_gradients,
	const uint32_t* __restrict__ ray_counter_depth,
	const uint32_t* __restrict__ super_ray_counter,
	const uint32_t* __restrict__ super_ray_counter_depth,
	float* __restrict__ sample_outputs,
	const bool add_DSnerf_loss,
	const float DS_nerf_supervision_lambda,
	const float DS_nerf_supervision_depth_sigma,
	const bool add_free_space_loss,
	const float free_space_supervision_lambda,
	const float free_space_supervision_distance
) {

	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) { return; }

    uint32_t prev_i = existing_ray_mapping_gpu[i];

    int32_t ray_idx = mapping_indices[prev_i];

    if (ray_idx < 0){
        return;
    }

    vec3 dloss_db(
        losses_and_gradients[i*4+0],
        losses_and_gradients[i*4+1],
        losses_and_gradients[i*4+2]
	);
	dloss_db *= rgb_supervision_lambda;

    vec3 rgb_ray(
        reconstructed_rgbd[4*ray_idx],
        reconstructed_rgbd[4*ray_idx+1],
        reconstructed_rgbd[4*ray_idx+2]
	);

    float depth_ray = reconstructed_rgbd[4*ray_idx+3];

    float depth_ray_target = ground_truth_rgbd[4*ray_idx+3];

    float depth_loss_gradient = depth_ray_target > 0.0f ? depth_supervision_lambda * losses_and_gradients[i*4+3] : 0;

    uint32_t total_n_rays = *super_ray_counter;
    uint32_t total_n_rays_depth = *super_ray_counter_depth;
    uint32_t total_n_rays_depth_for_DS = *ray_counter_depth;

    float loss_scale_rgb = loss_scale / (float)total_n_rays;
    float loss_scale_depth;
    if (total_n_rays_depth>0){
        loss_scale_depth = loss_scale / (float)total_n_rays_depth;
    } else {
        loss_scale_depth = 0.0;
    }
	float loss_scale_depth_DS = 0.0;
    if (total_n_rays_depth_for_DS>0){
        loss_scale_depth_DS = loss_scale / (float)total_n_rays_depth_for_DS;
	}

	float EPSILON = 1e-4f;

	//DEBUG.
	// loss_scale_rgb = loss_scale / (float)n_rays;
	// loss_scale_depth = loss_scale_rgb;

    // No regularization for pose optimization
    const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
    const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
    vec3 rgb_ray2 = { 0.f,0.f,0.f };
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

	vec3 ray_o = rays_in_unnormalized[ray_idx].o;

	//
	//
	//
	// Free Space loss: compute normalizer
	uint32_t n_steps_for_free_space_loss = 0;
	if (add_free_space_loss) {
		if (depth_ray_target > 0.0f){
			uint32_t tmp_j=0;
			for (; tmp_j < numsteps_compact; ++tmp_j) {
				const float dt = unwarp_dt(coords_in.ptr->dt);
 		       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
				float cur_depth = distance(pos, ray_o);
				
				if (cur_depth > (depth_ray_target - free_space_supervision_distance)) {
					break;
				}
				
				coords_in += 1;
			}
			if (tmp_j > 0) {
				n_steps_for_free_space_loss = tmp_j;
				coords_in -= tmp_j; // rewind the pointer
			}
		}
	}


    for (uint32_t j=0; j < numsteps_compact; ++j) {

        // Compact network inputs
	    NerfCoordinate* coord_out = coords_out(j);
	    const NerfCoordinate* coord_in = coords_in(j);
	    coord_out->copy(*coord_in, coords_out.stride_in_bytes);

	    const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
	    float depth = distance(pos, ray_o);

	    float dt = unwarp_dt(coord_in->dt);
	    const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
	    const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
	    const float density = network_to_density(float(local_network_output[3]), density_activation);
	    const float alpha = 1.f - __expf(-density * dt);
	    const float weight = alpha * T;
		const float h = alpha * T + EPSILON;
	    rgb_ray2 += weight * rgb;
	    depth_ray2 += weight * depth;
	    T *= (1.f - alpha);

	    // we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
	    const vec3 suffix = rgb_ray - rgb_ray2;
	    const vec3 dloss_by_drgb = weight * dloss_db;

	    tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;

	    // chain rule to go from dloss/drgb to dloss/dmlp_output
        local_dL_doutput[0] = (loss_scale_rgb/3.0) * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
	    local_dL_doutput[1] = (loss_scale_rgb/3.0) * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
	    local_dL_doutput[2] = (loss_scale_rgb/3.0) * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

	    float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
	    const float depth_suffix = depth_ray - depth_ray2;

        // if no target depth for that ray then no depth supervision
        float depth_supervision = 0.0f;
        if (depth_ray_target > 0.0f) {
	        depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);
        }

	    const vec3 tmp = dloss_db;
	    float dloss_by_dmlp = density_derivative * (
			dt * (loss_scale_rgb /3.0 * dot(tmp, T * rgb - suffix) + loss_scale_depth * depth_supervision)
	    );

		//adding free space supervision using depth
		if (add_free_space_loss) {
			if (depth_ray_target > 0.f){
				float target_depth = depth_ray_target;
				if (n_steps_for_free_space_loss>0){
					if (depth <= (target_depth - free_space_supervision_distance)) {
						dloss_by_dmlp += loss_scale_depth_DS * density_derivative * free_space_supervision_lambda * 2 * alpha * dt * __expf(-density*dt) / (float)n_steps_for_free_space_loss;
					}
				}
			}
		}

		if (add_DSnerf_loss) {
			if (depth_ray_target > 0.f){
					float target_depth = depth_ray_target;
					float tmp_ds_grad = -dt*dt*T/h*__expf(-((depth-target_depth)*(depth-target_depth))/(2*DS_nerf_supervision_depth_sigma*DS_nerf_supervision_depth_sigma));
					for (uint32_t k = j+1; k < numsteps_compact; ++k) {
						const NerfCoordinate* tmp_coord_in = coords_in(k);
						const vec3 tmp_pos = unwarp_position(tmp_coord_in->pos.p, aabb);
						float tmp_depth = distance(tmp_pos, ray_o);
						float tmp_dt = unwarp_dt(tmp_coord_in->dt);
						tmp_ds_grad += dt * tmp_dt * __expf(-((tmp_depth-target_depth)*(tmp_depth-target_depth))/(2*DS_nerf_supervision_depth_sigma*DS_nerf_supervision_depth_sigma));
					}
					dloss_by_dmlp += loss_scale_depth_DS * density_derivative * DS_nerf_supervision_lambda * tmp_ds_grad;
			}
		}

	    local_dL_doutput[3] =
	    	dloss_by_dmlp +
	    	(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
	    	(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
	    	;

	    *(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

	    dloss_doutput += padded_output_width;
	    network_output += padded_output_width;

		if (sample_outputs) { //For debug purposes
			sample_outputs[(base_compact+j)*4+0] = density;
			sample_outputs[(base_compact+j)*4+1] = alpha;
			sample_outputs[(base_compact+j)*4+2] = weight;
			sample_outputs[(base_compact+j)*4+3] = T;
		}

    }
}


__global__ void compute_cam_gradient_train_nerf_slam_gp(
	const uint32_t n_rays,
	const BoundingBox aabb,
	const uint32_t* __restrict__ rays_counter,
	const TrainingXForm* training_xforms,
	vec3* cam_pos_gradient,
	vec3* cam_rot_gradient,
	const TrainingImageMetadata* __restrict__ metadata,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<NerfCoordinate> coords,
	PitchedPtr<NerfCoordinate> coords_gradient,
	float* __restrict__ distortion_gradient,
	float* __restrict__ distortion_gradient_weight,
	const ivec2 distortion_resolution,
	const uint32_t indice_image_for_tracking_pose,
    const uint32_t* __restrict__ image_ids,
    const float* __restrict__ xy_image_pixel_indices,
	uint32_t* num_rays_per_image_counter,
	const bool is_dir_encoding_empty
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	if (numsteps == 0) {
		// The ray doesn't matter. So no gradient onto the camera
		return;
	}

	uint32_t base = numsteps_in[i*2+1];
	coords += base;
	coords_gradient += base;

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];

	uint32_t img;
	if (image_ids) {
    	img = image_ids[ray_idx];
	} else {
    	img = indice_image_for_tracking_pose;
	}

	if (num_rays_per_image_counter) {
		uint32_t one = 1;
		uint32_t a = atomicAdd(&num_rays_per_image_counter[img], one);
	}

	ivec2 resolution = metadata[img].resolution;

	const mat4x3& xform = training_xforms[img].start;

	Ray ray = rays_in_unnormalized[i];
	ray.d = normalize(ray.d);
	Ray ray_gradient = { vec3(0.0f), vec3(0.0f) };

	// Compute ray gradient
	for (uint32_t j = 0; j < numsteps; ++j) {
		const vec3 warped_pos = coords(j)->pos.p;
		vec3 pos_gradient = coords_gradient(j)->pos.p * warp_position_derivative(warped_pos, aabb);
		// in case gradient are nan at this point
        if ( std::isnan(pos_gradient.x) or std::isnan(pos_gradient.y) or std::isnan(pos_gradient.z)
          or std::isinf(pos_gradient.x) or std::isinf(pos_gradient.y) or std::isinf(pos_gradient.z) ) {
                pos_gradient.x = 0.f;
                pos_gradient.y = 0.f;
                pos_gradient.z = 0.f;
        }
		ray_gradient.o += pos_gradient;
		const vec3 pos = unwarp_position(warped_pos, aabb);

		// Scaled by t to account for the fact that further-away objects' position
		// changes more rapidly as the direction changes.
		float t = distance(pos, ray.o);
		vec3 dir_gradient = coords_gradient(j)->dir.d * warp_direction_derivative(coords(j)->dir.d);
		if (is_dir_encoding_empty) {
			dir_gradient = {0.,0.,0.};
		}

		// in case gradient are nan at this point
        if ( std::isnan(dir_gradient.x) or std::isnan(dir_gradient.y) or std::isnan(dir_gradient.z)
          or std::isinf(dir_gradient.x) or std::isinf(dir_gradient.y) or std::isinf(dir_gradient.z) ) {
                dir_gradient.x = 0.f;
                dir_gradient.y = 0.f;
                dir_gradient.z = 0.f;
        }
		ray_gradient.d += pos_gradient * t + dir_gradient;
	}

	float uv_pdf = 1.0f;
    vec2 uv(xy_image_pixel_indices[2*ray_idx], xy_image_pixel_indices[2*ray_idx+1]);

	if (distortion_gradient) {
		// Projection of the raydir gradient onto the plane normal to raydir,
		// because that's the only degree of motion that the raydir has.
		vec3 orthogonal_ray_gradient = ray_gradient.d - ray.d * dot(ray_gradient.d, ray.d);

		// Rotate ray gradient to obtain image plane gradient.
		// This has the effect of projecting the (already projected) ray gradient from the
		// tangent plane of the sphere onto the image plane (which is correct!).
		vec3 image_plane_gradient = inverse(mat3(xform)) * orthogonal_ray_gradient;

		// Splat the resulting 2D image plane gradient into the distortion params
		deposit_image_gradient(image_plane_gradient.xy() / uv_pdf, distortion_gradient, distortion_gradient_weight, distortion_resolution, uv);
	}

	if (cam_pos_gradient) {
		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_pos_gradient[img][j], ray_gradient.o[j] / uv_pdf);
		}
	}

	if (cam_rot_gradient) {
		// Rotation is averaged in log-space (i.e. by averaging angle-axes).
		// Due to our construction of ray_gradient.d, ray_gradient.d and ray.d are
		// orthogonal, leading to the angle_axis magnitude to equal the magnitude
		// of ray_gradient.d.
		vec3 angle_axis = cross(ray.d, ray_gradient.d);

		// Atomically reduce the ray gradient into the xform gradient
		NGP_PRAGMA_UNROLL
		for (uint32_t j = 0; j < 3; ++j) {
			atomicAdd(&cam_rot_gradient[img][j], angle_axis[j] / uv_pdf);
		}
	}
}






void Testbed::train_nerf_slam_tracking_step_with_gaussian_pyramid(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

    //NOTE: get settings/hyperparams for tracking
    const uint32_t kernel_window_size = 5;
    const uint32_t gaussian_pyramid_level = m_tracking_gaussian_pyramid_level;
    const uint32_t sample_away_from_border_margin_h = m_nerf.training.m_sample_away_from_border_margin_h_tracking;
    const uint32_t sample_away_from_border_margin_w = m_nerf.training.m_sample_away_from_border_margin_w_tracking;

    //NOTE: get downsized image dimensions
	ivec2 resolution = m_nerf.training.dataset.metadata[m_nerf.training.indice_image_for_tracking_pose].resolution;
	float _level = pow(2.0, (float) gaussian_pyramid_level);
    ivec2 resolution_at_level = ivec2(vec2(resolution) / _level );

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
	// DEBUG
	// DEBUG
	m_n_super_rays = n_super_rays;
	m_n_total_rays = n_total_rays;
	m_n_total_rays_for_gradient = n_total_rays_for_gradient;
	m_xy_image_pixel_indices_int = xy_image_pixel_indices_int_cpu;
	m_xy_image_super_pixel_at_level_indices_int = xy_image_super_pixel_at_level_indices_int_cpu;
	m_existing_ray_mapping = existing_ray_mapping;
	// DEBUG
	// DEBUG

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
		float, // xy_pixel_indices ---->>> 11
		int32_t, // mapping_indices
		uint32_t, // numsteps_compacted
		uint32_t, // num super rays counter
		uint32_t, // num super rays counter depth
		uint32_t, // mapping of existing rays
		float,  // super ray gradients
		uint32_t // ray_counter_depth
	>(
		stream, &alloc,
		n_total_rays_for_gradient, // counters.rays_per_batch,
		n_total_rays_for_gradient, // counters.rays_per_batch,
		n_total_rays_for_gradient * 2, // counters.rays_per_batch * 2,
		max_samples * floats_per_coord,
		max_samples,
		std::max(target_batch_size, max_samples) * padded_output_width,
		target_batch_size * padded_output_width,
		target_batch_size * floats_per_coord,
		target_batch_size * floats_per_coord,
		target_batch_size,
		1, // ray_counter
		n_total_rays * 2, // 11
		n_total_rays,
		n_total_rays_for_gradient * 2,
        1,
        1,
		n_total_rays,
		n_super_rays * 6,
		1
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
	uint32_t* numsteps_compacted = std::get<13>(scratch);
	uint32_t* super_ray_counter = std::get<14>(scratch);
	uint32_t* super_ray_counter_depth = std::get<15>(scratch);
    uint32_t* existing_ray_mapping_gpu = std::get<16>(scratch);
	float* super_ray_gradients = std::get<17>(scratch);
	uint32_t* ray_counter_depth = std::get<18>(scratch);

	//uint32_t max_inference = next_multiple(std::min(n_total_rays_for_gradient * 1024, max_samples), tcnn::batch_size_granularity);
	uint32_t max_inference=max_samples;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference;
	}

	GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
	GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_tracking_step == 0) {
		counters.n_rays_total = 0;
	}

	counters.n_rays_total += n_total_rays;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter_depth, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter_depth, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_gradients, 0, n_super_rays * 6 * sizeof(uint32_t), stream));

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

	//NOTE: create 5-tap kernel
    std::vector<float> kernel = make_5tap_kernel();
    tcnn::GPUMemory<float> kernel_gpu;
    kernel_gpu.enlarge(kernel_window_size * kernel_window_size);
    CUDA_CHECK_THROW(
       cudaMemcpy(kernel_gpu.data(),kernel.data(),kernel_window_size * kernel_window_size * sizeof(float),cudaMemcpyHostToDevice)
    );

	//NOTE: ship to GPU some data
    CUDA_CHECK_THROW(
       cudaMemcpy(std::get<11>(scratch),xy_image_pixel_indices_cpu.data(),n_total_rays * 2 * sizeof(float),cudaMemcpyHostToDevice)
    );
    CUDA_CHECK_THROW(
       cudaMemcpy(std::get<16>(scratch),existing_ray_mapping.data(),n_total_rays * sizeof(uint32_t),cudaMemcpyHostToDevice)
    );

	//DEBUG
	//DEBUG
	tcnn::GPUMemory<float> sample_z_vals;
	if (m_debug) {
		sample_z_vals.enlarge(max_samples);
	}
	//DEBUG
	//DEBUG

	//NOTE: get sample along each rays
	linear_kernel(generate_training_samples_for_tracking_gp, 0, stream,
		n_total_rays,
		m_aabb,
		max_inference,
		m_rng,
		ray_counter,
		ray_counter_depth,
		counters.numsteps_counter.data(),
		ray_indices,
		rays_unnormalized,
		numsteps,
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		m_nerf.training.dataset.metadata_gpu.data(),
		m_nerf.training.transforms_gpu.data(),
		((!m_use_sdf_in_nerf) && m_use_density_in_nerf_sampling) ? m_nerf.density_grid_bitfield.data() : nullptr, // --> DO NOT use density grid in SDF
		m_nerf.max_cascade,
		m_nerf.cone_angle_constant,
		m_distortion.view(),
		m_nerf.training.extra_dims_gpu.data(),
		m_nerf_network->n_extra_dims(),
       	m_nerf.training.indice_image_for_tracking_pose,
		nullptr,
		mapping_indices,
       	existing_ray_mapping_gpu,
       	xy_image_pixel_indices,
		m_debug ? sample_z_vals.data(): nullptr,
		m_use_custom_ray_marching,
		m_use_depth_guided_sampling,
		m_nerf.training.dt_for_regular_sampling,
		m_nerf.training.dt_for_depth_guided_sampling,
		m_nerf.training.truncation_distance_for_depth_guided_sampling
	);

	//NOTE: some checks on the # ray sampled
	uint32_t m_track_pose_nerf_num_rays_in_tracking_step = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&m_track_pose_nerf_num_rays_in_tracking_step,std::get<10>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
    if (m_track_pose_nerf_num_rays_in_tracking_step + 1000 <= n_total_rays_for_gradient) {
        tlog::warning()<<" num rays for gradient is different than the required num of rays: "<<m_track_pose_nerf_num_rays_in_tracking_step<< " != "<< n_total_rays_for_gradient << ". Consider increasing batch_size";
    }

	m_ray_counter = m_track_pose_nerf_num_rays_in_tracking_step;
	m_rays_per_batch = counters.rays_per_batch;

	// -- get sample coordinates for each ray
	if (m_debug) {
		m_coords_cpu.resize(max_samples * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_cpu.data(),std::get<3>(scratch), max_samples * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
		m_numsteps_cpu.resize(n_total_rays_for_gradient * 2);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_numsteps_cpu.data(),std::get<2>(scratch), n_total_rays_for_gradient * 2 * sizeof(uint32_t),cudaMemcpyDeviceToHost)
 	   );
		m_sample_z_vals_cpu.resize(max_samples);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_sample_z_vals_cpu.data(),sample_z_vals.data(), max_samples * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}
	//DEBUG
	//DEBUG



	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}

	m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

	tcnn::GPUMemory<float> reconstructed_rgbd_gpu;
    tcnn::GPUMemory<float> ground_truth_rgbd_gpu;
    tcnn::GPUMemory<float> reconstructed_depth_var_gpu;
    reconstructed_rgbd_gpu.enlarge(n_total_rays_for_gradient * 4);
    ground_truth_rgbd_gpu.enlarge(n_total_rays_for_gradient * 4);
    if (m_nerf.training.use_depth_var_in_tracking_loss) {
        reconstructed_depth_var_gpu.enlarge(n_total_rays_for_gradient);
	}



	linear_kernel(compute_GT_and_reconstructed_rgbd_gp, 0, stream,
		n_total_rays_for_gradient,
		m_aabb,
		m_rng,
		target_batch_size,
		ray_counter,
		padded_output_width,
		m_envmap.view(),
		m_background_color.rgb,
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
		m_nerf.rgb_activation,
		m_nerf.density_activation,
		m_nerf.training.cam_exposure_gpu.data(),
		m_nerf.training.depth_supervision_lambda,
      	m_nerf.training.indice_image_for_tracking_pose,
		nullptr,
      	xy_image_pixel_indices,
       	ground_truth_rgbd_gpu.data(),
       	reconstructed_rgbd_gpu.data(),
		m_nerf.training.use_depth_var_in_tracking_loss ? reconstructed_depth_var_gpu.data() : nullptr,
		m_add_DSnerf_loss_tracking,
		m_nerf.training.DS_nerf_supervision_depth_sigma,
		m_add_free_space_loss_tracking,
		m_nerf.training.free_space_supervision_distance
	);



    //NOTE: compute Gaussian pyramids (ie convs)
    std::vector<tcnn::GPUMemory<float> > ground_truth_rgbd_tensors;
    std::vector<tcnn::GPUMemory<float> > reconstructed_rgbd_tensors;
    std::vector<tcnn::GPUMemory<float> > reconstructed_depth_var_tensors;
    std::vector<tcnn::GPUMemory<float> > gradients_tensors;
    std::vector<uint32_t> dimensions;

    uint32_t cur_super_ray_window_size = super_ray_window_size;
    uint32_t cur_ray_stride = ray_stride;
    uint32_t cur_dim = n_total_rays;

    ground_truth_rgbd_tensors.push_back(ground_truth_rgbd_gpu);
    reconstructed_rgbd_tensors.push_back(reconstructed_rgbd_gpu);
    dimensions.push_back(cur_dim);
    if (m_nerf.training.use_depth_var_in_tracking_loss) {
        reconstructed_depth_var_tensors.push_back(reconstructed_depth_var_gpu);
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
        if (m_nerf.training.use_depth_var_in_tracking_loss) {
            new_rec_depth_var.enlarge(tmp_dim);
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
            m_nerf.training.use_depth_var_in_tracking_loss ? reconstructed_depth_var_tensors.back().data() : nullptr,
            m_nerf.training.use_depth_var_in_tracking_loss ? new_rec_depth_var.data() : nullptr,
            new_gt_rgbd.data(),
            new_rec_rgbd.data(),
            gradients.data()
	    );

        ground_truth_rgbd_tensors.push_back(new_gt_rgbd);
        reconstructed_rgbd_tensors.push_back(new_rec_rgbd);
        gradients_tensors.push_back(gradients);
        dimensions.push_back(tmp_dim);

        if (m_nerf.training.use_depth_var_in_tracking_loss) {
            reconstructed_depth_var_tensors.push_back(new_rec_depth_var);
        }

        cur_super_ray_window_size = tmp_super_ray_window_size;
        cur_ray_stride = tmp_ray_stride;
        cur_dim = tmp_dim;

    }

	//DEBUG
	//DEBUG
	// store rec and GT RGB-D values at level
	std::vector<float> gt_rgbd_at_level;
	std::vector<float> rec_rgbd_at_level;
	gt_rgbd_at_level.resize(ground_truth_rgbd_tensors.back().size());
	rec_rgbd_at_level.resize(reconstructed_rgbd_tensors.back().size());
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(gt_rgbd_at_level.data(), ground_truth_rgbd_tensors.back().data(), ground_truth_rgbd_tensors.back().size()*sizeof(float), cudaMemcpyDeviceToHost,stream)
    );
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(rec_rgbd_at_level.data(), reconstructed_rgbd_tensors.back().data(), reconstructed_rgbd_tensors.back().size()*sizeof(float), cudaMemcpyDeviceToHost,stream)
    );
	m_gt_rgbd_at_level = gt_rgbd_at_level;
	m_rec_rgbd_at_level = rec_rgbd_at_level;

	if (m_nerf.training.use_depth_var_in_tracking_loss) {
		std::vector<float> rec_depth_var_at_level;
		rec_depth_var_at_level.resize(reconstructed_depth_var_tensors.back().size());
		CUDA_CHECK_THROW(
 	      cudaMemcpyAsync(rec_depth_var_at_level.data(), reconstructed_depth_var_tensors.back().data(), reconstructed_depth_var_tensors.back().size()*sizeof(float), cudaMemcpyDeviceToHost,stream)
 	   );
		m_rec_depth_var_at_level = rec_depth_var_at_level;
	}
	//DEBUG
	//DEBUG



    tcnn::GPUMemory<float> dL_dB;
    dL_dB.enlarge(n_super_rays * 4);

    //NOTE: compute loss.
	linear_kernel(compute_loss_gp, 0, stream,
		n_super_rays,
		m_nerf.training.loss_type,
		m_nerf.training.depth_loss_type,
		counters.loss.data(),
        m_nerf.training.use_depth_var_in_tracking_loss,
        gaussian_pyramid_level==0 ? existing_ray_mapping_gpu : nullptr,
        gaussian_pyramid_level==0 ? mapping_indices : nullptr,
        ground_truth_rgbd_tensors.back().data(),
        reconstructed_rgbd_tensors.back().data(),
        m_nerf.training.use_depth_var_in_tracking_loss ? reconstructed_depth_var_tensors.back().data(): nullptr,
        dL_dB.data(),
		super_ray_counter,
		super_ray_counter_depth,
		m_nerf.training.depth_supervision_lambda_tracking,
		m_nerf.training.rgb_supervision_lambda_tracking
	);


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

	uint32_t super_ray_counter_cpu = 0;
	uint32_t super_ray_counter_depth_cpu = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&super_ray_counter_cpu,std::get<14>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&super_ray_counter_depth_cpu,std::get<15>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	m_super_ray_counter = super_ray_counter_cpu;


	//DEBUG
	//DEBUG
	tcnn::GPUMemory<float> sample_outputs;
	if (m_debug) {
		sample_outputs.enlarge(max_samples * 4);
	}
	//DEBUG
	//DEBUG

    //NOTE: compute loss and gradients.
	linear_kernel(compute_gradient_gp, 0, stream,
        n_total_rays,
		m_aabb,
		LOSS_SCALE,
		padded_output_width,
		mlp_out,
		rays_unnormalized,
		numsteps,
        numsteps_compacted,
		PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
		dloss_dmlp_out,
		m_nerf.rgb_activation,
		m_nerf.density_activation,
		m_nerf.density_grid_mean.data(),
		m_nerf.training.depth_supervision_lambda_tracking,
		m_nerf.training.rgb_supervision_lambda_tracking,
		m_nerf.training.near_distance,
		mapping_indices,
        ground_truth_rgbd_gpu.data(),
        reconstructed_rgbd_gpu.data(),
        existing_ray_mapping_gpu,
        partial_derivatives.back().data(),
		ray_counter_depth,
		super_ray_counter,
		super_ray_counter_depth,
		m_debug ? sample_outputs.data(): nullptr,
		m_add_DSnerf_loss_tracking,
		m_nerf.training.DS_nerf_supervision_lambda_tracking,
		m_nerf.training.DS_nerf_supervision_depth_sigma,
		m_add_free_space_loss_tracking,
		m_nerf.training.free_space_supervision_lambda_tracking,
		m_nerf.training.free_space_supervision_distance
	);


	if (m_debug) {
		m_numsteps_compacted_cpu.resize(n_total_rays_for_gradient * 2);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_numsteps_compacted_cpu.data(),std::get<13>(scratch), n_total_rays_for_gradient * 2 * sizeof(uint32_t),cudaMemcpyDeviceToHost)
 	   );
		m_sample_outputs_cpu.resize(max_samples*4);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_sample_outputs_cpu.data(),sample_outputs.data(), max_samples * 4 * sizeof(float), cudaMemcpyDeviceToHost)
 	   );
	}
	//DEBUG
	//DEBUG




	fill_rollover_and_rescale<network_precision_t><<<n_blocks_linear(target_batch_size*padded_output_width), n_threads_linear, 0, stream>>>(
		target_batch_size, padded_output_width, counters.numsteps_counter_compacted.data(), dloss_dmlp_out
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size * floats_per_coord), n_threads_linear, 0, stream>>>(
		target_batch_size, floats_per_coord, counters.numsteps_counter_compacted.data(), (float*)coords_compacted
	);
	fill_rollover<float><<<n_blocks_linear(target_batch_size), n_threads_linear, 0, stream>>>(
		target_batch_size, 1, counters.numsteps_counter_compacted.data(), max_level_compacted
	);

	if (m_debug) {
		m_coords_compacted_filled_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_compacted_filled_cpu.data(),std::get<7>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}

	bool train_camera = true;
	bool prepare_input_gradients = train_camera;
	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);

	{
		auto ctx = m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, prepare_input_gradients);
		m_network->backward(stream, *ctx, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, EGradientMode::Overwrite);
	}

	if (m_debug) {
		m_coords_gradient_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_gradient_cpu.data(),std::get<8>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}

	std::string dir_encoding = m_network_config["dir_encoding"].value("otype", "none");
	bool is_dir_encoding_empty = dir_encoding == "Empty";

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf_slam_gp, 0, stream,
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
       		m_nerf.training.indice_image_for_tracking_pose,
			nullptr,
       		xy_image_pixel_indices,
			nullptr,
			is_dir_encoding_empty
		);
	}

	m_rng.advance();

}
































/*
##############################################################################################
##############################################################################################
			Tracking with max_grid_level coarse-to-fine strategy
##############################################################################################
##############################################################################################
*/


__global__ void generate_training_samples_nerf_slam_mgl(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t max_samples,
	const uint32_t n_rays_total,
	default_rng_t rng,
	uint32_t* __restrict__ ray_counter,
	uint32_t* __restrict__ ray_counter_depth,
	uint32_t* __restrict__ numsteps_counter,
	uint32_t* __restrict__ ray_indices_out,
	Ray* __restrict__ rays_out_unnormalized,
	uint32_t* __restrict__ numsteps_out,
	PitchedPtr<NerfCoordinate> coords_out,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	const uint8_t* __restrict__ density_grid,
	uint32_t max_mip,
	float* __restrict__ max_level_ptr,
	bool snap_to_pixel_centers,
	bool train_envmap,
	float cone_angle_constant,
	Buffer2DView<const vec2> distortion,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 cdf_res,
	const float* __restrict__ extra_dims_gpu,
	uint32_t n_extra_dims,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const ivec2 sample_away_from_border_margin,
	const float max_grid_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_rays) return;

	uint32_t local_img = image_idx(i, n_rays, n_rays_total, n_training_images, cdf_img);
	uint32_t img = idx_images_for_mapping[local_img];
	ivec2 resolution = metadata[img].resolution;

	rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
	vec2 uv = nerf_random_image_pos_training_with_margin(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, cdf_res, img, sample_away_from_border_margin);

	// Negative values indicate masked-away regions
	size_t pix_idx = pixel_idx(uv, resolution, 0);
	if (read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type).x < 0.0f) {
		return;
	}

	float max_level = max_grid_level;

	float motionblur_time = 0.0;

	const vec2 focal_length = metadata[img].focal_length;
	const vec2 principal_point = metadata[img].principal_point;
	const float* extra_dims = extra_dims_gpu + img * n_extra_dims;
	const Lens lens = metadata[img].lens;

	const mat4x3 xform = get_xform_given_rolling_shutter(training_xforms[img], metadata[img].rolling_shutter, uv, motionblur_time);

	Ray ray_unnormalized;
	const Ray* rays_in_unnormalized = metadata[img].rays;
	if (rays_in_unnormalized) {
		// Rays have been explicitly supplied. Read them.
		ray_unnormalized = rays_in_unnormalized[pix_idx];
	} else {
		ray_unnormalized = uv_to_ray(0, uv, resolution, focal_length, xform, principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, lens, distortion);
		if (!ray_unnormalized.is_valid()) {
			ray_unnormalized = {xform[3], xform[2]};
		}
	}

	vec3 ray_d_normalized = normalize(ray_unnormalized.d);

	vec2 tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
	float cone_angle = calc_cone_angle(dot(ray_d_normalized, xform[2]), focal_length, cone_angle_constant);

	// The near distance prevents learning of camera-specific fudge right in front of the camera
	tminmax.x = fmaxf(tminmax.x, 0.0f);

	float startt = advance_n_steps(tminmax.x, cone_angle, random_val(rng));
	vec3 idir = vec3(1.0f) / ray_d_normalized;

	// first pass to compute an accurate number of steps
	uint32_t j = 0;
	float t = startt;
	vec3 pos;

	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			++j;
			t += dt;
		} else {
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
		}
	}
	if (j == 0 && !train_envmap) {
		return;
	}
	uint32_t numsteps = j;
	uint32_t base = atomicAdd(numsteps_counter, numsteps);	 // first entry in the array is a counter
	if (base + numsteps > max_samples) {
		return;
	}

	coords_out += base;

	uint32_t ray_idx = atomicAdd(ray_counter, 1);

	// get depth_ray_counters
	float target_depth = metadata[img].depth ? read_depth(uv, resolution, metadata[img].depth) : -1.0f;
	if (target_depth > 0.) {
		uint32_t ray_idx_depth = atomicAdd(ray_counter_depth, 1);
	}

	ray_indices_out[ray_idx] = i;
	rays_out_unnormalized[ray_idx] = ray_unnormalized;
	numsteps_out[ray_idx*2+0] = numsteps;
	numsteps_out[ray_idx*2+1] = base;

	vec3 warped_dir = warp_direction(ray_d_normalized);
	t=startt;
	j=0;
	while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
		float dt = calc_dt(t, cone_angle);
		uint32_t mip = mip_from_dt(dt, pos, max_mip);
		if (density_grid_occupied_at(pos, density_grid, mip)) {
			coords_out(j)->set_with_optional_extra_dims(warp_position(pos, aabb), warped_dir, warp_dt(dt), extra_dims, coords_out.stride_in_bytes);
			++j;
			t += dt;
		} else {
			t = advance_to_next_voxel(t, cone_angle, pos, ray_d_normalized, idir, mip);
		}
	}

	{
		max_level_ptr += base;
		for (j = 0; j < numsteps; ++j) {
			max_level_ptr[j] = max_level;
		}
	}
}



__global__ void compute_loss_kernel_train_nerf_slam_mgl(
	const uint32_t n_rays,
	BoundingBox aabb,
	const uint32_t n_rays_total,
	default_rng_t rng,
	const uint32_t max_samples_compacted,
	const uint32_t* __restrict__ rays_counter,
	const uint32_t* __restrict__ rays_counter_depth,
	float loss_scale,
	int padded_output_width,
	Buffer2DView<const vec4> envmap,
	float* __restrict__ envmap_gradient,
	const ivec2 envmap_resolution,
	ELossType envmap_loss_type,
	vec3 background_color,
	EColorSpace color_space,
	bool train_with_random_bg_color,
	bool train_in_linear_colors,
	const uint32_t n_training_images,
	const TrainingImageMetadata* __restrict__ metadata,
	const tcnn::network_precision_t* network_output,
	uint32_t* __restrict__ numsteps_counter,
	const uint32_t* __restrict__ ray_indices_in,
	const Ray* __restrict__ rays_in_unnormalized,
	uint32_t* __restrict__ numsteps_in,
	PitchedPtr<const NerfCoordinate> coords_in,
	PitchedPtr<NerfCoordinate> coords_out,
	tcnn::network_precision_t* dloss_doutput,
	ELossType loss_type,
	ELossType depth_loss_type,
	float* __restrict__ loss_output,
	float* __restrict__ max_level_compacted_ptr,
	ENerfActivation rgb_activation,
	ENerfActivation density_activation,
	bool snap_to_pixel_centers,
	float* __restrict__ error_map,
	const float* __restrict__ cdf_x_cond_y,
	const float* __restrict__ cdf_y,
	const float* __restrict__ cdf_img,
	const ivec2 error_map_res,
	const ivec2 error_map_cdf_res,
	const float* __restrict__ sharpness_data,
	ivec2 sharpness_resolution,
	float* __restrict__ sharpness_grid,
	float* __restrict__ density_grid,
	const float* __restrict__ mean_density_ptr,
	uint32_t max_mip,
	const vec3* __restrict__ exposure,
	vec3* __restrict__ exposure_gradient,
	float depth_supervision_lambda,
	float near_distance,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const ivec2 sample_away_from_border_margin,
	const bool use_depth_variance_in_loss,
	const float max_grid_level
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= *rays_counter) { return; }

	// grab the number of samples for this ray, and the first sample
	uint32_t numsteps = numsteps_in[i*2+0];
	uint32_t base = numsteps_in[i*2+1];

	coords_in += base;
	network_output += base * padded_output_width;

	float T = 1.f;

	float EPSILON = 1e-4f;

	vec3 rgb_ray = vec3(0.0f);
	vec3 hitpoint = vec3(0.0f);

	float depth_ray = 0.f;
	uint32_t compacted_numsteps = 0;
	vec3 ray_o = rays_in_unnormalized[i].o;
	for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
		if (T < EPSILON) {
			break;
		}

		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
		const float dt = unwarp_dt(coords_in.ptr->dt);
		float cur_depth = distance(pos, ray_o);
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

	// compute depth variance
	float depth_var = 0.f;
	if (use_depth_variance_in_loss) {
		//rewind pointers
		network_output -= compacted_numsteps * padded_output_width;
		coords_in -= compacted_numsteps;
		T = 1.f;
		for (uint32_t j=0; j < compacted_numsteps; ++j) {
			const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
 	       	const vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
			const float dt = unwarp_dt(coords_in.ptr->dt);
			float cur_depth = distance(pos, ray_o);
			float density = network_to_density(float(local_network_output[3]), density_activation);

 	       	float tmp = (cur_depth - depth_ray) * (cur_depth - depth_ray);

			const float alpha = 1.f - __expf(-density * dt);
			const float weight = alpha * T;

 	       	depth_var += weight * tmp;
			T *= (1.f - alpha);

			network_output += padded_output_width;
			coords_in += 1;
		}
	}

	// Must be same seed as above to obtain the same
	// background color.
	uint32_t ray_idx = ray_indices_in[i];
	rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

	float img_pdf = 1.0f;
	uint32_t local_img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images, cdf_img, &img_pdf);
	uint32_t img = idx_images_for_mapping[local_img];
	ivec2 resolution = metadata[img].resolution;

	float uv_pdf = 1.0f;
	vec2 uv = nerf_random_image_pos_training_with_margin(rng, resolution, snap_to_pixel_centers, cdf_x_cond_y, cdf_y, error_map_cdf_res, img, sample_away_from_border_margin, &uv_pdf);

	float max_level = max_grid_level;
	rng.advance(1); // motionblur_time

	if (train_with_random_bg_color) {
		background_color = random_val_3d(rng);
	}
	vec3 pre_envmap_background_color = background_color = srgb_to_linear(background_color);

	// Composit background behind envmap
	vec4 envmap_value;
	vec3 dir;
	if (envmap) {
		dir = normalize(rays_in_unnormalized[i].d);
		envmap_value = read_envmap(envmap, dir);
		background_color = envmap_value.rgb + background_color * (1.0f - envmap_value.a);
	}

	vec3 exposure_scale = exp(0.6931471805599453f * exposure[img]);
	// vec3 rgbtarget = composit_and_lerp(uv, resolution, img, training_images, background_color, exposure_scale);
	// vec3 rgbtarget = composit(uv, resolution, img, training_images, background_color, exposure_scale);
	vec4 texsamp = read_rgba(uv, resolution, metadata[img].pixels, metadata[img].image_data_type);

	vec3 rgbtarget;
	if (train_in_linear_colors || color_space == EColorSpace::Linear) {
		rgbtarget = exposure_scale * texsamp.rgb + (1.0f - texsamp.a) * background_color;

		if (!train_in_linear_colors) {
			rgbtarget = linear_to_srgb(rgbtarget);
			background_color = linear_to_srgb(background_color);
		}
	} else if (color_space == EColorSpace::SRGB) {
		background_color = linear_to_srgb(background_color);
		if (texsamp.a > 0) {
			rgbtarget = linear_to_srgb(exposure_scale * texsamp.rgb / texsamp.a) * texsamp.a + (1.0f - texsamp.a) * background_color;
		} else {
			rgbtarget = background_color;
		}
	}

	if (compacted_numsteps == numsteps) {
		// support arbitrary background colors
		rgb_ray += T * background_color;
	}

	// Step again, this time computing loss
	network_output -= padded_output_width * compacted_numsteps; // rewind the pointer
	coords_in -= compacted_numsteps;

	uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps); // first entry in the array is a counter
	compacted_numsteps = min(max_samples_compacted - min(max_samples_compacted, compacted_base), compacted_numsteps);
	numsteps_in[i*2+0] = compacted_numsteps;
	numsteps_in[i*2+1] = compacted_base;
	if (compacted_numsteps == 0) {
		return;
	}

	max_level_compacted_ptr += compacted_base;
	coords_out += compacted_base;

	dloss_doutput += compacted_base * padded_output_width;

	LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray, loss_type);
	lg.loss /= img_pdf * uv_pdf;

	float target_depth = length(rays_in_unnormalized[i].d) * ((depth_supervision_lambda > 0.0f && metadata[img].depth) ? read_depth(uv, resolution, metadata[img].depth) : -1.0f);
	LossAndGradient lg_depth = loss_and_gradient(vec3(target_depth), vec3(depth_ray), depth_loss_type);
	if (use_depth_variance_in_loss) {
        if (depth_var < 1e-6){
            depth_var = 1e-6;
        }
		float depth_std = sqrt(depth_var);
		lg_depth = lg_depth / depth_std;
	}
	float depth_loss_gradient = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.gradient.x : 0;
	float depth_loss_value    = target_depth > 0.0f ? depth_supervision_lambda * lg_depth.loss.x : 0;

	// Note: dividing the gradient by the PDF would cause unbiased loss estimates.
	// Essentially: variance reduction, but otherwise the same optimization.
	// We _dont_ want that. If importance sampling is enabled, we _do_ actually want
	// to change the weighting of the loss function. So don't divide.
	// lg.gradient /= img_pdf * uv_pdf;

	uint32_t total_rays = *rays_counter;
	uint32_t total_rays_depth = *rays_counter_depth;

	float mean_loss = compAdd(lg.loss) / 3.0f;
	if (loss_output) {
		loss_output[i] = mean_loss / (float)total_rays + depth_loss_value / (float)total_rays_depth;
	}

	if (error_map) {
		const vec2 pos = clamp(uv * vec2(error_map_res) - vec2(0.5f), vec2(0.0f), vec2(error_map_res) - vec2(1.0f + 1e-4f));
		const ivec2 pos_int = pos;
		const vec2 weight = pos - vec2(pos_int);

		ivec2 idx = clamp(pos_int, ivec2(0), resolution - ivec2(2));

		auto deposit_val = [&](int x, int y, float val) {
			atomicAdd(&error_map[local_img * compMul(error_map_res) + y * error_map_res.x + x], val);
		};

		if (sharpness_data && aabb.contains(hitpoint)) {
			ivec2 sharpness_pos = clamp(ivec2(uv * vec2(sharpness_resolution)), ivec2(0), sharpness_resolution - ivec2(1));
			float sharp = sharpness_data[img * compMul(sharpness_resolution) + sharpness_pos.y * sharpness_resolution.x + sharpness_pos.x] + 1e-6f;

			// The maximum value of positive floats interpreted in uint format is the same as the maximum value of the floats.
			float grid_sharp = __uint_as_float(atomicMax((uint32_t*)&cascaded_grid_at(hitpoint, sharpness_grid, mip_from_pos(hitpoint, max_mip)), __float_as_uint(sharp)));
			grid_sharp = fmaxf(sharp, grid_sharp); // atomicMax returns the old value, so compute the new one locally.

			mean_loss *= fmaxf(sharp / grid_sharp, 0.01f);
		}

		deposit_val(idx.x,   idx.y,   (1 - weight.x) * (1 - weight.y) * mean_loss);
		deposit_val(idx.x+1, idx.y,        weight.x  * (1 - weight.y) * mean_loss);
		deposit_val(idx.x,   idx.y+1, (1 - weight.x) *      weight.y  * mean_loss);
		deposit_val(idx.x+1, idx.y+1,      weight.x  *      weight.y  * mean_loss);
	}

	//loss_scale /= n_rays;
	float loss_scale_rgb = loss_scale / (float)total_rays;
	float loss_scale_depth = 0.0f;
	if (total_rays_depth > 0) {
		loss_scale_depth = loss_scale / (float)total_rays_depth;
	}

	const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
	const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

	// now do it again computing gradients
	vec3 rgb_ray2 = { 0.f,0.f,0.f };
	float depth_ray2 = 0.f;
	T = 1.f;
	for (uint32_t j = 0; j < compacted_numsteps; ++j) {
		{
			max_level_compacted_ptr[j] = max_level;
		}
		// Compact network inputs
		NerfCoordinate* coord_out = coords_out(j);
		const NerfCoordinate* coord_in = coords_in(j);
		coord_out->copy(*coord_in, coords_out.stride_in_bytes);

		const vec3 pos = unwarp_position(coord_in->pos.p, aabb);
		float depth = distance(pos, ray_o);

		float dt = unwarp_dt(coord_in->dt);
		const tcnn::vector_t<tcnn::network_precision_t, 4> local_network_output = *(tcnn::vector_t<tcnn::network_precision_t, 4>*)network_output;
		const vec3 rgb = network_to_rgb_vec(local_network_output, rgb_activation);
		const float density = network_to_density(float(local_network_output[3]), density_activation);
		const float alpha = 1.f - __expf(-density * dt);
		const float weight = alpha * T;
		rgb_ray2 += weight * rgb;
		depth_ray2 += weight * depth;
		T *= (1.f - alpha);

		// we know the suffix of this ray compared to where we are up to. note the suffix depends on this step's alpha as suffix = (1-alpha)*(somecolor), so dsuffix/dalpha = -somecolor = -suffix/(1-alpha)
		const vec3 suffix = rgb_ray - rgb_ray2;
		const vec3 dloss_by_drgb = weight * lg.gradient;

		tcnn::vector_t<tcnn::network_precision_t, 4> local_dL_doutput;

		// chain rule to go from dloss/drgb to dloss/dmlp_output
		local_dL_doutput[0] = (loss_scale_rgb/3.0) * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
		local_dL_doutput[1] = (loss_scale_rgb/3.0) * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
		local_dL_doutput[2] = (loss_scale_rgb/3.0) * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

		float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
		const float depth_suffix = depth_ray - depth_ray2;
		const float depth_supervision = depth_loss_gradient * (T * depth - depth_suffix);

		float dloss_by_dmlp = density_derivative * (
			dt * ( loss_scale_rgb / 3.0 * dot(lg.gradient, T * rgb - suffix) + loss_scale_depth * depth_supervision)
		);

		//static constexpr float mask_supervision_strength = 1.f; // we are already 'leaking' mask information into the nerf via the random bg colors; setting this to eg between 1 and  100 encourages density towards 0 in such regions.
		//dloss_by_dmlp += (texsamp.a<0.001f) ? mask_supervision_strength * weight : 0.f;

		local_dL_doutput[3] =
			//loss_scale * dloss_by_dmlp +
			dloss_by_dmlp +
			(float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) +
			(float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);
			;

		*(tcnn::vector_t<tcnn::network_precision_t, 4>*)dloss_doutput = local_dL_doutput;

		dloss_doutput += padded_output_width;
		network_output += padded_output_width;
	}

	if (exposure_gradient) {
		// Assume symmetric loss
		vec3 dloss_by_dgt = -lg.gradient / uv_pdf;

		if (!train_in_linear_colors) {
			dloss_by_dgt /= srgb_to_linear_derivative(rgbtarget);
		}

		// 2^exposure * log(2)
		vec3 dloss_by_dexposure = loss_scale * dloss_by_dgt * exposure_scale * 0.6931471805599453f;
		atomicAdd(&exposure_gradient[img].x, dloss_by_dexposure.x);
		atomicAdd(&exposure_gradient[img].y, dloss_by_dexposure.y);
		atomicAdd(&exposure_gradient[img].z, dloss_by_dexposure.z);
	}

	if (compacted_numsteps == numsteps && envmap_gradient) {
		vec3 loss_gradient = lg.gradient;
		if (envmap_loss_type != loss_type) {
			loss_gradient = loss_and_gradient(rgbtarget, rgb_ray, envmap_loss_type).gradient;
		}

		vec3 dloss_by_dbackground = T * loss_gradient;
		if (!train_in_linear_colors) {
			dloss_by_dbackground /= srgb_to_linear_derivative(background_color);
		}

		tcnn::vector_t<tcnn::network_precision_t, 4> dL_denvmap;
		dL_denvmap[0] = loss_scale * dloss_by_dbackground.x;
		dL_denvmap[1] = loss_scale * dloss_by_dbackground.y;
		dL_denvmap[2] = loss_scale * dloss_by_dbackground.z;


		float dloss_by_denvmap_alpha = -dot(dloss_by_dbackground, pre_envmap_background_color);

		// dL_denvmap[3] = loss_scale * dloss_by_denvmap_alpha;
		dL_denvmap[3] = (tcnn::network_precision_t)0;

		deposit_envmap_gradient(dL_denvmap, envmap_gradient, envmap_resolution, dir);
	}
}





void Testbed::train_nerf_slam_tracking_step_mgl_coarse_to_fine(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct
	const ivec2 sample_away_from_border_margin(m_nerf.training.m_sample_away_from_border_margin_h_tracking, m_nerf.training.m_sample_away_from_border_margin_w_tracking);

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
		uint32_t // ray_counter
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
		1
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
	uint32_t* ray_counter_depth = std::get<11>(scratch);

	uint32_t max_inference;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference = max_samples;
	} else {
		max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	}

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_tracking_step == 0) {
		counters.n_rays_total = 0;
	}

	uint32_t n_rays_total = counters.n_rays_total;
	counters.n_rays_total += counters.rays_per_batch;

	float* envmap_gradient = nullptr;
	bool sample_focal_plane_proportional_to_error = false;
	bool sample_image_proportional_to_error = false;
	bool include_sharpness_in_error = false;
	bool accumulate_error = false;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter_depth, 0, sizeof(uint32_t), stream));

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

	std::vector<uint32_t> indice_image_for_tracking_pose = {m_nerf.training.indice_image_for_tracking_pose};
	tcnn::GPUMemory<uint32_t> indice_image_for_tracking_pose_gpu;
	indice_image_for_tracking_pose_gpu.enlarge(1);
	CUDA_CHECK_THROW(
 		cudaMemcpy(indice_image_for_tracking_pose_gpu.data(),indice_image_for_tracking_pose.data(),indice_image_for_tracking_pose.size() * sizeof(uint32_t),cudaMemcpyHostToDevice)
 	);


		linear_kernel(generate_training_samples_nerf_slam_mgl, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			max_inference,
			n_rays_total,
			m_rng,
			ray_counter,
			ray_counter_depth,
			counters.numsteps_counter.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			1,
			m_nerf.training.dataset.metadata_gpu.data(),
			m_nerf.training.transforms_gpu.data(),
			m_nerf.density_grid_bitfield.data(),
			m_nerf.max_cascade,
			max_level,
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.train_envmap,
			m_nerf.cone_angle_constant,
			m_distortion.view(),
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			m_nerf.training.extra_dims_gpu.data(),
			m_nerf_network->n_extra_dims(),
			indice_image_for_tracking_pose_gpu.data(),
			sample_away_from_border_margin,
			m_tracking_max_grid_level
		);


	//DEBUG
	//DEBUG
	uint32_t m_track_pose_nerf_num_rays_in_tracking_step = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&m_track_pose_nerf_num_rays_in_tracking_step,std::get<10>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	m_ray_counter = m_track_pose_nerf_num_rays_in_tracking_step;
	m_rays_per_batch = counters.rays_per_batch;
	//DEBUG
	//DEBUG



		if (hg_enc) {
			hg_enc->set_max_level_gpu(max_level);
		}

		GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
		GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
		m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

		if (hg_enc) {
			hg_enc->set_max_level_gpu(max_level_compacted);
		}

		linear_kernel(compute_loss_kernel_train_nerf_slam_mgl, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			n_rays_total,
			m_rng,
			target_batch_size,
			ray_counter,
			ray_counter_depth,
			LOSS_SCALE,
			padded_output_width,
			m_envmap.view(),
			envmap_gradient,
			m_envmap.resolution,
			m_envmap.loss_type,
			m_background_color.rgb,
			m_color_space,
			m_nerf.training.random_bg_color,
			m_nerf.training.linear_colors,
			1,
			m_nerf.training.dataset.metadata_gpu.data(),
			mlp_out,
			counters.numsteps_counter_compacted.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
			dloss_dmlp_out,
			m_nerf.training.loss_type,
			m_nerf.training.depth_loss_type,
			counters.loss.data(),
			max_level_compacted,
			m_nerf.rgb_activation,
			m_nerf.density_activation,
			m_nerf.training.snap_to_pixel_centers,
			accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.resolution,
			m_nerf.training.error_map.cdf_resolution,
			include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
			m_nerf.training.dataset.sharpness_resolution,
			m_nerf.training.sharpness_grid.data(),
			m_nerf.density_grid.data(),
			m_nerf.density_grid_mean.data(),
			m_nerf.max_cascade,
			m_nerf.training.cam_exposure_gpu.data(),
			m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
			m_nerf.training.depth_supervision_lambda,
			m_nerf.training.near_distance,
			indice_image_for_tracking_pose_gpu.data(),
			sample_away_from_border_margin,
			m_nerf.training.use_depth_var_in_tracking_loss,
			m_tracking_max_grid_level
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

	std::string dir_encoding = m_network_config["dir_encoding"].value("otype", "none");
	bool is_dir_encoding_empty = dir_encoding == "Empty";

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf_slam, 0, stream,
			counters.rays_per_batch,
			n_rays_total,
			m_rng,
			m_aabb,
			ray_counter,
			m_nerf.training.transforms_gpu.data(),
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.cam_pos_gradient_gpu.data(),
			m_nerf.training.cam_rot_gradient_gpu.data(),
			1,
			m_nerf.training.dataset.metadata_gpu.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
			m_distortion.resolution,
			m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			indice_image_for_tracking_pose_gpu.data(),
			sample_away_from_border_margin,
			is_dir_encoding_empty
		);
	}

	m_rng.advance();

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}

}



















/*
##############################################################################################
##############################################################################################

##############################################################################################
##############################################################################################


			Bundle Adjustment


##############################################################################################
##############################################################################################

##############################################################################################
##############################################################################################
*/


__global__ void mark_unvisible_density_grid_slam(const uint32_t n_elements,  uint8_t* __restrict__ grid_out,
	const uint32_t n_training_images,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	//That voxel has already been observed (in slam mode observed voxels cannot become unobserved)
	if (grid_out[i]==0)	return;

	uint32_t level = i / NERF_GRID_N_CELLS();
	uint32_t pos_idx = i % NERF_GRID_N_CELLS();

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	float voxel_size = scalbnf(1.0f / NERF_GRIDSIZE(), level);
	vec3 pos = (vec3{(float)x, (float)y, (float)z} / (float)NERF_GRIDSIZE() - vec3(0.5f)) * scalbnf(1.0f, level) + vec3(0.5f);

	vec3 corners[8] = {
		pos + vec3{0.0f,       0.0f,       0.0f      },
		pos + vec3{voxel_size, 0.0f,       0.0f      },
		pos + vec3{0.0f,       voxel_size, 0.0f      },
		pos + vec3{voxel_size, voxel_size, 0.0f      },
		pos + vec3{0.0f,       0.0f,       voxel_size},
		pos + vec3{voxel_size, 0.0f,       voxel_size},
		pos + vec3{0.0f,       voxel_size, voxel_size},
		pos + vec3{voxel_size, voxel_size, voxel_size},
	};

	// Number of training views that need to see a voxel cell
	// at minimum for that cell to be marked trainable.
	// Floaters can be reduced by increasing this value to 2,
	// but at the cost of certain reconstruction artifacts.
	const uint32_t min_count = 1;
	uint32_t count = 0;

	for (uint32_t j = 0; j < n_training_images && count < min_count; ++j) {
		const uint32_t img = idx_images_for_mapping[j];
		const auto& xform = training_xforms[img].start;
		const auto& m = metadata[img];

		if (m.lens.mode == ELensMode::FTheta || m.lens.mode == ELensMode::LatLong || m.lens.mode == ELensMode::Equirectangular) {
			// FTheta lenses don't have a forward mapping, so are assumed seeing everything. Latlong and equirect lenses
			// by definition see everything.
			++count;
			continue;
		}

		for (uint32_t k = 0; k < 8; ++k) {
			// Only consider voxel corners in front of the camera
			vec3 dir = normalize(corners[k] - xform[3]);
			if (dot(dir, xform[2]) < 1e-4f) {
				continue;
			}

			// Check if voxel corner projects onto the image plane, i.e. uv must be in (0, 1)^2
			vec2 uv = pos_to_uv(corners[k], m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), {}, m.lens);

			// `pos_to_uv` is _not_ injective in the presence of lens distortion (which breaks down outside of the image plane).
			// So we need to check whether the produced uv location generates a ray that matches the ray that we started with.
			Ray ray = uv_to_ray(0.0f, uv, m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, m.lens);
			if (distance(normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
				++count;
				break;
				
				// // -- - //Check if voxel corner is in front of closest surface - Anything behind depth shouldn't be carved out by the system 
				// // -- - // e.g. in slam it is possible that another room is behind doors and walls. We need to be able to sample points from the space behind to allow proper tracking and mapping
				// // -- - float target_z = m.depth ? read_depth(uv, m.resolution, m.depth) : -1.0f;

				// // -- - if (target_z > 0) {
				// // -- - 	float cur_z = dot( (corners[k] - xform[3]), xform[2]);
				// // -- - 	if (cur_z < (target_z+1e-4)){
				// // -- - 		++count;
				// // -- - 		break;
				// // -- - 	}
				// // -- - }
			}
		}
	}

	grid_out[i] = (count >= min_count) ? 0 : 1;
}






__global__ void mark_untrained_density_grid_slam(const uint32_t n_elements,  float* __restrict__ grid_out,
	const uint32_t n_training_images,
	const uint32_t* __restrict__ idx_images_for_mapping,
	const TrainingImageMetadata* __restrict__ metadata,
	const TrainingXForm* training_xforms,
	bool clear_visible_voxels
) {
	const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_elements) return;

	uint32_t level = i / NERF_GRID_N_CELLS();
	uint32_t pos_idx = i % NERF_GRID_N_CELLS();

	uint32_t x = tcnn::morton3D_invert(pos_idx>>0);
	uint32_t y = tcnn::morton3D_invert(pos_idx>>1);
	uint32_t z = tcnn::morton3D_invert(pos_idx>>2);

	float voxel_size = scalbnf(1.0f / NERF_GRIDSIZE(), level);
	vec3 pos = (vec3{(float)x, (float)y, (float)z} / (float)NERF_GRIDSIZE() - vec3(0.5f)) * scalbnf(1.0f, level) + vec3(0.5f);

	vec3 corners[8] = {
		pos + vec3{0.0f,       0.0f,       0.0f      },
		pos + vec3{voxel_size, 0.0f,       0.0f      },
		pos + vec3{0.0f,       voxel_size, 0.0f      },
		pos + vec3{voxel_size, voxel_size, 0.0f      },
		pos + vec3{0.0f,       0.0f,       voxel_size},
		pos + vec3{voxel_size, 0.0f,       voxel_size},
		pos + vec3{0.0f,       voxel_size, voxel_size},
		pos + vec3{voxel_size, voxel_size, voxel_size},
	};

	// Number of training views that need to see a voxel cell
	// at minimum for that cell to be marked trainable.
	// Floaters can be reduced by increasing this value to 2,
	// but at the cost of certain reconstruction artifacts.
	const uint32_t min_count = 1;
	uint32_t count = 0;

	for (uint32_t j = 0; j < n_training_images && count < min_count; ++j) {
		const uint32_t img = idx_images_for_mapping[j];
		const auto& xform = training_xforms[img].start;
		const auto& m = metadata[img];

		if (m.lens.mode == ELensMode::FTheta || m.lens.mode == ELensMode::LatLong || m.lens.mode == ELensMode::Equirectangular) {
			// FTheta lenses don't have a forward mapping, so are assumed seeing everything. Latlong and equirect lenses
			// by definition see everything.
			++count;
			continue;
		}

		for (uint32_t k = 0; k < 8; ++k) {
			// Only consider voxel corners in front of the camera
			vec3 dir = normalize(corners[k] - xform[3]);
			if (dot(dir, xform[2]) < 1e-4f) {
				continue;
			}

			// Check if voxel corner projects onto the image plane, i.e. uv must be in (0, 1)^2
			vec2 uv = pos_to_uv(corners[k], m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), {}, m.lens);

			// `pos_to_uv` is _not_ injective in the presence of lens distortion (which breaks down outside of the image plane).
			// So we need to check whether the produced uv location generates a ray that matches the ray that we started with.
			Ray ray = uv_to_ray(0.0f, uv, m.resolution, m.focal_length, xform, m.principal_point, vec3(0.0f), 0.0f, 1.0f, 0.0f, {}, {}, m.lens);
			if (distance(normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
				++count;
				break;
			}
		}
	}

	if (clear_visible_voxels || (grid_out[i] < 0) != (count < min_count)) {
		grid_out[i] = (count >= min_count) ? 0.f : -1.f;
	}
}


void Testbed::update_density_grid_nerf_ba(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);

	m_nerf.density_grid.resize(n_elements);

	const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		NerfPosition,       // positions at which the NN will be queried for density evaluation
		uint32_t,           // indices of corresponding density grid cells
		float,              // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
		network_precision_t // output of the MLP before being converted to densities.
	>(stream, &alloc, n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

	NerfPosition* density_grid_positions = std::get<0>(scratch);
	uint32_t* density_grid_indices = std::get<1>(scratch);
	float* density_grid_tmp = std::get<2>(scratch);
	network_precision_t* mlp_out = std::get<3>(scratch);

    std::sort(m_nerf.training.idx_images_for_mapping.begin(), m_nerf.training.idx_images_for_mapping.end(), std::greater<>());
    std::sort(m_nerf.training.idx_images_for_mapping_prev.begin(), m_nerf.training.idx_images_for_mapping_prev.end(), std::greater<>());


	//DEBUG
	//DEBUG
	if (m_training_step == 0) {
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
		m_nerf.density_grid_ema_step = 0;
	    
		m_nerf.density_grid_unvisible_regions_bool.enlarge(n_elements);
	 	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_unvisible_regions_bool.data(), 1, sizeof(uint8_t)*n_elements, stream));

	}
	//DEBUG
	//DEBUG

	/*
	if (m_ba_step == 0 || m_nerf.training.idx_images_for_mapping != m_nerf.training.idx_images_for_mapping_prev) {
		m_nerf.training.idx_images_for_mapping_prev = m_nerf.training.idx_images_for_mapping;
		if (m_ba_step == 0) {
			m_nerf.density_grid_ema_step = 0;
		}

		//NOTE: send mapping image indices to device
		m_nerf.training.idx_images_for_mapping_gpu.enlarge(m_nerf.training.idx_images_for_mapping.size());
		CUDA_CHECK_THROW(
 	      cudaMemcpy( m_nerf.training.idx_images_for_mapping_gpu.data(), m_nerf.training.idx_images_for_mapping.data(), m_nerf.training.idx_images_for_mapping.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)
 	   );

		// Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
		if (!m_nerf.training.dataset.has_rays) {
			linear_kernel(mark_untrained_density_grid_slam, 0, stream, n_elements, m_nerf.density_grid.data(),
				m_nerf.training.idx_images_for_mapping.size(),
				m_nerf.training.idx_images_for_mapping_gpu.data(),
				m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
				m_ba_step == 0
			);
		} else {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
		}
	}
	*/


	//DEBUG
	//DEBUG
	//DEBUG
	// detect unvisible regions
	//NOTE: send mapping image indices to device
	if (m_training_step == 0 || m_nerf.training.idx_images_for_mapping != m_nerf.training.idx_images_for_mapping_prev) {
		m_nerf.training.idx_images_for_mapping_prev = m_nerf.training.idx_images_for_mapping;
	    m_nerf.training.idx_images_for_mapping_gpu.enlarge(m_nerf.training.idx_images_for_mapping.size());
	    CUDA_CHECK_THROW(
           cudaMemcpy( m_nerf.training.idx_images_for_mapping_gpu.data(), m_nerf.training.idx_images_for_mapping.data(), m_nerf.training.idx_images_for_mapping.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)
        );
	    linear_kernel(mark_unvisible_density_grid_slam, 0, stream, n_elements, m_nerf.density_grid_unvisible_regions_bool.data(),
	    	m_nerf.training.idx_images_for_mapping.size(),
	    	m_nerf.training.idx_images_for_mapping_gpu.data(),
	    	m_nerf.training.dataset.metadata_gpu.data(),
	    	m_nerf.training.transforms_gpu.data()
	    );
    }
	//DEBUG
	//DEBUG
	//DEBUG



	uint32_t n_steps = 1;
	for (uint32_t i = 0; i < n_steps; ++i) {
		CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float)*n_elements, stream));

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_uniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions,
			density_grid_indices,
			m_nerf.max_cascade+1,
			-0.01f
		);
		m_nerf.training.density_grid_rng.advance();

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_nonuniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions+n_uniform_density_grid_samples,
			density_grid_indices+n_uniform_density_grid_samples,
			m_nerf.max_cascade+1,
			NERF_MIN_OPTICAL_THICKNESS()
		);
		m_nerf.training.density_grid_rng.advance();

		// Evaluate density at the spawned locations in batches.
		// Otherwise, we can exhaust the maximum index range of cutlass
		size_t batch_size = NERF_GRID_N_CELLS() * 2;

		for (size_t i = 0; i < n_density_grid_samples; i += batch_size) {
			batch_size = std::min(batch_size, n_density_grid_samples - i);

			GPUMatrix<network_precision_t, RM> density_matrix(mlp_out + i, padded_output_width, batch_size);
			GPUMatrix<float> density_grid_position_matrix((float*)(density_grid_positions + i), sizeof(NerfPosition)/sizeof(float), batch_size);
			m_nerf_network->density(stream, density_grid_position_matrix, density_matrix, false);
		}

		linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream, n_density_grid_samples, density_grid_indices, mlp_out, density_grid_tmp, m_nerf.rgb_activation, m_nerf.density_activation, m_use_volsdf_in_nerf, m_use_stylesdf_in_nerf, m_use_coslam_sdf_in_nerf, m_nerf.training.volsdf_beta);
		linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, decay, m_nerf.density_grid_ema_step, m_nerf.density_grid.data(), density_grid_tmp);

		++m_nerf.density_grid_ema_step;
	}

	//DEBUG
	//DEBUG
	//update_density_grid_mean_and_bitfield(stream);
	update_density_grid_mean_and_bitfield_slam(stream);
	//DEBUG
	//DEBUG
}

void Testbed::density_grid_culling_using_keyframes() {
	cudaStream_t stream = m_stream.get();

	const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);
	//NOTE: send mapping image indices to device
	m_nerf.training.idx_images_for_mapping_gpu.enlarge(m_nerf.training.idx_images_for_mapping.size());
	CUDA_CHECK_THROW(
 	    cudaMemcpy( m_nerf.training.idx_images_for_mapping_gpu.data(), m_nerf.training.idx_images_for_mapping.data(), m_nerf.training.idx_images_for_mapping.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)
 	);
	// Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
	if (!m_nerf.training.dataset.has_rays) {
		linear_kernel(mark_untrained_density_grid_slam, 0, stream, n_elements, m_nerf.density_grid.data(),
			m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.idx_images_for_mapping_gpu.data(),
			m_nerf.training.dataset.metadata_gpu.data(),
			m_nerf.training.transforms_gpu.data(),
			m_training_step == 0
		);
	}

	update_density_grid_mean_and_bitfield(stream);

	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}


void Testbed::update_density_grid_nerf_mapping(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream) {
	const uint32_t n_elements = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);

	m_nerf.density_grid.resize(n_elements);

	const uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;

	const uint32_t padded_output_width = m_nerf_network->padded_density_output_width();

	GPUMemoryArena::Allocation alloc;
	auto scratch = allocate_workspace_and_distribute<
		NerfPosition,       // positions at which the NN will be queried for density evaluation
		uint32_t,           // indices of corresponding density grid cells
		float,              // the resulting densities `density_grid_tmp` to be merged with the running estimate of the grid
		network_precision_t // output of the MLP before being converted to densities.
	>(stream, &alloc, n_density_grid_samples, n_elements, n_elements, n_density_grid_samples * padded_output_width);

	NerfPosition* density_grid_positions = std::get<0>(scratch);
	uint32_t* density_grid_indices = std::get<1>(scratch);
	float* density_grid_tmp = std::get<2>(scratch);
	network_precision_t* mlp_out = std::get<3>(scratch);

    std::sort(m_nerf.training.idx_images_for_mapping.begin(), m_nerf.training.idx_images_for_mapping.end(), std::greater<>());
    std::sort(m_nerf.training.idx_images_for_mapping_prev.begin(), m_nerf.training.idx_images_for_mapping_prev.end(), std::greater<>());

	//DEBUG
	//DEBUG
	 if (m_training_step == 0) {
	 	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
		m_nerf.density_grid_ema_step = 0;
	    
		m_nerf.density_grid_unvisible_regions_bool.enlarge(n_elements);
	 	CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid_unvisible_regions_bool.data(), 1, sizeof(uint8_t)*n_elements, stream));
	 }
	//DEBUG
	//DEBUG
	/*
	if (m_training_step == 0 || m_nerf.training.idx_images_for_mapping != m_nerf.training.idx_images_for_mapping_prev) {
		m_nerf.training.idx_images_for_mapping_prev = m_nerf.training.idx_images_for_mapping;
		if (m_training_step == 0) {
			m_nerf.density_grid_ema_step = 0;
		}

		//NOTE: send mapping image indices to device
		m_nerf.training.idx_images_for_mapping_gpu.enlarge(m_nerf.training.idx_images_for_mapping.size());
		CUDA_CHECK_THROW(
 	      cudaMemcpy( m_nerf.training.idx_images_for_mapping_gpu.data(), m_nerf.training.idx_images_for_mapping.data(), m_nerf.training.idx_images_for_mapping.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)
 	   );

		// Only cull away empty regions where no camera is looking when the cameras are actually meaningful.
		if (!m_nerf.training.dataset.has_rays) {
			linear_kernel(mark_untrained_density_grid_slam, 0, stream, n_elements, m_nerf.density_grid.data(),
				m_nerf.training.idx_images_for_mapping.size(),
				m_nerf.training.idx_images_for_mapping_gpu.data(),
				m_nerf.training.dataset.metadata_gpu.data(),
				m_nerf.training.transforms_gpu.data(),
				m_training_step == 0
			);
		} else {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.density_grid.data(), 0, sizeof(float)*n_elements, stream));
		}
	}
	*/

	//DEBUG
	//DEBUG
	//DEBUG
	// detect unvisible regions
	//NOTE: send mapping image indices to device
	if (m_training_step == 0 || m_nerf.training.idx_images_for_mapping != m_nerf.training.idx_images_for_mapping_prev) {
		m_nerf.training.idx_images_for_mapping_prev = m_nerf.training.idx_images_for_mapping;
	    m_nerf.training.idx_images_for_mapping_gpu.enlarge(m_nerf.training.idx_images_for_mapping.size());
	    CUDA_CHECK_THROW(
           cudaMemcpy( m_nerf.training.idx_images_for_mapping_gpu.data(), m_nerf.training.idx_images_for_mapping.data(), m_nerf.training.idx_images_for_mapping.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)
        );
	    linear_kernel(mark_unvisible_density_grid_slam, 0, stream, n_elements, m_nerf.density_grid_unvisible_regions_bool.data(),
	    	m_nerf.training.idx_images_for_mapping.size(),
	    	m_nerf.training.idx_images_for_mapping_gpu.data(),
	    	m_nerf.training.dataset.metadata_gpu.data(),
	    	m_nerf.training.transforms_gpu.data()
	    );
    }
	//DEBUG
	//DEBUG
	//DEBUG


	uint32_t n_steps = 1;
	for (uint32_t i = 0; i < n_steps; ++i) {
		CUDA_CHECK_THROW(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float)*n_elements, stream));

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_uniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions,
			density_grid_indices,
			m_nerf.max_cascade+1,
			-0.01f
		);
		m_nerf.training.density_grid_rng.advance();

		linear_kernel(generate_grid_samples_nerf_nonuniform, 0, stream,
			n_nonuniform_density_grid_samples,
			m_nerf.training.density_grid_rng,
			m_nerf.density_grid_ema_step,
			m_aabb,
			m_nerf.density_grid.data(),
			density_grid_positions+n_uniform_density_grid_samples,
			density_grid_indices+n_uniform_density_grid_samples,
			m_nerf.max_cascade+1,
			NERF_MIN_OPTICAL_THICKNESS()
		);
		m_nerf.training.density_grid_rng.advance();

		// Evaluate density at the spawned locations in batches.
		// Otherwise, we can exhaust the maximum index range of cutlass
		size_t batch_size = NERF_GRID_N_CELLS() * 2;

		for (size_t i = 0; i < n_density_grid_samples; i += batch_size) {
			batch_size = std::min(batch_size, n_density_grid_samples - i);

			GPUMatrix<network_precision_t, RM> density_matrix(mlp_out + i, padded_output_width, batch_size);
			GPUMatrix<float> density_grid_position_matrix((float*)(density_grid_positions + i), sizeof(NerfPosition)/sizeof(float), batch_size);
			m_nerf_network->density(stream, density_grid_position_matrix, density_matrix, false);
		}

		linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor, 0, stream, 
						n_density_grid_samples, density_grid_indices, mlp_out, density_grid_tmp, 
						m_nerf.rgb_activation, m_nerf.density_activation, m_use_volsdf_in_nerf, m_use_stylesdf_in_nerf, m_use_coslam_sdf_in_nerf, m_nerf.training.volsdf_beta);
		linear_kernel(ema_grid_samples_nerf, 0, stream, n_elements, decay, m_nerf.density_grid_ema_step, m_nerf.density_grid.data(), density_grid_tmp);

		++m_nerf.density_grid_ema_step;
	}

	//DEBUG
	//DEBUG
	//DEBUG
	//update_density_grid_mean_and_bitfield(stream);
	update_density_grid_mean_and_bitfield_slam(stream);
	//DEBUG
	//DEBUG
	//DEBUG
}



void Testbed::train_nerf_slam_bundle_adjustment(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream) {
	if (m_nerf.training.n_images_for_training == 0) {
		return;
	}

	if (m_nerf.training.include_sharpness_in_error) {
		size_t n_cells = NERF_GRID_N_CELLS() * NERF_CASCADES();
		if (m_nerf.training.sharpness_grid.size() < n_cells) {
			m_nerf.training.sharpness_grid.enlarge(NERF_GRID_N_CELLS() * NERF_CASCADES());
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		}

		if ((m_training_step == 0) & (m_ba_step == 0)) {
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.sharpness_grid.data(), 0, m_nerf.training.sharpness_grid.get_bytes(), stream));
		} else {
			linear_kernel(decay_sharpness_grid_nerf, 0, stream, m_nerf.training.sharpness_grid.size(), 0.95f, m_nerf.training.sharpness_grid.data());
		}
	}

	if (m_nerf.training.m_set_fix_num_rays_to_sample) {
    	m_nerf.training.counters_rgb_ba.rays_per_batch = m_nerf.training.m_target_num_rays_for_ba;
	}
	m_nerf.training.counters_rgb_ba.prepare_for_training_steps(stream);

	if (m_nerf.training.n_steps_since_cam_update == 0) {
		if (!m_nerf.training.m_use_ray_counter_per_image_in_ba) {
			m_num_rays_per_images.clear();
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(), 0, m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradient_weights(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
			CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(), 0, m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), stream));
		} else {
			if (m_nerf.training.m_reset_ray_counters_and_gradients_for_ba) {
				m_num_rays_per_images.clear();
				m_num_rays_per_images.resize(m_nerf.training.dataset.n_images, 0);
 	   			m_nerf.training.m_num_rays_per_image_gpu.enlarge(m_num_rays_per_images.size());
				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.m_num_rays_per_image_gpu.data(), 0, m_nerf.training.m_num_rays_per_image_gpu.get_bytes(), stream));

				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_pos_gradient_gpu.data(), 0, m_nerf.training.cam_pos_gradient_gpu.get_bytes(), stream));
				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_rot_gradient_gpu.data(), 0, m_nerf.training.cam_rot_gradient_gpu.get_bytes(), stream));
				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_exposure_gradient_gpu.data(), 0, m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), stream));
				CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradients(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
				CUDA_CHECK_THROW(cudaMemsetAsync(m_distortion.map->gradient_weights(), 0, sizeof(float)*m_distortion.map->n_params(), stream));
				CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.cam_focal_length_gradient_gpu.data(), 0, m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), stream));

				m_nerf.training.m_reset_ray_counters_and_gradients_for_ba = false;
			}
		}
	}

	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0 && m_nerf.training.optimize_extra_dims;
	uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
	if (train_extra_dims) {
		uint32_t n = n_extra_dims * m_nerf.training.n_images_for_training;
		m_nerf.training.extra_dims_gradient_gpu.enlarge(n);
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.extra_dims_gradient_gpu.data(), 0, m_nerf.training.extra_dims_gradient_gpu.get_bytes(), stream));
	}

	if (m_nerf.training.n_steps_since_error_map_update == 0 && !m_nerf.training.dataset.metadata.empty()) {
		uint32_t n_samples_per_image = (m_nerf.training.n_steps_between_error_map_updates * m_nerf.training.counters_rgb_ba.rays_per_batch) / m_nerf.training.idx_images_for_mapping.size();
		ivec2 res = m_nerf.training.dataset.metadata[0].resolution;
		m_nerf.training.error_map.resolution = min(ivec2((int)(std::sqrt(std::sqrt((float)n_samples_per_image)) * 3.5f)), res);
		m_nerf.training.error_map.data.resize(compMul(m_nerf.training.error_map.resolution) * m_nerf.training.idx_images_for_mapping.size());
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.data.data(), 0, m_nerf.training.error_map.data.get_bytes(), stream));
	}

	float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
	if (envmap_gradient) {
		CUDA_CHECK_THROW(cudaMemsetAsync(envmap_gradient, 0, sizeof(float)*m_envmap.envmap->n_params(), stream));
	}

	//NOTE: send mapping image indices to device
	m_nerf.training.idx_images_for_mapping_gpu.enlarge(m_nerf.training.idx_images_for_mapping.size());
	CUDA_CHECK_THROW(
       cudaMemcpy(
          m_nerf.training.idx_images_for_mapping_gpu.data(),
          m_nerf.training.idx_images_for_mapping.data(),
          m_nerf.training.idx_images_for_mapping.size() * sizeof(uint32_t),
          cudaMemcpyHostToDevice
       )
    );

	if (m_ba_mode==1) {
		train_nerf_slam_bundle_adjustment_step_with_gaussian_pyramid(target_batch_size, m_nerf.training.counters_rgb_ba, stream);
	} else if (m_ba_mode==2) {
		train_nerf_slam_bundle_adjustment_step_mgl_coarse_to_fine(target_batch_size, m_nerf.training.counters_rgb_ba, stream);
	} else {
		throw std::runtime_error{"Invalid bundle adjustment mode."};
	}

	m_trainer->optimizer_step(stream, LOSS_SCALE);

	++m_ba_step;

	if (envmap_gradient) {
		m_envmap.trainer->optimizer_step(stream, LOSS_SCALE);
	}

	float loss_scalar = m_nerf.training.counters_rgb_ba.update_after_training(target_batch_size, get_loss_scalar, stream);
	bool zero_records = m_nerf.training.counters_rgb_ba.measured_batch_size == 0;
	if (get_loss_scalar) {
		m_loss_scalar.update(loss_scalar);
	}

	if (zero_records) {
		m_loss_scalar.set(0.f);
		tlog::warning() << "Nerf training generated 0 samples. Aborting training.";
		m_train = false;
	}

	// Compute CDFs from the error map
	m_nerf.training.n_steps_since_error_map_update += 1;
	// This is low-overhead enough to warrant always being on.
	// It makes for useful visualizations of the training error.
	bool accumulate_error = true;
	if (accumulate_error && m_nerf.training.n_steps_since_error_map_update >= m_nerf.training.n_steps_between_error_map_updates) {
		m_nerf.training.error_map.cdf_resolution = m_nerf.training.error_map.resolution;
		m_nerf.training.error_map.cdf_x_cond_y.resize(compMul(m_nerf.training.error_map.cdf_resolution) * m_nerf.training.idx_images_for_mapping.size());
		m_nerf.training.error_map.cdf_y.resize(m_nerf.training.error_map.cdf_resolution.y * m_nerf.training.idx_images_for_mapping.size());
		m_nerf.training.error_map.cdf_img.resize(m_nerf.training.idx_images_for_mapping.size());

		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_x_cond_y.data(), 0, m_nerf.training.error_map.cdf_x_cond_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_y.data(), 0, m_nerf.training.error_map.cdf_y.get_bytes(), stream));
		CUDA_CHECK_THROW(cudaMemsetAsync(m_nerf.training.error_map.cdf_img.data(), 0, m_nerf.training.error_map.cdf_img.get_bytes(), stream));

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)m_nerf.training.error_map.cdf_resolution.y, threads.x), div_round_up((uint32_t)m_nerf.training.idx_images_for_mapping.size(), threads.y), 1 };
		construct_cdf_2d<<<blocks, threads, 0, stream>>>(
			m_nerf.training.idx_images_for_mapping.size(), m_nerf.training.error_map.cdf_resolution.y, m_nerf.training.error_map.cdf_resolution.x,
			m_nerf.training.error_map.data.data(),
			m_nerf.training.error_map.cdf_x_cond_y.data(),
			m_nerf.training.error_map.cdf_y.data()
		);
		linear_kernel(construct_cdf_1d, 0, stream,
			m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.error_map.cdf_resolution.y,
			m_nerf.training.error_map.cdf_y.data(),
			m_nerf.training.error_map.cdf_img.data()
		);

		// Compute image CDF on the CPU. It's single-threaded anyway. No use parallelizing.
		m_nerf.training.error_map.pmf_img_cpu.resize(m_nerf.training.error_map.cdf_img.size());
		m_nerf.training.error_map.cdf_img.copy_to_host(m_nerf.training.error_map.pmf_img_cpu);
		std::vector<float> cdf_img_cpu = m_nerf.training.error_map.pmf_img_cpu; // Copy unnormalized PDF into CDF buffer
		float cum = 0;
		for (float& f : cdf_img_cpu) {
			cum += f;
			f = cum;
		}
		float norm = 1.0f / cum;
		for (size_t i = 0; i < cdf_img_cpu.size(); ++i) {
			constexpr float MIN_PMF = 0.1f;
			m_nerf.training.error_map.pmf_img_cpu[i] = (1.0f - MIN_PMF) * m_nerf.training.error_map.pmf_img_cpu[i] * norm + MIN_PMF / (float)m_nerf.training.idx_images_for_mapping.size();
			cdf_img_cpu[i] = (1.0f - MIN_PMF) * cdf_img_cpu[i] * norm + MIN_PMF * (float)(i+1) / (float)m_nerf.training.idx_images_for_mapping.size();
		}
		m_nerf.training.error_map.cdf_img.copy_from_host(cdf_img_cpu);

		// Reset counters and decrease update rate.
		m_nerf.training.n_steps_since_error_map_update = 0;
		m_nerf.training.n_rays_since_error_map_update = 0;
		m_nerf.training.error_map.is_cdf_valid = true;

		//NOTE: in SLAM mode the mapping frames changes over time.
		m_nerf.training.n_steps_between_error_map_updates = (uint32_t)(m_nerf.training.n_steps_between_error_map_updates * 1.5f);
	}

	// Get extrinsics gradients
	m_nerf.training.n_steps_since_cam_update += 1;

	if (train_extra_dims) {
		std::vector<float> extra_dims_gradient(m_nerf.training.extra_dims_gradient_gpu.size());
		m_nerf.training.extra_dims_gradient_gpu.copy_to_host(extra_dims_gradient);

		// Optimization step
		for (uint32_t x = 0; x < m_nerf.training.idx_images_for_mapping.size(); ++x) {
			uint32_t i = m_nerf.training.idx_images_for_mapping[x];
			std::vector<float> gradient(n_extra_dims);
			for (uint32_t j = 0; j < n_extra_dims; ++j) {
				gradient[j] = extra_dims_gradient[i * n_extra_dims + j] / LOSS_SCALE;
			}

			//float l2_reg = 1e-4f;
			//gradient += m_nerf.training.extra_dims_opt[i].variable() * l2_reg;

			m_nerf.training.extra_dims_opt[i].set_learning_rate(m_optimizer->learning_rate());
			m_nerf.training.extra_dims_opt[i].step(gradient);
		}

		m_nerf.training.update_extra_dims();
	}

	bool train_camera = m_nerf.training.optimize_extrinsics || m_nerf.training.optimize_distortion || m_nerf.training.optimize_focal_length || m_nerf.training.optimize_exposure;
	if (train_camera && m_nerf.training.n_steps_since_cam_update >= m_nerf.training.n_steps_between_cam_updates) {
		//float per_camera_loss_scale = (float)m_nerf.training.idx_images_for_training_extrinsics.size() / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;
		float per_camera_loss_scale = (float)m_nerf.training.idx_images_for_mapping.size() / LOSS_SCALE / (float)m_nerf.training.n_steps_between_cam_updates;

		if (m_nerf.training.optimize_extrinsics) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			m_nerf.training.m_num_rays_per_image_gpu.copy_to_host(m_num_rays_per_images);

			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

			// Optimization step
			for (uint32_t k = 0; k < m_nerf.training.idx_images_for_training_extrinsics.size(); ++k) {

				uint32_t i = m_nerf.training.idx_images_for_training_extrinsics[k];

				if (!m_nerf.training.m_use_ray_counter_per_image_in_ba) {

					vec3 pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
					vec3 rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

					float l2_reg = m_nerf.training.extrinsic_l2_reg;
					pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
					rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

					if (m_nerf.training.m_use_pose_scheduler_in_mapping){
						m_nerf.training.cam_pos_offset[i].set_learning_rate(
                            std::max(
                                     m_nerf.training.extrinsic_learning_rate_ba_pos * std::pow(m_nerf.training.m_pose_scheduler_coef, (float)(m_nerf.training.cam_pos_offset[i].step() / m_nerf.training.m_pose_scheduler_norm)),
                                     m_optimizer->learning_rate()/1000.0f)
                        );
						m_nerf.training.cam_rot_offset[i].set_learning_rate(
                            std::max(
                                     m_nerf.training.extrinsic_learning_rate_ba_rot * std::pow(m_nerf.training.m_pose_scheduler_coef, (float)(m_nerf.training.cam_rot_offset[i].step() / m_nerf.training.m_pose_scheduler_norm)),
                                     m_optimizer->learning_rate()/1000.0f)
                        );
					} else {
						m_nerf.training.cam_pos_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_ba_pos);
						m_nerf.training.cam_rot_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate_ba_rot);
					}


					if ( std::isnan(pos_gradient.x) or std::isnan(pos_gradient.y) or std::isnan(pos_gradient.z) or
					     std::isnan(rot_gradient.x) or std::isnan(rot_gradient.y) or std::isnan(rot_gradient.z) ) {
	   	            	 pos_gradient.x = 0.f;
	   	            	 pos_gradient.y = 0.f;
	   	            	 pos_gradient.z = 0.f;
	   	            	 rot_gradient.x = 0.f;
	   	            	 rot_gradient.y = 0.f;
	   	            	 rot_gradient.z = 0.f;
	   	         	}

                    if (m_nerf.training.m_use_gradient_clipping_for_extrinsics) {
	   	         	    float max_step_pos = 100;
	   	         	    float max_step_rot = 100;

					    float norm_pos = glm::length(pos_gradient);
					    float norm_rot = glm::length(rot_gradient);
					    if (norm_pos > max_step_pos){
					    		pos_gradient = max_step_pos * pos_gradient / norm_pos;
					    }
					    if (norm_rot > max_step_rot){
					    		rot_gradient = max_step_rot * rot_gradient / norm_rot;
					    }
                    }

					m_nerf.training.cam_pos_offset[i].step(pos_gradient);
					m_nerf.training.cam_rot_offset[i].step(rot_gradient);

					m_nerf.training.update_transforms(i, i+1);
				} else {
					if (m_num_rays_per_images[i] > m_nerf.training.m_min_num_rays_per_image_for_pose_update) {

						vec3 pos_gradient = m_nerf.training.cam_pos_gradient[i] * per_camera_loss_scale;
						vec3 rot_gradient = m_nerf.training.cam_rot_gradient[i] * per_camera_loss_scale;

						float l2_reg = m_nerf.training.extrinsic_l2_reg;
						pos_gradient += m_nerf.training.cam_pos_offset[i].variable() * l2_reg;
						rot_gradient += m_nerf.training.cam_rot_offset[i].variable() * l2_reg;

						//m_nerf.training.cam_pos_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_pos_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));
						//m_nerf.training.cam_rot_offset[i].set_learning_rate(std::max(m_nerf.training.extrinsic_learning_rate * std::pow(0.33f, (float)(m_nerf.training.cam_rot_offset[i].step() / 128)), m_optimizer->learning_rate()/1000.0f));
						m_nerf.training.cam_pos_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate);
						m_nerf.training.cam_rot_offset[i].set_learning_rate(m_nerf.training.extrinsic_learning_rate);

						if ( std::isnan(pos_gradient.x) or std::isnan(pos_gradient.y) or std::isnan(pos_gradient.z) or
						     std::isnan(rot_gradient.x) or std::isnan(rot_gradient.y) or std::isnan(rot_gradient.z) ) {
		   	            	 pos_gradient.x = 0.f;
		   	            	 pos_gradient.y = 0.f;
		   	            	 pos_gradient.z = 0.f;
		   	            	 rot_gradient.x = 0.f;
		   	            	 rot_gradient.y = 0.f;
		   	            	 rot_gradient.z = 0.f;
		   	         	}

                        if (m_nerf.training.m_use_gradient_clipping_for_extrinsics) {
		   	         	    float max_step_pos = 100;
		   	         	    float max_step_rot = 100;

						    float norm_pos = glm::length(pos_gradient);
						    float norm_rot = glm::length(rot_gradient);
						    if (norm_pos > max_step_pos){
						    		pos_gradient = max_step_pos * pos_gradient / norm_pos;
						    }
						    if (norm_rot > max_step_rot){
						    		rot_gradient = max_step_rot * rot_gradient / norm_rot;
						    }
                        }

						m_nerf.training.cam_pos_offset[i].step(pos_gradient);
						m_nerf.training.cam_rot_offset[i].step(rot_gradient);

						m_nerf.training.update_transforms(i, i+1);

						//reset counters and grad
						m_num_rays_per_images[i] = 0;
						m_nerf.training.cam_pos_gradient[i] = vec3(0.0f);
						m_nerf.training.cam_rot_gradient[i] = vec3(0.0f);
					}
				}
			}

			if (m_nerf.training.m_use_ray_counter_per_image_in_ba) {
				m_nerf.training.m_num_rays_per_image_gpu.copy_from_host(m_num_rays_per_images);
				CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_pos_gradient_gpu.data(), m_nerf.training.cam_pos_gradient.data(), m_nerf.training.cam_pos_gradient_gpu.get_bytes(), cudaMemcpyHostToDevice, stream));
				CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_rot_gradient_gpu.data(), m_nerf.training.cam_rot_gradient.data(), m_nerf.training.cam_rot_gradient_gpu.get_bytes(), cudaMemcpyHostToDevice, stream));
			}

		}

		if (m_nerf.training.optimize_distortion) {
			linear_kernel(safe_divide, 0, stream,
				m_distortion.map->n_params(),
				m_distortion.map->gradients(),
				m_distortion.map->gradient_weights()
			);
			m_distortion.trainer->optimizer_step(stream, LOSS_SCALE*(float)m_nerf.training.n_steps_between_cam_updates);
		}

		if (m_nerf.training.optimize_focal_length) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(&m_nerf.training.cam_focal_length_gradient, m_nerf.training.cam_focal_length_gradient_gpu.data(), m_nerf.training.cam_focal_length_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));
			CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
			vec2 focal_length_gradient = m_nerf.training.cam_focal_length_gradient * per_camera_loss_scale;
			float l2_reg = m_nerf.training.intrinsic_l2_reg;
			focal_length_gradient += m_nerf.training.cam_focal_length_offset.variable() * l2_reg;
			m_nerf.training.cam_focal_length_offset.set_learning_rate(std::max(1e-3f * std::pow(0.33f, (float)(m_nerf.training.cam_focal_length_offset.step() / 128)),m_optimizer->learning_rate() / 1000.0f));
			m_nerf.training.cam_focal_length_offset.step(focal_length_gradient);
			m_nerf.training.dataset.update_metadata();
		}

		if (m_nerf.training.optimize_exposure) {
			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gradient.data(), m_nerf.training.cam_exposure_gradient_gpu.data(), m_nerf.training.cam_exposure_gradient_gpu.get_bytes(), cudaMemcpyDeviceToHost, stream));

			vec3 mean_exposure = vec3(0.0f);

			// Optimization step
			for (uint32_t k = 0; k < m_nerf.training.idx_images_for_training_extrinsics.size(); ++k) {
				uint32_t i = m_nerf.training.idx_images_for_training_extrinsics[k];

				vec3 gradient = m_nerf.training.cam_exposure_gradient[i] * per_camera_loss_scale;

				float l2_reg = m_nerf.training.exposure_l2_reg;
				gradient += m_nerf.training.cam_exposure[i].variable() * l2_reg;

				m_nerf.training.cam_exposure[i].set_learning_rate(m_optimizer->learning_rate());
				m_nerf.training.cam_exposure[i].step(gradient);

				mean_exposure += m_nerf.training.cam_exposure[i].variable();
			}

			mean_exposure /= m_nerf.training.idx_images_for_training_extrinsics.size();

			// Renormalize
			std::vector<vec3> cam_exposures(m_nerf.training.n_images_for_training);
			for (uint32_t i = 0; i < m_nerf.training.n_images_for_training; ++i) {
				cam_exposures[i] = m_nerf.training.cam_exposure[i].variable();
			}
			for (uint32_t k = 0; k < m_nerf.training.idx_images_for_training_extrinsics.size(); ++k) {
				uint32_t i = m_nerf.training.idx_images_for_training_extrinsics[k];
				cam_exposures[i] = m_nerf.training.cam_exposure[i].variable() -= mean_exposure;
			}

			CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.training.cam_exposure_gpu.data(), cam_exposures.data(), m_nerf.training.n_images_for_training * sizeof(vec3), cudaMemcpyHostToDevice, stream));
		}

		m_nerf.training.n_steps_since_cam_update = 0;
	}
}



/*
##############################################################################################
##############################################################################################

			BA - Gaussian Pyramir

##############################################################################################
##############################################################################################
*/




void Testbed::train_nerf_slam_bundle_adjustment_step_with_gaussian_pyramid(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct

    //NOTE: get settings/hyperparams for tracking
    const uint32_t kernel_window_size = 5;
    const uint32_t gaussian_pyramid_level = m_tracking_gaussian_pyramid_level;
    const uint32_t sample_away_from_border_margin_h = m_nerf.training.m_sample_away_from_border_margin_h_mapping;
    const uint32_t sample_away_from_border_margin_w = m_nerf.training.m_sample_away_from_border_margin_w_mapping;

    //NOTE: get downsized image dimensions
	ivec2 resolution = m_nerf.training.dataset.metadata[m_nerf.training.idx_images_for_mapping[0]].resolution;
	float _level = pow(2.0, (float) gaussian_pyramid_level);
    ivec2 resolution_at_level = ivec2(vec2(resolution) / _level );

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
    std::vector<uint32_t> image_ids;
    sample_pixels_for_ba_with_gaussian_pyramid(
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
        gaussian_pyramid_level,
		m_nerf.training.idx_images_for_mapping,
		image_ids
    );

	// DEBUG
	// DEBUG
	m_n_super_rays = n_super_rays;
	m_n_total_rays = n_total_rays;
	m_n_total_rays_for_gradient = n_total_rays_for_gradient;
	m_rays_per_batch = counters.rays_per_batch;
;	//m_xy_image_pixel_indices_int = xy_image_pixel_indices_int_cpu;
	//m_existing_ray_mapping = existing_ray_mapping;
	// DEBUG
	// DEBUG

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
		float, // xy_pixel_indices ---->>> 11
		int32_t, // mapping_indices
		uint32_t, // numsteps_compacted
		uint32_t, // num super rays counter
		uint32_t, // num super rays counter depth
		uint32_t, // mapping of existing rays
		float,  // super ray gradients
		uint32_t, // image ids of sampled rays
		uint32_t // ray_counter_depth
	>(
		stream, &alloc,
		n_total_rays_for_gradient, // counters.rays_per_batch,
		n_total_rays_for_gradient, // counters.rays_per_batch,
		n_total_rays_for_gradient * 2, // counters.rays_per_batch * 2,
		max_samples * floats_per_coord,
		max_samples,
		std::max(target_batch_size, max_samples) * padded_output_width,
		target_batch_size * padded_output_width,
		target_batch_size * floats_per_coord,
		target_batch_size * floats_per_coord,
		target_batch_size,
		1, // ray_counter
		n_total_rays * 2, // 11
		n_total_rays,
		n_total_rays_for_gradient * 2,
        1,
        1,
		n_total_rays,
		n_super_rays * 6,
		n_total_rays,
		1
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
	uint32_t* numsteps_compacted = std::get<13>(scratch);
	uint32_t* super_ray_counter = std::get<14>(scratch);
	uint32_t* super_ray_counter_depth = std::get<15>(scratch);
    uint32_t* existing_ray_mapping_gpu = std::get<16>(scratch);
	float* super_ray_gradients = std::get<17>(scratch);
    uint32_t* image_ids_gpu = std::get<18>(scratch);
	uint32_t* ray_counter_depth = std::get<19>(scratch);

	//uint32_t max_inference = next_multiple(std::min(n_total_rays_for_gradient * 1024, max_samples), tcnn::batch_size_granularity);
    //counters.measured_batch_size_before_compaction = max_inference;

	//uint32_t max_inference;
	//if (counters.measured_batch_size_before_compaction == 0) {
	//	counters.measured_batch_size_before_compaction = max_inference = max_samples;
	//} else {
	//	max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	//}
	
	uint32_t max_inference=max_samples;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference;
	}





	GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
	GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_training_step == 0) {
		counters.n_rays_total = 0;
	}

	//uint32_t n_rays_total = counters.n_rays_total;
	counters.n_rays_total += counters.rays_per_batch;
	m_nerf.training.n_rays_since_error_map_update += counters.rays_per_batch;

	//counters.n_rays_total += n_total_rays;

	//TODO: add the following options when doing BA
	// float* envmap_gradient = m_nerf.training.train_envmap ? m_envmap.envmap->gradients() : nullptr;
	// bool sample_focal_plane_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_focal_plane_proportional_to_error;
	// bool sample_image_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
	// bool include_sharpness_in_error = m_nerf.training.include_sharpness_in_error;
	// bool accumulate_error = true;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter_depth, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_counter_depth, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(super_ray_gradients, 0, n_super_rays * 6 * sizeof(uint32_t), stream));

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

	//NOTE: create 5-tap kernel
    std::vector<float> kernel = make_5tap_kernel();
    tcnn::GPUMemory<float> kernel_gpu;
    kernel_gpu.enlarge(kernel_window_size * kernel_window_size);
    CUDA_CHECK_THROW(
       cudaMemcpy(kernel_gpu.data(),kernel.data(),kernel_window_size * kernel_window_size * sizeof(float),cudaMemcpyHostToDevice)
    );

	//NOTE: ship to GPU some data
    CUDA_CHECK_THROW(
       cudaMemcpy(std::get<11>(scratch),xy_image_pixel_indices_cpu.data(),n_total_rays * 2 * sizeof(float),cudaMemcpyHostToDevice)
    );
    CUDA_CHECK_THROW(
       cudaMemcpy(std::get<16>(scratch),existing_ray_mapping.data(),n_total_rays * sizeof(uint32_t),cudaMemcpyHostToDevice)
    );
    CUDA_CHECK_THROW(
       cudaMemcpy(std::get<18>(scratch),image_ids.data(),n_total_rays * sizeof(uint32_t),cudaMemcpyHostToDevice)
    );

	//DEBUG
	//DEBUG
	tcnn::GPUMemory<float> sample_z_vals;
	if (m_debug) {
		sample_z_vals.enlarge(max_samples);
	}
	//DEBUG
	//DEBUG

    //NOTE: get sample along each rays
	linear_kernel(generate_training_samples_for_tracking_gp, 0, stream,
		n_total_rays,
		m_aabb,
		max_inference,
		m_rng,
		ray_counter,
		ray_counter_depth,
		counters.numsteps_counter.data(),
		ray_indices,
		rays_unnormalized,
		numsteps,
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		m_nerf.training.dataset.metadata_gpu.data(),
		m_nerf.training.transforms_gpu.data(),
		((!m_use_sdf_in_nerf) && m_use_density_in_nerf_sampling) ? m_nerf.density_grid_bitfield.data() : nullptr, // --> DO NOT use density grid in SDF
		m_nerf.max_cascade,
		m_nerf.cone_angle_constant,
		m_distortion.view(),
		m_nerf.training.extra_dims_gpu.data(),
		m_nerf_network->n_extra_dims(),
       	m_nerf.training.indice_image_for_tracking_pose,
		image_ids_gpu,
		mapping_indices,
       	existing_ray_mapping_gpu,
       	xy_image_pixel_indices,
		m_debug ? sample_z_vals.data(): nullptr,
		m_use_custom_ray_marching,
		m_use_depth_guided_sampling,
		m_nerf.training.dt_for_regular_sampling,
		m_nerf.training.dt_for_depth_guided_sampling,
		m_nerf.training.truncation_distance_for_depth_guided_sampling
	);


	//NOTE: some checks on the # ray sampled
	uint32_t m_track_pose_nerf_num_rays_in_tracking_step = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&m_track_pose_nerf_num_rays_in_tracking_step,std::get<10>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	//DEBUG
	//DEBUG
	m_ray_counter = m_track_pose_nerf_num_rays_in_tracking_step;
	m_rays_per_batch = counters.rays_per_batch;
	//DEBUG
	//DEBUG
	// -- get sample coordinates for each ray
	if (m_debug) {
		m_coords_cpu.resize(max_samples * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_cpu.data(),std::get<3>(scratch), max_samples * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
		m_numsteps_cpu.resize(n_total_rays_for_gradient * 2);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_numsteps_cpu.data(),std::get<2>(scratch), n_total_rays_for_gradient * 2 * sizeof(uint32_t),cudaMemcpyDeviceToHost)
 	   );
		m_sample_z_vals_cpu.resize(max_samples);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_sample_z_vals_cpu.data(),sample_z_vals.data(), max_samples * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}
	//DEBUG
	//DEBUG

	if (m_track_pose_nerf_num_rays_in_tracking_step != n_total_rays_for_gradient) {
        //tlog::warning()<<" num rays for gradient is different than the required num of rays: "<<m_track_pose_nerf_num_rays_in_tracking_step<< " != "<< n_total_rays_for_gradient << ". Consider increasing batch_size";
    }

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}

	m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

	tcnn::GPUMemory<float> reconstructed_rgbd_gpu;
    tcnn::GPUMemory<float> ground_truth_rgbd_gpu;
    tcnn::GPUMemory<float> reconstructed_depth_var_gpu;
    reconstructed_rgbd_gpu.enlarge(n_total_rays_for_gradient * 4);
    ground_truth_rgbd_gpu.enlarge(n_total_rays_for_gradient * 4);
    if (m_nerf.training.use_depth_var_in_tracking_loss) {
        reconstructed_depth_var_gpu.enlarge(n_total_rays_for_gradient);
	}

		linear_kernel(compute_GT_and_reconstructed_rgbd_gp, 0, stream,
			n_total_rays_for_gradient,
			m_aabb,
			m_rng,
			target_batch_size,
			ray_counter,
			padded_output_width,
			m_envmap.view(),
			m_background_color.rgb,
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
			m_nerf.rgb_activation,
			m_nerf.density_activation,
			m_nerf.training.cam_exposure_gpu.data(),
			m_nerf.training.depth_supervision_lambda,
       		m_nerf.training.indice_image_for_tracking_pose,
			image_ids_gpu,
       		xy_image_pixel_indices,
        	ground_truth_rgbd_gpu.data(),
        	reconstructed_rgbd_gpu.data(),
			m_nerf.training.use_depth_var_in_tracking_loss ? reconstructed_depth_var_gpu.data() : nullptr,
			m_add_DSnerf_loss,
		    m_nerf.training.DS_nerf_supervision_depth_sigma,
			m_add_free_space_loss,
			m_nerf.training.free_space_supervision_distance
		);


    //NOTE: compute Gaussian pyramids (ie convs)
    std::vector<tcnn::GPUMemory<float> > ground_truth_rgbd_tensors;
    std::vector<tcnn::GPUMemory<float> > reconstructed_rgbd_tensors;
    std::vector<tcnn::GPUMemory<float> > reconstructed_depth_var_tensors;
    std::vector<tcnn::GPUMemory<float> > gradients_tensors;
    std::vector<uint32_t> dimensions;

    uint32_t cur_super_ray_window_size = super_ray_window_size;
    uint32_t cur_ray_stride = ray_stride;
    uint32_t cur_dim = n_total_rays;

    ground_truth_rgbd_tensors.push_back(ground_truth_rgbd_gpu);
    reconstructed_rgbd_tensors.push_back(reconstructed_rgbd_gpu);
    dimensions.push_back(cur_dim);
    if (m_nerf.training.use_depth_var_in_tracking_loss) {
        reconstructed_depth_var_tensors.push_back(reconstructed_depth_var_gpu);
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
        if (m_nerf.training.use_depth_var_in_tracking_loss) {
            new_rec_depth_var.enlarge(tmp_dim);
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
            m_nerf.training.use_depth_var_in_tracking_loss ? reconstructed_depth_var_tensors.back().data() : nullptr,
            m_nerf.training.use_depth_var_in_tracking_loss ? new_rec_depth_var.data() : nullptr,
            new_gt_rgbd.data(),
            new_rec_rgbd.data(),
            gradients.data()
	    );

        ground_truth_rgbd_tensors.push_back(new_gt_rgbd);
        reconstructed_rgbd_tensors.push_back(new_rec_rgbd);
        gradients_tensors.push_back(gradients);
        dimensions.push_back(tmp_dim);

        if (m_nerf.training.use_depth_var_in_tracking_loss) {
            reconstructed_depth_var_tensors.push_back(new_rec_depth_var);
        }

        cur_super_ray_window_size = tmp_super_ray_window_size;
        cur_ray_stride = tmp_ray_stride;
        cur_dim = tmp_dim;

    }

    tcnn::GPUMemory<float> dL_dB;
    dL_dB.enlarge(n_super_rays * 4);

    //NOTE: compute loss.
	linear_kernel(compute_loss_gp, 0, stream,
		n_super_rays,
		m_nerf.training.loss_type,
		m_nerf.training.depth_loss_type,
		counters.loss.data(),
        m_nerf.training.use_depth_var_in_tracking_loss,
        gaussian_pyramid_level==0 ? existing_ray_mapping_gpu : nullptr,
        gaussian_pyramid_level==0 ? mapping_indices : nullptr,
        ground_truth_rgbd_tensors.back().data(),
        reconstructed_rgbd_tensors.back().data(),
        m_nerf.training.use_depth_var_in_tracking_loss ? reconstructed_depth_var_tensors.back().data(): nullptr,
        dL_dB.data(),
		super_ray_counter,
		super_ray_counter_depth,
		m_nerf.training.depth_supervision_lambda,
		m_nerf.training.rgb_supervision_lambda
	);


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

	uint32_t super_ray_counter_cpu = 0;
	uint32_t super_ray_counter_depth_cpu = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&super_ray_counter_cpu,std::get<14>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&super_ray_counter_depth_cpu,std::get<15>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );

    //NOTE: compute loss and gradients.
	linear_kernel(compute_gradient_gp, 0, stream,
        n_total_rays,
		m_aabb,
		LOSS_SCALE,
		padded_output_width,
		mlp_out,
		rays_unnormalized,
		numsteps,
        numsteps_compacted,
		PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
		PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
		dloss_dmlp_out,
		m_nerf.rgb_activation,
		m_nerf.density_activation,
		m_nerf.density_grid_mean.data(),
		m_nerf.training.depth_supervision_lambda,
		m_nerf.training.rgb_supervision_lambda,
		m_nerf.training.near_distance,
		mapping_indices,
        ground_truth_rgbd_gpu.data(),
        reconstructed_rgbd_gpu.data(),
        existing_ray_mapping_gpu,
        partial_derivatives.back().data(),
		ray_counter_depth,
		super_ray_counter,
		super_ray_counter_depth,
		nullptr,
		m_add_DSnerf_loss,
		m_nerf.training.DS_nerf_supervision_lambda,
		m_nerf.training.DS_nerf_supervision_depth_sigma,
		m_add_free_space_loss,
		m_nerf.training.free_space_supervision_lambda,
		m_nerf.training.free_space_supervision_distance
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

	if (m_debug) {
		m_coords_compacted_filled_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_compacted_filled_cpu.data(),std::get<7>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}

	bool train_camera = true;
	bool prepare_input_gradients = train_camera;
	GPUMatrix<float> coords_gradient_matrix((float*)coords_gradient, floats_per_coord, target_batch_size);

	{
		auto ctx = m_network->forward(stream, compacted_coords_matrix, &compacted_rgbsigma_matrix, false, prepare_input_gradients);
		m_network->backward(stream, *ctx, compacted_coords_matrix, compacted_rgbsigma_matrix, gradient_matrix, prepare_input_gradients ? &coords_gradient_matrix : nullptr, false, EGradientMode::Overwrite);
	}

	if (m_debug) {
		m_coords_gradient_cpu.resize(target_batch_size * floats_per_coord);
		CUDA_CHECK_THROW(
 	      cudaMemcpy(m_coords_gradient_cpu.data(),std::get<8>(scratch), target_batch_size * floats_per_coord * sizeof(float),cudaMemcpyDeviceToHost)
 	   );
	}

	std::string dir_encoding = m_network_config["dir_encoding"].value("otype", "none");
	bool is_dir_encoding_empty = dir_encoding == "Empty";

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf_slam_gp, 0, stream,
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
       		m_nerf.training.indice_image_for_tracking_pose,
			image_ids_gpu,
       		xy_image_pixel_indices,
			m_nerf.training.m_use_ray_counter_per_image_in_ba ? m_nerf.training.m_num_rays_per_image_gpu.data(): nullptr,
			is_dir_encoding_empty
		);
	}

	m_rng.advance();

}










/*
##############################################################################################
##############################################################################################

			BA - max_grid_level coarse-to-fine

##############################################################################################
##############################################################################################
*/

void Testbed::train_nerf_slam_bundle_adjustment_step_mgl_coarse_to_fine(uint32_t target_batch_size, Testbed::NerfCounters& counters, cudaStream_t stream) {
	const uint32_t padded_output_width = m_network->padded_output_width();
	const uint32_t max_samples = target_batch_size * 16; // Somewhat of a worst case
	const uint32_t floats_per_coord = sizeof(NerfCoordinate) / sizeof(float) + m_nerf_network->n_extra_dims();
	const uint32_t extra_stride = m_nerf_network->n_extra_dims() * sizeof(float); // extra stride on top of base NerfCoordinate struct
	const ivec2 sample_away_from_border_margin(m_nerf.training.m_sample_away_from_border_margin_h_mapping, m_nerf.training.m_sample_away_from_border_margin_w_mapping);

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
		uint32_t // ray_counter_depth
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
		1
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
	uint32_t* ray_counter_depth = std::get<11>(scratch);

	uint32_t max_inference;
	if (counters.measured_batch_size_before_compaction == 0) {
		counters.measured_batch_size_before_compaction = max_inference = max_samples;
	} else {
		max_inference = next_multiple(std::min(counters.measured_batch_size_before_compaction, max_samples), tcnn::batch_size_granularity);
	}

	GPUMatrix<float> compacted_coords_matrix((float*)coords_compacted, floats_per_coord, target_batch_size);
	GPUMatrix<network_precision_t> compacted_rgbsigma_matrix(mlp_out, padded_output_width, target_batch_size);

	GPUMatrix<network_precision_t> gradient_matrix(dloss_dmlp_out, padded_output_width, target_batch_size);

	if (m_ba_step == 0) {
		counters.n_rays_total = 0;
	}

	uint32_t n_rays_total = counters.n_rays_total;
	counters.n_rays_total += counters.rays_per_batch;
	m_nerf.training.n_rays_since_error_map_update += counters.rays_per_batch;

	float* envmap_gradient = nullptr;
	bool sample_focal_plane_proportional_to_error = false;
	bool sample_image_proportional_to_error = m_nerf.training.error_map.is_cdf_valid && m_nerf.training.sample_image_proportional_to_error;
	bool include_sharpness_in_error = false;
	bool accumulate_error = true;

	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter, 0, sizeof(uint32_t), stream));
	CUDA_CHECK_THROW(cudaMemsetAsync(ray_counter_depth, 0, sizeof(uint32_t), stream));

	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());

		linear_kernel(generate_training_samples_nerf_slam_mgl, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			max_inference,
			n_rays_total,
			m_rng,
			ray_counter,
			ray_counter_depth,
			counters.numsteps_counter.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
    		m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.dataset.metadata_gpu.data(),
			m_nerf.training.transforms_gpu.data(),
			m_nerf.density_grid_bitfield.data(),
			m_nerf.max_cascade,
			max_level,
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.train_envmap,
			m_nerf.cone_angle_constant,
			m_distortion.view(),
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
			m_nerf.training.extra_dims_gpu.data(),
			m_nerf_network->n_extra_dims(),
    		m_nerf.training.idx_images_for_mapping_gpu.data(),
			sample_away_from_border_margin,
			m_tracking_max_grid_level
		);

	//DEBUG
	//DEBUG
	uint32_t m_track_pose_nerf_num_rays_in_tracking_step = 0;
	CUDA_CHECK_THROW(
       cudaMemcpyAsync(&m_track_pose_nerf_num_rays_in_tracking_step,std::get<10>(scratch),sizeof(uint32_t),cudaMemcpyDeviceToHost,stream)
    );
	m_ray_counter = m_track_pose_nerf_num_rays_in_tracking_step;
	m_rays_per_batch = counters.rays_per_batch;
	//DEBUG
	//DEBUG

		if (hg_enc) {
			hg_enc->set_max_level_gpu(max_level);
		}

		GPUMatrix<float> coords_matrix((float*)coords, floats_per_coord, max_inference);
		GPUMatrix<network_precision_t> rgbsigma_matrix(mlp_out, padded_output_width, max_inference);
		m_network->inference_mixed_precision(stream, coords_matrix, rgbsigma_matrix, false);

		if (hg_enc) {
			hg_enc->set_max_level_gpu(max_level_compacted);
		}

		linear_kernel(compute_loss_kernel_train_nerf_slam_mgl, 0, stream,
			counters.rays_per_batch,
			m_aabb,
			n_rays_total,
			m_rng,
			target_batch_size,
			ray_counter,
			ray_counter_depth,
			LOSS_SCALE,
			padded_output_width,
			m_envmap.view(),
			envmap_gradient,
			m_envmap.resolution,
			m_envmap.loss_type,
			m_background_color.rgb,
			m_color_space,
			m_nerf.training.random_bg_color,
			m_nerf.training.linear_colors,
    		m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.dataset.metadata_gpu.data(),
			mlp_out,
			counters.numsteps_counter_compacted.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<const NerfCoordinate>((NerfCoordinate*)coords, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1 ,0, extra_stride),
			dloss_dmlp_out,
			m_nerf.training.loss_type,
			m_nerf.training.depth_loss_type,
			counters.loss.data(),
			max_level_compacted,
			m_nerf.rgb_activation,
			m_nerf.density_activation,
			m_nerf.training.snap_to_pixel_centers,
			accumulate_error ? m_nerf.training.error_map.data.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.resolution,
			m_nerf.training.error_map.cdf_resolution,
			include_sharpness_in_error ? m_nerf.training.dataset.sharpness_data.data() : nullptr,
			m_nerf.training.dataset.sharpness_resolution,
			m_nerf.training.sharpness_grid.data(),
			m_nerf.density_grid.data(),
			m_nerf.density_grid_mean.data(),
			m_nerf.max_cascade,
			m_nerf.training.cam_exposure_gpu.data(),
			m_nerf.training.optimize_exposure ? m_nerf.training.cam_exposure_gradient_gpu.data() : nullptr,
			m_nerf.training.depth_supervision_lambda,
			m_nerf.training.near_distance,
    		m_nerf.training.idx_images_for_mapping_gpu.data(),
			sample_away_from_border_margin,
			m_nerf.training.use_depth_var_in_tracking_loss,
			m_tracking_max_grid_level
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

	std::string dir_encoding = m_network_config["dir_encoding"].value("otype", "none");
	bool is_dir_encoding_empty = dir_encoding == "Empty";

	if (train_camera) {
		// Compute camera gradients
		linear_kernel(compute_cam_gradient_train_nerf_slam, 0, stream,
			counters.rays_per_batch,
			n_rays_total,
			m_rng,
			m_aabb,
			ray_counter,
			m_nerf.training.transforms_gpu.data(),
			m_nerf.training.snap_to_pixel_centers,
			m_nerf.training.cam_pos_gradient_gpu.data(),
			m_nerf.training.cam_rot_gradient_gpu.data(),
    		m_nerf.training.idx_images_for_mapping.size(),
			m_nerf.training.dataset.metadata_gpu.data(),
			ray_indices,
			rays_unnormalized,
			numsteps,
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_compacted, 1, 0, extra_stride),
			PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_gradient, 1, 0, extra_stride),
			m_nerf.training.optimize_distortion ? m_distortion.map->gradients() : nullptr,
			m_nerf.training.optimize_distortion ? m_distortion.map->gradient_weights() : nullptr,
			m_distortion.resolution,
			m_nerf.training.optimize_focal_length ? m_nerf.training.cam_focal_length_gradient_gpu.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_x_cond_y.data() : nullptr,
			sample_focal_plane_proportional_to_error ? m_nerf.training.error_map.cdf_y.data() : nullptr,
			sample_image_proportional_to_error ? m_nerf.training.error_map.cdf_img.data() : nullptr,
			m_nerf.training.error_map.cdf_resolution,
    		m_nerf.training.idx_images_for_mapping_gpu.data(),
			sample_away_from_border_margin,
			is_dir_encoding_empty
		);
	}

	m_rng.advance();

	if (hg_enc) {
		hg_enc->set_max_level_gpu(nullptr);
	}

}


void Testbed::send_image_to_gpu(const uint32_t image_id) {
	uint32_t i = image_id;
	const LoadedImageInfoCPU& m = m_nerf.training.dataset.images_info[i];
	m_nerf.training.dataset.set_training_image(i, m.res, m.pixels,
			m.depth_pixels, m.depth_scale * m_nerf.training.dataset.scale,
			m.image_data_on_gpu, m.image_type,
			EDepthDataType::UShort, m_nerf.sharpen,
			m.white_transparent, m.black_transparent,
			m.mask_color, m.rays);
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	if (m.image_data_on_gpu) {
		CUDA_CHECK_THROW(cudaFree(m.pixels));
	} else {
		free(m.pixels);
	}
	free(m.rays);
	free(m.depth_pixels);
}

void Testbed::remove_image_from_gpu(const uint32_t image_id) {
	uint32_t i = image_id;
	m_nerf.training.dataset.raymemory[i].free_memory();
	m_nerf.training.dataset.pixelmemory[i].free_memory();
	m_nerf.training.dataset.depthmemory[i].free_memory();
}


std::vector<uint8_t> Testbed::get_image_from_gpu(const uint32_t image_id) {
		std::vector<uint8_t> image;
		image.resize(m_nerf.training.dataset.pixelmemory[image_id].size());
		m_nerf.training.dataset.pixelmemory[image_id].copy_to_host(image);
		return image;
}

std::vector<float> Testbed::get_depth_from_gpu(const uint32_t image_id) {
		std::vector<float> image;
		image.resize(m_nerf.training.dataset.depthmemory[image_id].size());
		m_nerf.training.dataset.depthmemory[image_id].copy_to_host(image);
		return image;
}

NGP_NAMESPACE_END
