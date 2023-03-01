
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

#ifdef copysign
#undef copysign
#endif

using namespace Eigen;
using namespace tcnn;
namespace fs = filesystem;

NGP_NAMESPACE_BEGIN


inline constexpr __device__ float NERF_RENDERING_NEAR_DISTANCE() { return 0.05f; }
inline constexpr __device__ uint32_t NERF_STEPS() { return 1024; } // finest number of steps per unit length
inline constexpr __device__ uint32_t NERF_CASCADES() { return 8; }

inline constexpr __device__ float SQRT3() { return 1.73205080757f; }
inline constexpr __device__ float STEPSIZE() { return (SQRT3() / NERF_STEPS()); } // for nerf raymarch
inline constexpr __device__ float MIN_CONE_STEPSIZE() { return STEPSIZE(); }
// Maximum step size is the width of the coarsest gridsize cell.
inline constexpr __device__ float MAX_CONE_STEPSIZE() { return STEPSIZE() * (1<<(NERF_CASCADES()-1)) * NERF_STEPS() / NERF_GRIDSIZE(); }

// Used to index into the PRNG stream. Must be larger than the number of
// samples consumed by any given training ray.
inline constexpr __device__ uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() { return 8; }

// Any alpha below this is considered "invisible" and is thus culled away.
inline constexpr __device__ float NERF_MIN_OPTICAL_THICKNESS() { return 0.01f; }

static constexpr uint32_t MARCH_ITER = 10000;

static constexpr uint32_t MIN_STEPS_INBETWEEN_COMPACTION = 1;
static constexpr uint32_t MAX_STEPS_INBETWEEN_COMPACTION = 8;

static constexpr float UNIFORM_SAMPLING_FRACTION = 0.5f;


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

inline __device__ Vector2f sample_cdf_2d(Vector2f sample, uint32_t img, const Vector2i& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, float* __restrict__ pdf) {
	if (sample.x() < UNIFORM_SAMPLING_FRACTION) {
		sample.x() /= UNIFORM_SAMPLING_FRACTION;
		return sample;
	}

	sample.x() = (sample.x() - UNIFORM_SAMPLING_FRACTION) / (1.0f - UNIFORM_SAMPLING_FRACTION);

	cdf_y += img * res.y();

	// First select row according to cdf_y
	uint32_t y = binary_search(sample.y(), cdf_y, res.y());
	float prev = y > 0 ? cdf_y[y-1] : 0.0f;
	float pmf_y = cdf_y[y] - prev;
	sample.y() = (sample.y() - prev) / pmf_y;

	cdf_x_cond_y += img * res.y() * res.x() + y * res.x();

	// Then, select col according to x
	uint32_t x = binary_search(sample.x(), cdf_x_cond_y, res.x());
	prev = x > 0 ? cdf_x_cond_y[x-1] : 0.0f;
	float pmf_x = cdf_x_cond_y[x] - prev;
	sample.x() = (sample.x() - prev) / pmf_x;

	if (pdf) {
		*pdf = pmf_x * pmf_y * res.prod();
	}

	return {((float)x + sample.x()) / (float)res.x(), ((float)y + sample.y()) / (float)res.y()};
}



inline __device__ Vector2f nerf_random_image_pos_training(default_rng_t& rng, const Vector2i& resolution, bool snap_to_pixel_centers, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y, const Vector2i& cdf_res, uint32_t img, float* __restrict__ pdf = nullptr) {
	Vector2f xy = random_val_2d(rng);

	if (cdf_x_cond_y) {
		xy = sample_cdf_2d(xy, img, cdf_res, cdf_x_cond_y, cdf_y, pdf);
	} else if (pdf) {
		*pdf = 1.0f;
	}

	if (snap_to_pixel_centers) {
		xy = (xy.cwiseProduct(resolution.cast<float>()).cast<int>().cwiseMax(0).cwiseMin(resolution - Vector2i::Ones()).cast<float>() + Vector2f::Constant(0.5f)).cwiseQuotient(resolution.cast<float>());
	}
	return xy;
}


inline __host__ __device__ uint32_t grid_mip_offset(uint32_t mip) {
	return NERF_GRID_N_CELLS() * mip;
}

inline __host__ __device__ float calc_cone_angle(float cosine, const Eigen::Vector2f& focal_length, float cone_angle_constant) {
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


__device__ inline float network_to_rgb(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -10.0f, 10.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ inline float network_to_rgb_derivative(float val, ENerfActivation activation) {
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
__device__ inline vec3 network_to_rgb_derivative_vec(const T& val, ENerfActivation activation) {
	return {
		network_to_rgb_derivative(float(val[0]), activation),
		network_to_rgb_derivative(float(val[1]), activation),
		network_to_rgb_derivative(float(val[2]), activation),
	};
}



__device__ inline float network_to_density(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return val;
		case ENerfActivation::ReLU: return val > 0.0f ? val : 0.0f;
		case ENerfActivation::Logistic: return tcnn::logistic(val);
		case ENerfActivation::Exponential: return __expf(val);
		default: assert(false);
	}
	return 0.0f;
}

__device__ inline float network_to_density_derivative(float val, ENerfActivation activation) {
	switch (activation) {
		case ENerfActivation::None: return 1.0f;
		case ENerfActivation::ReLU: return val > 0.0f ? 1.0f : 0.0f;
		case ENerfActivation::Logistic: { float density = tcnn::logistic(val); return density * (1 - density); };
		case ENerfActivation::Exponential: return __expf(tcnn::clamp(val, -15.0f, 15.0f));
		default: assert(false);
	}
	return 0.0f;
}

__device__ inline Array3f network_to_rgb(const tcnn::vector_t<tcnn::network_precision_t, 4>& local_network_output, ENerfActivation activation) {
	return {
		network_to_rgb(float(local_network_output[0]), activation),
		network_to_rgb(float(local_network_output[1]), activation),
		network_to_rgb(float(local_network_output[2]), activation)
	};
}


template <typename T>
__device__ inline vec3 network_to_rgb_vec(const T& val, ENerfActivation activation) {
	return {
		network_to_rgb(float(val[0]), activation),
		network_to_rgb(float(val[1]), activation),
		network_to_rgb(float(val[2]), activation),
	};
}




__device__ inline vec3 warp_position(const vec3& pos, const BoundingBox& aabb) {
	// return {tcnn::logistic(pos.x - 0.5f), tcnn::logistic(pos.y - 0.5f), tcnn::logistic(pos.z - 0.5f)};
	// return pos;
	return aabb.relative_pos(pos);
}


__device__ inline vec3 unwarp_position(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x) + 0.5f, logit(pos.y) + 0.5f, logit(pos.z) + 0.5f};
	// return pos;

	return aabb.min + pos * aabb.diag();
}

__device__ inline vec3 unwarp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
	// return {logit(pos.x()) + 0.5f, logit(pos.y()) + 0.5f, logit(pos.z()) + 0.5f};
	// return pos;

	return aabb.diag();
}

__device__ inline vec3 warp_position_derivative(const vec3& pos, const BoundingBox& aabb) {
	return vec3(1.0f) / unwarp_position_derivative(pos, aabb);
}

__host__ __device__ inline vec3 warp_direction(const vec3& dir) {
	return (dir + vec3(1.0f)) * 0.5f;
}

__device__ inline vec3 unwarp_direction(const vec3& dir) {
	return dir * 2.0f - vec3(1.0f);
}


__device__ inline vec3 warp_direction_derivative(const vec3& dir) {
	return vec3(0.5f);
}

__device__ inline vec3 unwarp_direction_derivative(const vec3& dir) {
	return vec3(2.0f);
}

__device__ inline float warp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return (dt - MIN_CONE_STEPSIZE()) / (max_stepsize - MIN_CONE_STEPSIZE());
}

__device__ inline float unwarp_dt(float dt) {
	float max_stepsize = MIN_CONE_STEPSIZE() * (1<<(NERF_CASCADES()-1));
	return dt * (max_stepsize - MIN_CONE_STEPSIZE()) + MIN_CONE_STEPSIZE();
}


__device__ inline uint32_t cascaded_grid_idx_at(vec3 pos, uint32_t mip) {
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


__device__ inline bool density_grid_occupied_at(const vec3& pos, const uint8_t* density_grid_bitfield, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return false;
	}
	return density_grid_bitfield[idx/8+grid_mip_offset(mip)/8] & (1<<(idx%8));
}


__device__ inline float cascaded_grid_at(vec3 pos, const float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		return 0.0f;
	}
	return cascaded_grid[idx+grid_mip_offset(mip)];
}


__device__ inline float& cascaded_grid_at(vec3 pos, float* cascaded_grid, uint32_t mip) {
	uint32_t idx = cascaded_grid_idx_at(pos, mip);
	if (idx == 0xFFFFFFFF) {
		idx = 0;
		printf("WARNING: invalid cascaded grid access.");
	}
	return cascaded_grid[idx+grid_mip_offset(mip)];
}










inline __device__ int mip_from_pos(const Vector3f& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	int exponent;
	float maxval = (pos - Vector3f::Constant(0.5f)).cwiseAbs().maxCoeff();
	frexpf(maxval, &exponent);
	return min(max_cascade, max(0, exponent+1));
}

inline __device__ int mip_from_dt(float dt, const Vector3f& pos, uint32_t max_cascade = NERF_CASCADES()-1) {
	int mip = mip_from_pos(pos, max_cascade);
	dt *= 2*NERF_GRIDSIZE();
	if (dt<1.f) return mip;
	int exponent;
	frexpf(dt, &exponent);
	return min(max_cascade, max(exponent, mip));
}


inline __device__ float pdf_2d(Vector2f sample, uint32_t img, const Vector2i& res, const float* __restrict__ cdf_x_cond_y, const float* __restrict__ cdf_y) {
	Vector2i p = (sample.cwiseProduct(res.cast<float>())).cast<int>().cwiseMax(0).cwiseMin(res - Vector2i::Ones());

	cdf_y += img * res.y();
	cdf_x_cond_y += img * res.y() * res.x() + p.y() * res.x();

	float pmf_y = cdf_y[p.y()];
	if (p.y() > 0) {
		pmf_y -= cdf_y[p.y()-1];
	}

	float pmf_x = cdf_x_cond_y[p.x()];
	if (p.x() > 0) {
		pmf_x -= cdf_x_cond_y[p.x()-1];
	}

	// Probability mass of picking the pixel
	float pmf = pmf_x * pmf_y;

	// To convert to probability density, divide by area of pixel
	return UNIFORM_SAMPLING_FRACTION + pmf * res.prod() * (1.0f - UNIFORM_SAMPLING_FRACTION);
}


inline __device__ Vector2f nerf_random_image_pos_for_tracking(default_rng_t& rng, const Vector2i& resolution, bool snap_to_pixel_centers, const Vector2i& margins, const Vector2i& half_kernel_size) {
	Vector2f xy = random_val_2d(rng);

    Vector2i bounds = resolution - 2*margins - 2*half_kernel_size;

    Vector2i xy_int = xy.cwiseProduct(bounds.cast<float>()).cast<int>();
    xy_int = xy_int + margins + half_kernel_size;
    xy = xy_int.cwiseMax(margins + half_kernel_size).cwiseMin(resolution - margins - half_kernel_size - Vector2i::Ones()).cast<float>();

	if (snap_to_pixel_centers) {
		xy = (xy + Vector2f::Constant(0.5f)).cwiseQuotient(resolution.cast<float>());
	} else {
		xy = xy.cwiseQuotient(resolution.cast<float>());
    }
    return xy;
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

__device__ inline LossAndGradient loss_and_gradient(const Vector3f& target, const Vector3f& prediction, ELossType loss_type) {
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




NGP_NAMESPACE_END
