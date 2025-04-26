/**
 * Limiter V2 - Professional broadcast-quality limiter for OBS Studio
 *
 * Copyright (C) 2023-2025 OBS Project Contributors
 * Based on collaborative discussion and refinement.
 * Contributor: AI Assistant & User Collaboration
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * Features:
 * - Look-ahead processing (0.1ms - 20ms) using efficient circular buffers.
 * - Program-dependent release tuned for speech transparency.
 * - Basic True Peak detection (4x linear interpolation estimation).
 * - Output gain control (-20dB to +20dB).
 * - User presets for common use cases.
 * - Dynamic UI for lookahead settings.
 * - Correct latency reporting for A/V sync.
 * - Efficient buffer management and initialization.
 */

#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <string.h> // For memcpy, memset, memmove, strcmp
#include <stdlib.h> // For fabsf, fmaxf, fminf
#include <stdbool.h> // For bool type

#include <obs-module.h>
#include <media-io/audio-math.h> // For db_to_mul, mul_to_db
#include <util/platform.h>
#include <util/bmem.h> // For bzalloc, brealloc, bfree
#include <util/circlebuf.h> // For circular buffer implementation
// #include <util/threading.h> // Not strictly needed for this filter's core logic

/* -------------------------------------------------------- */
/* Logging                                                  */
/* -------------------------------------------------------- */

// Macro for logging, includes filter name context if available
#define do_log(level, format, ...)                                          \
	blog(level, "[limiter v2: '%s'] " format,                         \
	     cd ? obs_source_get_name(cd->context) : "(unknown)", \
	     ##__VA_ARGS__)

#define warn(format, ...) do_log(LOG_WARNING, format, ##__VA_ARGS__)
#define info(format, ...) do_log(LOG_INFO, format, ##__VA_ARGS__)

#ifdef _DEBUG
#define debug(format, ...) do_log(LOG_DEBUG, format, ##__VA_ARGS__)
#else
#define debug(format, ...)
#endif

/* -------------------------------------------------------- */
/* Definitions and Constants                                */
/* -------------------------------------------------------- */

/* clang-format off */

// Settings Keys (Unique strings identifying filter parameters in obs_data_t)
#define S_PRESET                        "preset_selection"
#define S_FILTER_THRESHOLD              "threshold"
#define S_RELEASE_TIME                  "release_time"
#define S_OUTPUT_GAIN                   "output_gain"
#define S_LOOKAHEAD_ENABLED             "lookahead_enabled"
#define S_LOOKAHEAD_TIME_MS             "lookahead_time_ms"
#define S_ADAPTIVE_RELEASE_ENABLED      "adaptive_release_enabled"
#define S_TRUE_PEAK_ENABLED             "true_peak_enabled"

// UI Text Keys (For localization via obs_module_text)
#define MT_ obs_module_text
#define TEXT_FILTER_NAME                MT_("LimiterV2")
#define TEXT_PRESET                     MT_("LimiterV2.Preset")
#define TEXT_PRESET_DESC                MT_("LimiterV2.Preset.Description")
#define TEXT_PRESET_CUSTOM              MT_("LimiterV2.Preset.Custom")
#define TEXT_PRESET_DEFAULT             MT_("LimiterV2.Preset.Default")
#define TEXT_PRESET_PODCAST             MT_("LimiterV2.Preset.Podcast")
#define TEXT_PRESET_STREAMING           MT_("LimiterV2.Preset.Streaming")
#define TEXT_PRESET_AGGRESSIVE          MT_("LimiterV2.Preset.Aggressive")
#define TEXT_PRESET_TRANSPARENT         MT_("LimiterV2.Preset.Transparent")
#define TEXT_PRESET_MUSIC               MT_("LimiterV2.Preset.Music")
#define TEXT_PRESET_BRICKWALL           MT_("LimiterV2.Preset.Brickwall")
#define TEXT_THRESHOLD                  MT_("LimiterV2.Threshold")
#define TEXT_RELEASE_TIME               MT_("LimiterV2.ReleaseTime")
#define TEXT_RELEASE_TIME_DESC          MT_("LimiterV2.ReleaseTime.Description")
#define TEXT_OUTPUT_GAIN                MT_("LimiterV2.OutputGain")
#define TEXT_LOOKAHEAD_ENABLED          MT_("LimiterV2.LookaheadEnabled")
#define TEXT_LOOKAHEAD_TIME_MS          MT_("LimiterV2.LookaheadTimeMs")
#define TEXT_LOOKAHEAD_TIME_MS_DESC     MT_("LimiterV2.LookaheadTimeMs.Description")
#define TEXT_ADAPTIVE_RELEASE_ENABLED   MT_("LimiterV2.AdaptiveReleaseEnabled")
#define TEXT_ADAPTIVE_RELEASE_DESC      MT_("LimiterV2.AdaptiveReleaseEnabled.Description")
#define TEXT_TRUE_PEAK_ENABLED          MT_("LimiterV2.TruePeakEnabled")
#define TEXT_TRUE_PEAK_DESC             MT_("LimiterV2.TruePeakDescription")

// Preset Internal Value Strings (used in code and settings storage)
#define PRESET_VAL_CUSTOM               "" // Empty string represents custom
#define PRESET_VAL_DEFAULT              "default"
#define PRESET_VAL_PODCAST              "podcast"
#define PRESET_VAL_STREAMING            "streaming"
#define PRESET_VAL_AGGRESSIVE           "aggressive"
#define PRESET_VAL_TRANSPARENT          "transparent"
#define PRESET_VAL_MUSIC                "music"
#define PRESET_VAL_BRICKWALL            "brickwall"

// Parameter Ranges & Defaults
#define MIN_THRESHOLD_DB                -60.0f
#define MAX_THRESHOLD_DB                0.0f
#define MIN_RELEASE_MS                  1.0f
#define MAX_RELEASE_MS                  1000.0f
#define DEFAULT_THRESHOLD_DB            -6.0
#define DEFAULT_RELEASE_MS              60.0
#define DEFAULT_OUTPUT_GAIN_DB          0.0
#define MIN_OUTPUT_GAIN_DB              -20.0f
#define MAX_OUTPUT_GAIN_DB              20.0f
#define DEFAULT_ADAPTIVE_RELEASE        true
#define DEFAULT_LOOKAHEAD_ENABLED       true
#define DEFAULT_LOOKAHEAD_MS            5.0
#define MIN_LOOKAHEAD_MS                0.1f
#define MAX_LOOKAHEAD_MS                20.0f
#define DEFAULT_TRUE_PEAK_ENABLED       true

// Internal Constants
#define FIXED_ATTACK_TIME_MS            1.0f
#define MS_IN_S                         1000
#define MS_IN_S_F                       ((float)MS_IN_S)
#define MIN_ENV_CLAMP                   1e-10f
#define NUM_ENV_HISTORY                 3

// Adaptive Release Tuning Constants
#define ADAPT_SENSITIVITY_THRESHOLD     0.05f
#define ADAPT_SPEED_FACTOR              15.0f
#define ADAPT_MAX_SPEEDUP_FACTOR        3.0f
#define MIN_FAST_RELEASE_MS             1.0f

// True Peak Estimation Constant
#define TP_OVERSAMPLE_FACTOR            4

// Misc Constants
#define MAX_AUDIO_CHANNELS              8
#define INITIAL_ENV_BUF_MS              20

/* clang-format on */

/* -------------------------------------------------------- */
/* Filter Data Structure                                    */
/* -------------------------------------------------------- */

/** @brief Main data structure for the Limiter V2 filter. */
struct limiter_v2_data {
	obs_source_t *context;

	// --- Settings Cache ---
	float threshold_db;
	float release_time_ms;
	float output_gain_db;
	bool adaptive_release_enabled;
	bool lookahead_enabled;
	float lookahead_time_ms;
	bool true_peak_enabled;

	// --- Calculated Coefficients ---
	float attack_coeff;
	float release_coeff; // Base release coefficient
	float output_gain;   // Output gain multiplier

	// --- Processing State ---
	float *envelope_buf;
	size_t envelope_buf_len;
	float envelope;

	// --- Lookahead State ---
	struct circlebuf lookahead_circbuf[MAX_AUDIO_CHANNELS];
	size_t lookahead_samples;
	bool lookahead_buffers_initialized;

	// --- Program-Dependent Release State ---
	float prev_env_vals[NUM_ENV_HISTORY];
	uint32_t prev_env_pos;

	// --- System Info ---
	uint32_t sample_rate;
	size_t num_channels;
};

/* -------------------------------------------------------- */
/* Helper Functions                                         */
/* -------------------------------------------------------- */

/**
 * @brief Calculates the gain coefficient for exponential smoothing (attack/release).
 *
 * @param sample_rate Audio sample rate in Hz.
 * @param time_ms Time constant in milliseconds.
 * @return float Gain coefficient (0.0 to < 1.0). Returns 0.0 for invalid input.
 */
static inline float gain_coefficient(uint32_t sample_rate, float time_ms)
{
	if (sample_rate == 0 || time_ms <= 0.0f) {
		return 0.0f;
	}
	float time_sec = time_ms / MS_IN_S_F;
	return (float)exp(-1.0f / ((sample_rate * time_sec) + MIN_ENV_CLAMP));
}

/**
 * @brief Calculates a simple measure of envelope change rate based on recent history.
 *
 * @param cd Limiter data structure containing previous envelope values.
 * @return float Average absolute difference between recent envelope values.
 */
static inline float calculate_env_change_rate(struct limiter_v2_data *cd)
{
	if (!cd) return 0.0f;
	float change_sum = 0.0f;
	for (uint32_t i = 0; i < NUM_ENV_HISTORY - 1; ++i) {
		uint32_t idx = (cd->prev_env_pos + i) % NUM_ENV_HISTORY;
		uint32_t next_idx = (cd->prev_env_pos + i + 1) % NUM_ENV_HISTORY;
		change_sum += fabsf(cd->prev_env_vals[next_idx] - cd->prev_env_vals[idx]);
	}
	return change_sum / (float)(NUM_ENV_HISTORY - 1 + MIN_ENV_CLAMP);
}

/**
 * @brief Ensures the internal envelope buffer is large enough. Resizes if necessary.
 *
 * @param cd Filter data structure.
 * @param num_samples Required number of samples for the buffer.
 * @return bool True if buffer is valid and large enough, false otherwise.
 */
static bool ensure_env_buffer(struct limiter_v2_data *cd, size_t num_samples)
{
	if (!cd) return false;
	if (cd->envelope_buf_len < num_samples) {
		size_t new_len = num_samples;
		float *new_buf = brealloc(cd->envelope_buf, new_len * sizeof(float));
		if (!new_buf) {
			warn("Failed to reallocate envelope buffer (requested %zu samples)", new_len);
			bfree(cd->envelope_buf);
			cd->envelope_buf = NULL;
			cd->envelope_buf_len = 0;
			return false;
		}
		cd->envelope_buf = new_buf;
		cd->envelope_buf_len = new_len;
		debug("Resized envelope buffer to %zu samples", new_len);
	}
	return (cd->envelope_buf != NULL);
}

/**
 * @brief Initializes or updates the lookahead circular buffers.
 *
 * @param cd Filter data structure.
 * @return bool True on success, false on allocation failure.
 */
static bool update_lookahead_buffers(struct limiter_v2_data *cd)
{
	if (!cd) return false;

	if (cd->lookahead_buffers_initialized) {
		for (size_t i = 0; i < cd->num_channels; ++i) {
			if (i < MAX_AUDIO_CHANNELS) {
				circlebuf_free(&cd->lookahead_circbuf[i]);
			}
		}
		cd->lookahead_buffers_initialized = false;
		debug("Freed existing lookahead buffers.");
	}

	if (!cd->lookahead_enabled || cd->lookahead_samples == 0 || cd->num_channels == 0) {
		debug("Lookahead disabled or zero samples/channels, buffers not needed.");
		return true;
	}

	debug("Initializing lookahead buffers for %zu samples, %zu channels.",
	      cd->lookahead_samples, cd->num_channels);

	size_t block_estimate = (cd->sample_rate * INITIAL_ENV_BUF_MS / MS_IN_S_F) + 1;
	if (block_estimate == 0) block_estimate = 1024;
	size_t required_capacity_samples = cd->lookahead_samples + block_estimate;
	size_t required_capacity_bytes = required_capacity_samples * sizeof(float);

	float *zero_block = NULL;
	bool success = true;

	size_t init_zeros_size_bytes = cd->lookahead_samples * sizeof(float);
	if (init_zeros_size_bytes > 0) {
		zero_block = bzalloc(init_zeros_size_bytes);
		if (!zero_block) {
			warn("Failed to allocate zero block for lookahead initialization.");
			return false;
		}
	}

	for (size_t i = 0; i < cd->num_channels; ++i) {
		if (i >= MAX_AUDIO_CHANNELS) {
			warn("Channel index %zu exceeds MAX_AUDIO_CHANNELS %d", i, MAX_AUDIO_CHANNELS);
			success = false; break;
		}

		memset(&cd->lookahead_circbuf[i], 0, sizeof(struct circlebuf));
		circlebuf_reserve(&cd->lookahead_circbuf[i], required_capacity_bytes);

		if (cd->lookahead_circbuf[i].capacity < required_capacity_bytes) {
			warn("Failed to reserve sufficient capacity (%zu bytes) for lookahead buffer channel %zu.",
			     required_capacity_bytes, i);
			circlebuf_free(&cd->lookahead_circbuf[i]);
			success = false;
			break;
		}

		if (zero_block && init_zeros_size_bytes > 0) {
			circlebuf_push_back(&cd->lookahead_circbuf[i], zero_block, init_zeros_size_bytes);
		}
	}

	bfree(zero_block);

	if (!success) {
		warn("Lookahead buffer initialization failed. Cleaning up.");
		for (size_t i = 0; i < cd->num_channels; ++i) {
			if (i < MAX_AUDIO_CHANNELS) {
				circlebuf_free(&cd->lookahead_circbuf[i]);
			}
		}
		cd->lookahead_buffers_initialized = false;
		return false;
	}

	cd->lookahead_buffers_initialized = true;
	debug("Lookahead buffers initialized successfully.");
	return true;
}

/**
 * @brief Estimates the true peak level using basic linear interpolation.
 *
 * @param current_sample The current audio sample value.
 * @param next_sample The next audio sample value.
 * @return float Estimated absolute peak value between the two samples.
 */
static inline float get_inter_sample_peak_estimate(float current_sample, float next_sample)
{
	if (!isfinite(current_sample) || !isfinite(next_sample)) {
		return 0.0f;
	}
	float max_abs_val = fmaxf(fabsf(current_sample), fabsf(next_sample));
	for (int j = 1; j < TP_OVERSAMPLE_FACTOR; ++j) {
		float t = (float)j / TP_OVERSAMPLE_FACTOR;
		float interpolated_sample = (1.0f - t) * current_sample + t * next_sample;
		if (isfinite(interpolated_sample)) {
			max_abs_val = fmaxf(max_abs_val, fabsf(interpolated_sample));
		}
	}
	return max_abs_val;
}

/* -------------------------------------------------------- */
/* Core Processing Functions                                */
/* -------------------------------------------------------- */

/**
 * @brief Calculates the signal envelope. Incorporates TP and PDR.
 *
 * @param cd Filter data structure.
 * @param samples Input audio sample buffer (undelayed).
 * @param num_samples Number of samples to analyze.
 */
static void analyze_envelope(struct limiter_v2_data *cd, float **samples, const uint32_t num_samples)
{
	if (num_samples == 0 || !cd || !samples) return;
	if (!ensure_env_buffer(cd, num_samples)) {
		warn("Cannot analyze envelope, buffer invalid or too small.");
		return;
	}

	const float attack_coeff = cd->attack_coeff;
	memset(cd->envelope_buf, 0, num_samples * sizeof(float));

	for (size_t chan = 0; chan < cd->num_channels; ++chan) {
		if (!samples[chan]) continue;

		float *envelope_buf = cd->envelope_buf;
		float env = cd->envelope;

		for (uint32_t i = 0; i < num_samples; ++i) {
			float env_in;
			if (cd->true_peak_enabled && i < num_samples - 1) {
				env_in = get_inter_sample_peak_estimate(samples[chan][i], samples[chan][i + 1]);
			} else {
				env_in = fabsf(samples[chan][i]);
			}
			if (!isfinite(env_in)) env_in = 0.0f;

			float current_release_coeff = cd->release_coeff;
			if (cd->adaptive_release_enabled) {
				float env_change_rate = calculate_env_change_rate(cd);
				if (env_change_rate > ADAPT_SENSITIVITY_THRESHOLD) {
					float release_factor = fmaxf(1.0f, fminf(ADAPT_MAX_SPEEDUP_FACTOR, env_change_rate * ADAPT_SPEED_FACTOR));
					float fast_release_time_ms = cd->release_time_ms / release_factor;
					fast_release_time_ms = fmaxf(fast_release_time_ms, MIN_FAST_RELEASE_MS);
					current_release_coeff = gain_coefficient(cd->sample_rate, fast_release_time_ms);
				}
			}

			if (env < env_in) {
				env = env_in + attack_coeff * (env - env_in);
			} else {
				env = env_in + current_release_coeff * (env - env_in);
			}
			if (!isfinite(env) || env < MIN_ENV_CLAMP) env = 0.0f;

			envelope_buf[i] = fmaxf(envelope_buf[i], env);
		}
	}

	cd->envelope = (num_samples > 0) ? cd->envelope_buf[num_samples - 1] : cd->envelope;
	if (!isfinite(cd->envelope)) cd->envelope = 0.0f;

	cd->prev_env_pos = (cd->prev_env_pos + 1) % NUM_ENV_HISTORY;
	cd->prev_env_vals[cd->prev_env_pos] = cd->envelope;
}


/**
 * @brief Applies gain reduction based on the calculated envelope and threshold.
 *
 * @param cd Filter data structure.
 * @param samples Audio sample buffer (potentially delayed).
 * @param num_samples Number of samples to process.
 */
static inline void process_compression(const struct limiter_v2_data *cd, float **samples, uint32_t num_samples)
{
	if (num_samples == 0) return;
	if (!cd || !samples || !cd->envelope_buf || cd->envelope_buf_len < num_samples) {
		warn("Cannot process compression, invalid state or buffer");
		return;
	}

	for (uint32_t i = 0; i < num_samples; ++i) {
		const float env_lin = cd->envelope_buf[i];
		float gain_reduction_multiplier = 1.0f;

		if (env_lin > MIN_ENV_CLAMP) {
			const float env_db = mul_to_db(env_lin);
			if (isfinite(env_db) && env_db > cd->threshold_db) {
				float gain_reduction_db = cd->threshold_db - env_db;
				gain_reduction_multiplier = db_to_mul(fminf(0.0f, gain_reduction_db));
				if (!isfinite(gain_reduction_multiplier)) gain_reduction_multiplier = 0.0f;
			}
		}

		float final_gain = gain_reduction_multiplier * cd->output_gain;
		if (!isfinite(final_gain)) final_gain = 0.0f;

		for (size_t c = 0; c < cd->num_channels; ++c) {
			if (samples[c] && isfinite(samples[c][i])) {
				samples[c][i] *= final_gain;
			} else if (samples[c]) {
				samples[c][i] = 0.0f;
			}
		}
	}
}


/* -------------------------------------------------------- */
/* OBS Filter API Implementation                            */
/* -------------------------------------------------------- */

/** Returns the display name of the filter */
static const char *limiter_v2_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return TEXT_FILTER_NAME;
}

/** Sets the default values for the filter settings */
static void limiter_v2_defaults(obs_data_t *s)
{
	obs_data_set_default_string(s, S_PRESET, PRESET_VAL_DEFAULT);
	obs_data_set_default_double(s, S_FILTER_THRESHOLD, DEFAULT_THRESHOLD_DB);
	obs_data_set_default_double(s, S_RELEASE_TIME, DEFAULT_RELEASE_MS);
	obs_data_set_default_double(s, S_OUTPUT_GAIN, DEFAULT_OUTPUT_GAIN_DB);
	obs_data_set_default_bool(s, S_ADAPTIVE_RELEASE_ENABLED, DEFAULT_ADAPTIVE_RELEASE);
	obs_data_set_default_bool(s, S_LOOKAHEAD_ENABLED, DEFAULT_LOOKAHEAD_ENABLED);
	obs_data_set_default_double(s, S_LOOKAHEAD_TIME_MS, DEFAULT_LOOKAHEAD_MS);
	obs_data_set_default_bool(s, S_TRUE_PEAK_ENABLED, DEFAULT_TRUE_PEAK_ENABLED);
}

/** Modified callback for when a preset is selected */
static bool preset_modified_callback(obs_properties_t *props, obs_property_t *p, obs_data_t *s)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(p);

	const char* selected_preset = obs_data_get_string(s, S_PRESET);
	blog(LOG_INFO, "[limiter v2] Preset selected: %s", selected_preset ? selected_preset : "(null / custom)");

	if (!selected_preset) return true; // Should not happen if default is set

	// Apply settings based on the selected preset string
	if (strcmp(selected_preset, PRESET_VAL_DEFAULT) == 0) {
		obs_data_set_double(s, S_FILTER_THRESHOLD, DEFAULT_THRESHOLD_DB);
		obs_data_set_double(s, S_RELEASE_TIME, DEFAULT_RELEASE_MS);
		obs_data_set_double(s, S_OUTPUT_GAIN, DEFAULT_OUTPUT_GAIN_DB);
		obs_data_set_bool(s, S_ADAPTIVE_RELEASE_ENABLED, DEFAULT_ADAPTIVE_RELEASE);
		obs_data_set_bool(s, S_LOOKAHEAD_ENABLED, DEFAULT_LOOKAHEAD_ENABLED);
		obs_data_set_double(s, S_LOOKAHEAD_TIME_MS, DEFAULT_LOOKAHEAD_MS);
		obs_data_set_bool(s, S_TRUE_PEAK_ENABLED, DEFAULT_TRUE_PEAK_ENABLED);
	} else if (strcmp(selected_preset, PRESET_VAL_PODCAST) == 0) {
		obs_data_set_double(s, S_FILTER_THRESHOLD, -8.0);
		obs_data_set_double(s, S_RELEASE_TIME, 80.0);
		obs_data_set_double(s, S_OUTPUT_GAIN, 0.0);
		obs_data_set_bool(s, S_ADAPTIVE_RELEASE_ENABLED, true);
		obs_data_set_bool(s, S_LOOKAHEAD_ENABLED, true);
		obs_data_set_double(s, S_LOOKAHEAD_TIME_MS, 8.0);
		obs_data_set_bool(s, S_TRUE_PEAK_ENABLED, true);
	} else if (strcmp(selected_preset, PRESET_VAL_STREAMING) == 0) {
		obs_data_set_double(s, S_FILTER_THRESHOLD, -7.0);
		obs_data_set_double(s, S_RELEASE_TIME, 70.0);
		obs_data_set_double(s, S_OUTPUT_GAIN, 1.0);
		obs_data_set_bool(s, S_ADAPTIVE_RELEASE_ENABLED, true);
		obs_data_set_bool(s, S_LOOKAHEAD_ENABLED, true);
		obs_data_set_double(s, S_LOOKAHEAD_TIME_MS, 3.0);
		obs_data_set_bool(s, S_TRUE_PEAK_ENABLED, true);
	} else if (strcmp(selected_preset, PRESET_VAL_AGGRESSIVE) == 0) {
		obs_data_set_double(s, S_FILTER_THRESHOLD, -5.0);
		obs_data_set_double(s, S_RELEASE_TIME, 40.0);
		obs_data_set_double(s, S_OUTPUT_GAIN, 3.0);
		obs_data_set_bool(s, S_ADAPTIVE_RELEASE_ENABLED, true);
		obs_data_set_bool(s, S_LOOKAHEAD_ENABLED, true);
		obs_data_set_double(s, S_LOOKAHEAD_TIME_MS, 2.0);
		obs_data_set_bool(s, S_TRUE_PEAK_ENABLED, true);
	} else if (strcmp(selected_preset, PRESET_VAL_TRANSPARENT) == 0) {
		obs_data_set_double(s, S_FILTER_THRESHOLD, -1.5);
		obs_data_set_double(s, S_RELEASE_TIME, 50.0);
		obs_data_set_double(s, S_OUTPUT_GAIN, 0.0);
		obs_data_set_bool(s, S_ADAPTIVE_RELEASE_ENABLED, true);
		obs_data_set_bool(s, S_LOOKAHEAD_ENABLED, true);
		obs_data_set_double(s, S_LOOKAHEAD_TIME_MS, 5.0);
		obs_data_set_bool(s, S_TRUE_PEAK_ENABLED, true);
	} else if (strcmp(selected_preset, PRESET_VAL_MUSIC) == 0) {
		obs_data_set_double(s, S_FILTER_THRESHOLD, -2.0);
		obs_data_set_double(s, S_RELEASE_TIME, 200.0);
		obs_data_set_double(s, S_OUTPUT_GAIN, 0.0);
		obs_data_set_bool(s, S_ADAPTIVE_RELEASE_ENABLED, false);
		obs_data_set_bool(s, S_LOOKAHEAD_ENABLED, true);
		obs_data_set_double(s, S_LOOKAHEAD_TIME_MS, 2.0);
		obs_data_set_bool(s, S_TRUE_PEAK_ENABLED, true);
	} else if (strcmp(selected_preset, PRESET_VAL_BRICKWALL) == 0) {
		obs_data_set_double(s, S_FILTER_THRESHOLD, -0.3);
		obs_data_set_double(s, S_RELEASE_TIME, 50.0);
		obs_data_set_double(s, S_OUTPUT_GAIN, 0.0);
		obs_data_set_bool(s, S_ADAPTIVE_RELEASE_ENABLED, false);
		obs_data_set_bool(s, S_LOOKAHEAD_ENABLED, true);
		obs_data_set_double(s, S_LOOKAHEAD_TIME_MS, 1.5);
		obs_data_set_bool(s, S_TRUE_PEAK_ENABLED, true);
	} else {
		// Selected "Custom" - do not change other settings.
		return true;
	}
	return true;
}

/** Modified callback to set preset to "Custom" if any individual parameter is changed */
static bool parameter_modified_callback(obs_properties_t *props, obs_property_t *p, obs_data_t *s)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(p);
	const char* current_preset = obs_data_get_string(s, S_PRESET);
	if (current_preset && strcmp(current_preset, PRESET_VAL_CUSTOM) != 0) {
		blog(LOG_DEBUG, "[limiter v2] Manual parameter change detected, setting preset to Custom.");
		obs_data_set_string(s, S_PRESET, PRESET_VAL_CUSTOM);
	}
	return true;
}

/** Modified callback for lookahead enable toggle (handles visibility) */
static bool lookahead_enabled_modified_callback(obs_properties_t *props, obs_property_t *p, obs_data_t *s)
{
	// First, handle setting preset to custom
	parameter_modified_callback(props, p, s);

	// Then, handle visibility
	bool enabled = obs_data_get_bool(s, S_LOOKAHEAD_ENABLED);
	obs_property_t *time_slider = obs_properties_get(props, S_LOOKAHEAD_TIME_MS);
	if (time_slider) {
		obs_property_set_visible(time_slider, enabled);
	}
	return true;
}

/** Creates the properties view for the filter in the OBS UI */
static obs_properties_t *limiter_v2_properties(void *data)
{
	struct limiter_v2_data *cd = data;

	obs_properties_t *props = obs_properties_create();
	if (!props) {
		warn("Failed to create properties");
		return NULL;
	}

	obs_properties_set_flags(props, OBS_PROPERTIES_DEFER_UPDATE);

	// --- Presets Dropdown ---
	obs_property_t *preset_list = obs_properties_add_list(props, S_PRESET, TEXT_PRESET,
	                                                     OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	if (preset_list) {
		obs_property_list_add_string(preset_list, TEXT_PRESET_CUSTOM, PRESET_VAL_CUSTOM);
		obs_property_list_add_string(preset_list, TEXT_PRESET_DEFAULT, PRESET_VAL_DEFAULT);
		obs_property_list_add_string(preset_list, TEXT_PRESET_PODCAST, PRESET_VAL_PODCAST);
		obs_property_list_add_string(preset_list, TEXT_PRESET_STREAMING, PRESET_VAL_STREAMING);
		obs_property_list_add_string(preset_list, TEXT_PRESET_AGGRESSIVE, PRESET_VAL_AGGRESSIVE);
		obs_property_list_add_string(preset_list, TEXT_PRESET_TRANSPARENT, PRESET_VAL_TRANSPARENT);
		obs_property_list_add_string(preset_list, TEXT_PRESET_MUSIC, PRESET_VAL_MUSIC);
		obs_property_list_add_string(preset_list, TEXT_PRESET_BRICKWALL, PRESET_VAL_BRICKWALL);
		obs_property_set_long_description(preset_list, TEXT_PRESET_DESC);
		obs_property_set_modified_callback(preset_list, preset_modified_callback);
	} else {
		warn("Failed to create preset list property");
	}

	// --- Individual Parameter Controls ---
	obs_property_t *p;

	p = obs_properties_add_float_slider(props, S_FILTER_THRESHOLD, TEXT_THRESHOLD,
					    MIN_THRESHOLD_DB, MAX_THRESHOLD_DB, 0.1);
	obs_property_float_set_suffix(p, " dB");
	obs_property_set_modified_callback(p, parameter_modified_callback);

	p = obs_properties_add_float_slider(props, S_RELEASE_TIME, TEXT_RELEASE_TIME,
					  MIN_RELEASE_MS, MAX_RELEASE_MS, 1.0);
	obs_property_float_set_suffix(p, " ms");
	obs_property_set_long_description(p, TEXT_RELEASE_TIME_DESC);
	obs_property_set_modified_callback(p, parameter_modified_callback);

	p = obs_properties_add_float_slider(props, S_OUTPUT_GAIN, TEXT_OUTPUT_GAIN,
					    MIN_OUTPUT_GAIN_DB, MAX_OUTPUT_GAIN_DB, 0.1);
	obs_property_float_set_suffix(p, " dB");
	obs_property_set_modified_callback(p, parameter_modified_callback);

	p = obs_properties_add_bool(props, S_LOOKAHEAD_ENABLED, TEXT_LOOKAHEAD_ENABLED);
	// Use specific callback for lookahead toggle to handle both visibility and preset change
	obs_property_set_modified_callback(p, lookahead_enabled_modified_callback);

	p = obs_properties_add_float_slider(props, S_LOOKAHEAD_TIME_MS, TEXT_LOOKAHEAD_TIME_MS,
					    MIN_LOOKAHEAD_MS, MAX_LOOKAHEAD_MS, 0.1);
	obs_property_float_set_suffix(p, " ms");
	obs_property_set_long_description(p, TEXT_LOOKAHEAD_TIME_MS_DESC);
    obs_property_set_visible(p, cd ? cd->lookahead_enabled : DEFAULT_LOOKAHEAD_ENABLED);
    obs_property_set_modified_callback(p, parameter_modified_callback);

	p = obs_properties_add_bool(props, S_ADAPTIVE_RELEASE_ENABLED, TEXT_ADAPTIVE_RELEASE_ENABLED);
	obs_property_set_long_description(p, TEXT_ADAPTIVE_RELEASE_DESC);
	obs_property_set_modified_callback(p, parameter_modified_callback);

	p = obs_properties_add_bool(props, S_TRUE_PEAK_ENABLED, TEXT_TRUE_PEAK_ENABLED);
    obs_property_set_long_description(p, TEXT_TRUE_PEAK_DESC);
    obs_property_set_modified_callback(p, parameter_modified_callback);

	return props;
}


/** Called when filter settings are updated */
static void limiter_v2_update(void *data, obs_data_t *s)
{
	struct limiter_v2_data *cd = data;
	if (!cd) return;

	const uint32_t sample_rate = audio_output_get_sample_rate(obs_get_audio());
	size_t num_channels = audio_output_get_channels(obs_get_audio());

	// Clamp channel count and check validity
	if (num_channels > MAX_AUDIO_CHANNELS) {
		warn("Limiter V2 supports up to %d channels, clamped to %d.", MAX_AUDIO_CHANNELS, MAX_AUDIO_CHANNELS);
		num_channels = MAX_AUDIO_CHANNELS;
	}
	uint32_t current_sample_rate = sample_rate;
	if (num_channels == 0 || current_sample_rate == 0) {
		warn("Invalid audio parameters (Channels: %zu, Sample Rate: %u). Limiter may not function.", num_channels, current_sample_rate);
		if (current_sample_rate == 0) {
			current_sample_rate = 48000; // Assign a default if SR is 0
			warn("Sample rate was 0, assigned default %uHz.", current_sample_rate);
		}
	}


	bool reset_lookahead = false;
	// Check if critical audio parameters changed
	if (cd->sample_rate != current_sample_rate || cd->num_channels != num_channels) {
		info("Audio parameters changed (SR: %u->%u, Ch: %zu->%zu), resetting state.",
		     cd->sample_rate, current_sample_rate, cd->num_channels, num_channels);
		cd->sample_rate = current_sample_rate;
		cd->num_channels = num_channels;
		reset_lookahead = true;
		cd->envelope = 0.0f;
		memset(cd->prev_env_vals, 0, sizeof(cd->prev_env_vals));
		cd->prev_env_pos = 0;
	}

	// --- Read ACTUAL parameter values from settings ---
	cd->threshold_db = (float)obs_data_get_double(s, S_FILTER_THRESHOLD);
	cd->release_time_ms = (float)obs_data_get_double(s, S_RELEASE_TIME);
	cd->output_gain_db = (float)obs_data_get_double(s, S_OUTPUT_GAIN);
	cd->adaptive_release_enabled = obs_data_get_bool(s, S_ADAPTIVE_RELEASE_ENABLED);
	cd->true_peak_enabled = obs_data_get_bool(s, S_TRUE_PEAK_ENABLED);

	bool new_lookahead_enabled = obs_data_get_bool(s, S_LOOKAHEAD_ENABLED);
	float new_lookahead_time_ms = (float)obs_data_get_double(s, S_LOOKAHEAD_TIME_MS);

	// Clamp lookahead time read
	new_lookahead_time_ms = fmaxf(0.0f, fminf(MAX_LOOKAHEAD_MS, new_lookahead_time_ms));
	if (new_lookahead_enabled && new_lookahead_time_ms < MIN_LOOKAHEAD_MS) {
		 new_lookahead_time_ms = MIN_LOOKAHEAD_MS;
	} else if (!new_lookahead_enabled) {
		new_lookahead_time_ms = 0.0f;
	}


	// Check if lookahead settings changed compared to cached values
	if (new_lookahead_enabled != cd->lookahead_enabled ||
	    new_lookahead_time_ms != cd->lookahead_time_ms) {
		reset_lookahead = true;
		cd->lookahead_enabled = new_lookahead_enabled;
		cd->lookahead_time_ms = new_lookahead_time_ms;
	}

	// --- Update Internal State ---
	cd->attack_coeff = gain_coefficient(cd->sample_rate, FIXED_ATTACK_TIME_MS);
	cd->release_time_ms = fmaxf(MIN_RELEASE_MS, fminf(MAX_RELEASE_MS, cd->release_time_ms));
	cd->release_coeff = gain_coefficient(cd->sample_rate, cd->release_time_ms);
	cd->output_gain = db_to_mul(cd->output_gain_db);

	// --- Update Lookahead Buffer (Only if necessary) ---
	if (reset_lookahead) {
		size_t new_lookahead_samples = 0;
		if (cd->lookahead_enabled && cd->lookahead_time_ms >= MIN_LOOKAHEAD_MS && cd->sample_rate > 0 && cd->num_channels > 0) {
			new_lookahead_samples = (size_t)((cd->sample_rate * cd->lookahead_time_ms) / MS_IN_S_F + 0.5f);
			if (new_lookahead_samples == 0) new_lookahead_samples = 1; // Minimum 1 sample
		}
		cd->lookahead_samples = new_lookahead_samples;

		if (!update_lookahead_buffers(cd)) {
			warn("Disabling lookahead due to buffer allocation failure.");
			cd->lookahead_enabled = false;
			cd->lookahead_samples = 0;
		}
	}

	// --- Report Latency to OBS ---
	uint64_t latency_ns = 0;
	if (cd->lookahead_enabled && cd->lookahead_buffers_initialized && cd->lookahead_samples > 0 && cd->sample_rate > 0) {
		latency_ns = (uint64_t)(((double)cd->lookahead_samples / cd->sample_rate) * 1.0e9);
	}
	obs_source_set_audio_latency(cd->context, latency_ns);

	// --- Ensure Envelope Buffer Exists ---
	if (!cd->envelope_buf) {
		size_t initial_env_len = (cd->sample_rate * INITIAL_ENV_BUF_MS) / MS_IN_S;
		if (initial_env_len == 0) initial_env_len = 1024;
		ensure_env_buffer(cd, initial_env_len);
		if (cd->envelope_buf) {
             memset(cd->envelope_buf, 0, cd->envelope_buf_len * sizeof(float));
        }
	}
}

/** Called when the filter is created */
static void *limiter_v2_create(obs_data_t *settings, obs_source_t *filter)
{
	struct limiter_v2_data *cd = bzalloc(sizeof(struct limiter_v2_data));
	if (!cd) {
		blog(LOG_ERROR, "[limiter v2] Failed to allocate filter data structure");
		return NULL;
	}
	cd->context = filter;
	cd->envelope = 0.0f;
	memset(cd->prev_env_vals, 0, sizeof(cd->prev_env_vals));
	cd->prev_env_pos = 0;
	cd->lookahead_buffers_initialized = false;

	// Apply defaults first to ensure all settings exist in 'settings' object
	// This prevents issues if the filter is added for the first time and settings are empty
	limiter_v2_defaults(settings);
	// Then call update to initialize state based on potentially saved/modified settings
	limiter_v2_update(cd, settings);
	info("Limiter v2 filter created");
	return cd;
}

/** Called when the filter is destroyed */
static void limiter_v2_destroy(void *data)
{
	struct limiter_v2_data *cd = data;
	if (!cd) return;

	debug("Destroying limiter v2 filter");

	if (cd->lookahead_buffers_initialized) {
		for (size_t i = 0; i < cd->num_channels; ++i) {
			if (i < MAX_AUDIO_CHANNELS) {
				circlebuf_free(&cd->lookahead_circbuf[i]);
			}
		}
	}

	bfree(cd->envelope_buf);
	bfree(cd);
}


/** The main audio processing callback */
static struct obs_audio_data *limiter_v2_filter_audio(void *data, struct obs_audio_data *audio)
{
	struct limiter_v2_data *cd = data;
	if (!cd || !audio || !audio->data || audio->frames == 0 || !cd->sample_rate || cd->num_channels == 0) {
		return audio;
	}

	const uint32_t num_samples = audio->frames;
	const size_t sample_size = sizeof(float);
	float **samples = (float **)audio->data;
	bool lookahead_active = cd->lookahead_enabled && cd->lookahead_buffers_initialized && cd->lookahead_samples > 0;

	// --- 1. Analyze Undelayed Input ---
	analyze_envelope(cd, samples, num_samples);

	// --- 2. Apply Lookahead Delay (if enabled) ---
	if (lookahead_active) {
		for (size_t c = 0; c < cd->num_channels; ++c) {
			if (c >= MAX_AUDIO_CHANNELS || !samples[c]) continue;
			// Safety check buffer before use
			if (cd->lookahead_circbuf[c].size == 0 && cd->lookahead_circbuf[c].capacity == 0) {
				warn("Lookahead buffer for channel %zu not initialized, skipping lookahead for block.", c);
				lookahead_active = false; break;
			}
			circlebuf_push_back(&cd->lookahead_circbuf[c], samples[c], num_samples * sample_size);
			circlebuf_pop_front(&cd->lookahead_circbuf[c], samples[c], num_samples * sample_size);
		}
	}

	// --- 3. Process Compression ---
	process_compression(cd, samples, num_samples);

	return audio;
}


/* -------------------------------------------------------- */
/* OBS Filter Definition Structure                          */
/* -------------------------------------------------------- */

struct obs_source_info limiter_v2_filter = {
	.id             = "limiter_v2_filter", // Unique ID
	.type           = OBS_SOURCE_TYPE_FILTER,
	.output_flags   = OBS_SOURCE_AUDIO,
	.get_name       = limiter_v2_name,
	.create         = limiter_v2_create,
	.destroy        = limiter_v2_destroy,
	.update         = limiter_v2_update,
	.filter_audio   = limiter_v2_filter_audio,
	.get_defaults   = limiter_v2_defaults,
	.get_properties = limiter_v2_properties,
};

// Integration Notes:
// 1. Register `limiter_v2_filter` in obs-filters.c `obs_module_load`.
// 2. Add localization strings (LimiterV2, LimiterV2.Preset.*, etc.) to en-US.ini and other locale files.
// 3. Add this file to plugins/obs-filters/CMakeLists.txt.
// 4. Ensure build environment links necessary OBS utilities (libobs).