/**
 * Limiter V2 - Enhanced audio limiter filter for OBS Studio
 *
 * Copyright (C) 2025-Present OBS Project Contributors
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
 * Key Features:
 * - Lookahead limiting (0.1ms - 20ms) using efficient circular buffers to pre-analyze peaks.
 * - Program-dependent release (PDR) logic tuned for transparent speech compression.
 * - Basic True Peak detection via 4x oversampled linear interpolation estimation (not filter-bank).
 * - Output gain control (-20dB to +20dB) applied post-limiting.
 * - User-selectable presets for common broadcast, streaming, and music use cases.
 * - Dynamic UI: enables/disables lookahead slider based on lookahead toggle.
 * - Accurate audio latency reporting for proper A/V sync with lookahead enabled.
 * - Safe and efficient buffer management for low CPU overhead and reliable operation.
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

/* Prevent compiler warnings for unused parameters */
#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) (void)x
#endif


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

// Settings Keys
#define S_PRESET                        "preset_selection"
#define S_FILTER_THRESHOLD              "threshold"
#define S_RELEASE_TIME                  "release_time"
#define S_OUTPUT_GAIN                   "output_gain"
#define S_LOOKAHEAD_ENABLED             "lookahead_enabled"
#define S_LOOKAHEAD_TIME_MS             "lookahead_time_ms"
#define S_ADAPTIVE_RELEASE_ENABLED      "adaptive_release_enabled"
#define S_TRUE_PEAK_ENABLED             "true_peak_enabled"

// UI Text Keys
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

// Preset Internal Value Strings
#define PRESET_VAL_CUSTOM               ""
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
#define SMALL_EPSILON                   1e-10f // Used to prevent division by zero etc.
#define NUM_ENV_HISTORY                 3      // Size of envelope history for PDR

// Adaptive Release Tuning Constants (Empirical starting points)
#define ADAPT_SENSITIVITY_THRESHOLD     0.05f  // Threshold of avg envelope change rate to trigger adaptation.
#define ADAPT_SPEED_FACTOR              15.0f  // Multiplier affecting adaptation speed based on change rate.
#define ADAPT_MAX_SPEEDUP_FACTOR        3.0f   // Max factor by which release can speed up (e.g., 3x).
#define MIN_FAST_RELEASE_MS             1.0f   // Floor for the adapted release time.

// True Peak Estimation Constant
#define TP_OVERSAMPLE_FACTOR            4      // Factor for linear interpolation between samples.

// Misc Constants
#define MAX_AUDIO_CHANNELS              8      // Static limit for channel processing arrays.
#define INITIAL_ENV_BUF_MS              20     // Default envelope buffer size guess.

/* clang-format on */

/* -------------------------------------------------------- */
/* Filter Data Structure                                    */
/* -------------------------------------------------------- */

/**
 * @brief Main data structure for the Limiter V2 filter instance
 *
 * This structure holds all the state data for a limiter instance including:
 * - Current user settings from the UI
 * - Calculated coefficients for DSP operations
 * - Processing state (envelope tracking, lookahead buffers)
 * - Audio system parameters (sample rate, channel count)
 *
 * Memory management:
 * - The structure itself is allocated in limiter_v2_create
 * - Dynamic buffers (envelope_buf, lookahead_circbuf) are managed internally
 * - Everything is freed in limiter_v2_destroy
 *
 * Thread safety:
 * - This structure and its members are accessed only from the OBS audio
 *   thread during filter callbacks (create, destroy, update, filter_audio)
 * - No additional synchronization is needed as OBS guarantees thread safety
 */
struct limiter_v2_data {
	obs_source_t *context; // OBS filter context

	// Settings Cache
	float threshold_db;
	float release_time_ms;
	float output_gain_db;
	bool adaptive_release_enabled;
	bool lookahead_enabled;
	float lookahead_time_ms;
	bool true_peak_enabled;

	// Calculated Coefficients
	float attack_coeff;
	float release_coeff; // Base release coefficient
	float output_gain;   // Output gain multiplier

	// Processing State
	float *envelope_buf;
	size_t envelope_buf_len;
	float envelope; // Last envelope value

	// Lookahead State
	struct circlebuf lookahead_circbuf[MAX_AUDIO_CHANNELS];
	size_t lookahead_samples;
	bool lookahead_buffers_initialized;

	// Program-Dependent Release State
	float prev_env_vals[NUM_ENV_HISTORY];
	uint32_t prev_env_pos;

	// System Info
	uint32_t sample_rate;
	size_t num_channels;
};

/* -------------------------------------------------------- */
/* Helper Functions                                         */
/* -------------------------------------------------------- */

/**
 * @brief Calculates the coefficient for smoothed envelope attack/release
 * @param sample_rate The audio sample rate in Hz
 * @param time_ms The time constant in milliseconds
 * @return The coefficient value (between 0.0 and 1.0)
 *
 * Converts the time constant (in ms) to a coefficient suitable for
 * single-pole IIR filter implementation of attack and release curves.
 * This is a digital approximation of analog RC time constants often
 * used in analog dynamics processors.
 */
static inline float gain_coefficient(uint32_t sample_rate, float time_ms)
{
	if (sample_rate == 0 || time_ms <= 0.0f) {
		return 0.0f;
	}
	float time_sec = time_ms / MS_IN_S_F;
	return (float)exp(-1.0f / ((sample_rate * time_sec) + SMALL_EPSILON));
}

/**
 * @brief Calculates the rate of change in the signal envelope for adaptive release
 * @param cd The limiter filter data containing envelope history
 * @return A value representing the average rate of envelope change
 *
 * Used by the program-dependent release system to determine how quickly the
 * audio signal's envelope is changing. Rapid changes (transients, onset of words)
 * trigger faster release times for more transparent dynamics processing.
 *
 * The algorithm averages differences between consecutive envelope history values.
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
	return change_sum / (float)(NUM_ENV_HISTORY - 1 + SMALL_EPSILON);
}

/**
 * @brief Ensures the envelope buffer is large enough for the requested samples
 * @param cd The limiter filter data
 * @param num_samples The number of samples needed in the buffer
 * @return true if the buffer is valid and sufficiently sized, false on error
 *
 * Manages dynamic resizing of the envelope buffer used to track signal levels.
 * If the buffer is too small, it will be reallocated to fit at least the
 * requested number of samples. On failure, it cleans up and returns false.
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
 * @brief Initializes or updates the circular buffers used for lookahead processing
 * @param cd The limiter filter data
 * @return true if initialization succeeded, false if it failed
 *
 * This function manages the circular buffers that allow the limiter to "look ahead"
 * in the audio stream. When enabled, it creates buffers for each channel and pre-fills
 * them with zeros to create the initial delay. If already initialized, it properly
 * cleans up existing buffers before recreating them with the new parameters.
 *
 * The lookahead feature allows the limiter to react to peaks before they occur,
 * providing cleaner limiting at the cost of added latency.
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
 * @brief Estimates inter-sample peaks using linear interpolation
 * @param current_sample The current sample value
 * @param next_sample The next sample value
 * @return The maximum absolute value found, including interpolated points
 *
 * Estimates peaks that may occur between digital samples by linearly
 * interpolating between adjacent samples at TP_OVERSAMPLE_FACTOR points.
 * This helps catch potential clipping that could occur during reconstruction
 * in D/A converters, though it is not a full implementation of the ITU-R BS.1770
 * True Peak algorithm (which uses a filter-bank approach).
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
 * @brief Calculates the signal envelope using peak detection
 * @param cd The limiter filter data
 * @param samples Array of audio channel data
 * @param num_samples Number of samples to process
 *
 * Analyzes all audio channels to find the maximum envelope at each sample.
 * Incorporates optional true peak detection and maintains historical
 * envelope values for program-dependent release calculations.
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
		if (chan >= MAX_AUDIO_CHANNELS || !samples[chan]) continue;

		float *envelope_buf = cd->envelope_buf;
		float current_env = cd->envelope; // Use clearer name

		for (uint32_t i = 0; i < num_samples; ++i) {
			float input_env; // Use clearer name
			if (cd->true_peak_enabled && i < num_samples - 1) {
				input_env = get_inter_sample_peak_estimate(samples[chan][i], samples[chan][i + 1]);
			} else {
				input_env = fabsf(samples[chan][i]);
			}
			if (!isfinite(input_env)) input_env = 0.0f;

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

			if (current_env < input_env) { // Attack
				current_env = input_env + attack_coeff * (current_env - input_env);
			} else { // Release
				current_env = input_env + current_release_coeff * (current_env - input_env);
			}
			if (!isfinite(current_env) || current_env < SMALL_EPSILON) current_env = 0.0f;

			envelope_buf[i] = fmaxf(envelope_buf[i], current_env);
		}
	}

	cd->envelope = (num_samples > 0 && cd->envelope_buf) ? cd->envelope_buf[num_samples - 1] : cd->envelope;
	if (!isfinite(cd->envelope)) cd->envelope = 0.0f;

	cd->prev_env_pos = (cd->prev_env_pos + 1) % NUM_ENV_HISTORY;
	cd->prev_env_vals[cd->prev_env_pos] = cd->envelope;
}

/**
 * @brief Applies gain reduction based on the calculated envelope
 * @param cd The limiter filter data
 * @param samples Array of audio channel data to process
 * @param num_samples Number of samples to process
 *
 * For each sample position, calculates the required gain reduction to
 * bring the signal level to the threshold (when it exceeds the threshold),
 * then applies this gain reduction to all channels at that position.
 * Also applies the final output gain.
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

		if (env_lin > SMALL_EPSILON) {
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
			if (c < MAX_AUDIO_CHANNELS && samples[c] && isfinite(samples[c][i])) {
				samples[c][i] *= final_gain;
			} else if (c < MAX_AUDIO_CHANNELS && samples[c]) {
				samples[c][i] = 0.0f;
			}
		}
	}
}


/* -------------------------------------------------------- */
/* OBS Filter API                                           */
/* -------------------------------------------------------- */

/**
 * @brief Returns the display name for this filter in the OBS interface
 * @param unused Unused parameter (required by OBS API)
 * @return The localized name of the filter
 *
 * This function provides the human-readable name shown in the OBS UI
 * when users are adding or viewing filters. The returned string is
 * localized through OBS's text translation system.
 */
static const char *limiter_v2_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return TEXT_FILTER_NAME;
}

/**
 * @brief Registers default values for all filter settings
 * @param s The settings object to fill with defaults
 *
 * This function sets initial values for all settings when the filter
 * is first created. These values provide a balanced starting point
 * that works reasonably well for most audio sources.
 *
 * Users can then adjust from these defaults or select presets for
 * more specific use cases.
 */
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

/**
 * @brief Callback that runs when a preset is selected from the dropdown
 * @param props The properties object
 * @param p The property that was modified
 * @param s The settings data
 * @return true to update all properties
 *
 * Updates all limiter parameters to match the selected preset values.
 * Each preset provides a starting point tuned for different audio material
 * and limiting goals (podcast, streaming, music, etc.).
 */
static bool preset_modified_callback(obs_properties_t *props, obs_property_t *p, obs_data_t *s)
{
	UNUSED_PARAMETER(props);
	UNUSED_PARAMETER(p);

	const char* selected_preset = obs_data_get_string(s, S_PRESET);
	blog(LOG_INFO, "[limiter v2] Preset selected: %s", selected_preset ? selected_preset : "(null / custom)");

	if (!selected_preset) return true;

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
		// Selected "Custom" - do nothing.
		return true;
	}
	return true;
}

/**
 * @brief Callback that runs when any individual parameter is modified
 * @param props The properties object
 * @param p The property that was modified
 * @param s The settings data
 * @return true to update all properties
 *
 * Automatically sets the preset selection to "Custom" whenever a user
 * manually adjusts any parameter. This provides visual feedback that
 * the current settings no longer match a predefined preset.
 */
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

/**
 * @brief Callback that runs when the lookahead enable toggle is changed
 * @param props The properties object
 * @param p The property that was modified
 * @param s The settings data
 * @return true to update all properties
 *
 * Handles two tasks:
 * 1. Sets the preset to "Custom" since a parameter was changed
 * 2. Shows/hides the lookahead time slider based on the enable toggle state
 *
 * This creates a dynamic UI where the lookahead time control is only visible
 * when lookahead processing is enabled.
 */
static bool lookahead_enabled_modified_callback(obs_properties_t *props, obs_property_t *p, obs_data_t *s)
{
	parameter_modified_callback(props, p, s); // Handle setting preset to custom

	bool enabled = obs_data_get_bool(s, S_LOOKAHEAD_ENABLED);
	obs_property_t *time_slider = obs_properties_get(props, S_LOOKAHEAD_TIME_MS);
	if (time_slider) {
		obs_property_set_visible(time_slider, enabled);
	}
	return true;
}

/**
 * @brief Creates the settings UI for the filter in the OBS interface
 * @param data The limiter filter instance data
 * @return An obs_properties_t object defining the settings UI
 *
 * This function builds the property UI shown when a user configures the filter.
 * It defines sliders, toggles, and the preset dropdown along with their
 * ranges, default values, and tooltips.
 *
 * The limiter provides a structured UI with:
 * - Preset selection for common use cases
 * - Core parameters (threshold, release time, output gain)
 * - Advanced options (lookahead, adaptive release, true peak detection)
 *
 * Dynamic UI updates are handled through callback functions that respond
 * to user interactions.
 */
static obs_properties_t *limiter_v2_properties(void *data)
{
	struct limiter_v2_data *cd = data;

	obs_properties_t *props = obs_properties_create();
	if (!props) {
		blog(LOG_ERROR, "[limiter v2] Failed to create properties");
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
		blog(LOG_WARNING, "[limiter v2] Failed to create preset list property");
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
	obs_property_t* lookahead_toggle = p; // Save pointer for visibility logic
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

	if (!lookahead_toggle) {
		blog(LOG_WARNING, "[limiter v2] Could not get lookahead toggle property for dynamic UI setup");
	}

	return props;
}


/**
 * @brief Updates limiter parameters and state when settings change
 * @param data The limiter filter instance data
 * @param s The updated settings
 *
 * This function is called when:
 * - The filter is created
 * - User changes settings in the UI
 * - Audio system parameters (sample rate, channel count) change
 *
 * It updates all internal parameters, recalculates coefficients,
 * manages lookahead buffer state, and updates the audio latency
 * reporting to ensure proper A/V sync when lookahead is enabled.
 */
static void limiter_v2_update(void *data, obs_data_t *s)
{
	struct limiter_v2_data *cd = data;
	if (!cd) return;

	const uint32_t sample_rate = audio_output_get_sample_rate(obs_get_audio());
	size_t num_channels = audio_output_get_channels(obs_get_audio());

	if (num_channels > MAX_AUDIO_CHANNELS) {
		warn("Limiter V2 supports up to %d channels, clamped to %d.", MAX_AUDIO_CHANNELS, MAX_AUDIO_CHANNELS);
		num_channels = MAX_AUDIO_CHANNELS;
	}
	uint32_t current_sample_rate = sample_rate;
	if (num_channels == 0 || current_sample_rate == 0) {
		warn("Invalid audio parameters (Channels: %zu, Sample Rate: %u). Limiter fallback to default SR if unavailable.", num_channels, current_sample_rate);
		if (current_sample_rate == 0) {
			current_sample_rate = 48000;
			warn("Sample rate was 0, fallback to default %uHz.", current_sample_rate);
		}
	}

	bool reset_lookahead = false;
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

	cd->threshold_db = (float)obs_data_get_double(s, S_FILTER_THRESHOLD);
	cd->release_time_ms = (float)obs_data_get_double(s, S_RELEASE_TIME);
	cd->output_gain_db = (float)obs_data_get_double(s, S_OUTPUT_GAIN);
	cd->adaptive_release_enabled = obs_data_get_bool(s, S_ADAPTIVE_RELEASE_ENABLED);
	cd->true_peak_enabled = obs_data_get_bool(s, S_TRUE_PEAK_ENABLED);

	bool new_lookahead_enabled = obs_data_get_bool(s, S_LOOKAHEAD_ENABLED);
	float new_lookahead_time_ms = (float)obs_data_get_double(s, S_LOOKAHEAD_TIME_MS);

	new_lookahead_time_ms = fmaxf(0.0f, fminf(MAX_LOOKAHEAD_MS, new_lookahead_time_ms));
	if (new_lookahead_enabled && new_lookahead_time_ms < MIN_LOOKAHEAD_MS) {
		 new_lookahead_time_ms = MIN_LOOKAHEAD_MS;
	} else if (!new_lookahead_enabled) {
		new_lookahead_time_ms = 0.0f;
	}

	if (new_lookahead_enabled != cd->lookahead_enabled ||
	    new_lookahead_time_ms != cd->lookahead_time_ms) {
		reset_lookahead = true;
		cd->lookahead_enabled = new_lookahead_enabled;
		cd->lookahead_time_ms = new_lookahead_time_ms;
	}

	cd->attack_coeff = gain_coefficient(cd->sample_rate, FIXED_ATTACK_TIME_MS);
	cd->release_time_ms = fmaxf(MIN_RELEASE_MS, fminf(MAX_RELEASE_MS, cd->release_time_ms));
	cd->release_coeff = gain_coefficient(cd->sample_rate, cd->release_time_ms);
	cd->output_gain = db_to_mul(cd->output_gain_db);

	if (reset_lookahead) {
		size_t new_lookahead_samples = 0;
		if (cd->lookahead_enabled && cd->lookahead_time_ms >= MIN_LOOKAHEAD_MS && cd->sample_rate > 0 && cd->num_channels > 0) {
			new_lookahead_samples = (size_t)((cd->sample_rate * cd->lookahead_time_ms) / MS_IN_S_F + 0.5f);
			if (new_lookahead_samples == 0) new_lookahead_samples = 1;
		}
		cd->lookahead_samples = new_lookahead_samples;

		if (!update_lookahead_buffers(cd)) {
			warn("Disabling lookahead due to buffer allocation failure.");
			cd->lookahead_enabled = false;
			cd->lookahead_samples = 0;
		}
	}

	uint64_t latency_ns = 0;
	if (cd->lookahead_enabled && cd->lookahead_buffers_initialized && cd->lookahead_samples > 0 && cd->sample_rate > 0) {
		latency_ns = (uint64_t)(((double)cd->lookahead_samples / cd->sample_rate) * 1.0e9);
	}
	obs_source_set_audio_latency(cd->context, latency_ns);

	if (!cd->envelope_buf) {
		size_t initial_env_len = (cd->sample_rate * INITIAL_ENV_BUF_MS) / MS_IN_S;
		if (initial_env_len == 0) initial_env_len = 1024;
		ensure_env_buffer(cd, initial_env_len);
		if (cd->envelope_buf) {
             memset(cd->envelope_buf, 0, cd->envelope_buf_len * sizeof(float));
        }
	}
}

/**
 * @brief Creates and initializes a new limiter filter instance
 * @param settings The initial settings for the filter
 * @param filter The OBS source object representing this filter
 * @return Pointer to the newly created filter data, or NULL on failure
 *
 * This function is called by OBS when the filter is added to a source.
 * It allocates the data structure, initializes all values to safe defaults,
 * and calls limiter_v2_update to configure the filter with the initial settings.
 */
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
	cd->envelope_buf = NULL;
	cd->envelope_buf_len = 0;

	limiter_v2_defaults(settings);
	limiter_v2_update(cd, settings);
	info("Limiter v2 filter created");
	return cd;
}

/**
 * @brief Cleans up and frees resources when the filter is removed
 * @param data The filter instance data to clean up
 *
 * This function is called by OBS when the filter is removed from a source
 * or when the source is being destroyed. It releases all allocated buffers
 * and frees the filter data structure.
 */
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
	// cd = NULL; // Optional defensive programming
}


/**
 * @brief The main audio processing callback for the limiter
 * @param data The filter instance data
 * @param audio The audio buffer to process
 * @return The processed audio buffer
 *
 * This function implements the three-stage limiter processing:
 * 1. Envelope detection with optional true peak analysis
 * 2. Lookahead delay (if enabled)
 * 3. Gain reduction based on envelope vs threshold
 *
 * Audio is processed in-place and the same buffer is returned.
 */
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

	// --- 1. Analyze Undelayed Input (Sidechain Path) ---
	analyze_envelope(cd, samples, num_samples);

	// --- 2. Apply Lookahead Delay (Main Path Delay) ---
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

	// --- 3. Process Compression (Main Path Processing) ---
	process_compression(cd, samples, num_samples);

	return audio;
}


/* -------------------------------------------------------- */
/* Filter Definition                                        */
/* -------------------------------------------------------- */

/**
 * @brief Filter definition structure for registering with OBS
 *
 * This structure defines the entry points and metadata for the limiter filter:
 * - Identifies the filter with a unique ID
 * - Specifies it as an audio filter type
 * - Maps OBS filter lifecycle events to our handler functions
 *
 * When OBS loads the plugin, this structure is registered via the
 * external "limiter_v2_filter" symbol in obs-filters.c, making the
 * filter available in the OBS user interface.
 */
struct obs_source_info limiter_v2_filter = {
	.id             = "limiter_v2_filter", // Unique ID
	.type           = OBS_SOURCE_TYPE_FILTER,
	.output_flags   = OBS_SOURCE_AUDIO, // Outputs audio
	.get_name       = limiter_v2_name,
	.create         = limiter_v2_create,
	.destroy        = limiter_v2_destroy,
	.update         = limiter_v2_update,
	.filter_audio   = limiter_v2_filter_audio,
	.get_defaults   = limiter_v2_defaults,
	.get_properties = limiter_v2_properties,
};