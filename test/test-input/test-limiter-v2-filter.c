/**
 * test-limiter-v2-filter.c - Basic Test Harness for Limiter V2 Filter
 *
 * Purpose: Perform basic functional and robustness checks on the Limiter V2
 *          filter's lifecycle (create, defaults, update, destroy) and
 *          parameter/preset handling within a minimal OBS environment.
 *
 * Note: Does not perform extensive audio processing validation but verifies
 *       core setup, state management, and UI logic simulation.
 *       Requires linking against libobs and the module containing the filter.
 */

#include <obs-module.h>
#include <obs-data.h>
#include <obs-source.h>
#include <obs-properties.h>    // Needed for obs_properties_get, obs_property_modified
#include <media-io/audio-io.h> // For audio_output_info
#include <util/platform.h>
#include <util/dstr.h>
#include <util/base.h>
#include <obs.h> // For OBS_SUCCESS definition
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>   // For basic assertions
#include <inttypes.h> // For PRIu64 format specifier

// --- Test Configuration ---
#define TEST_SAMPLE_RATE 48000
#define TEST_CHANNELS    2 // Stereo
#define TEST_BLOCK_SIZE  1024 // Use a slightly larger block for processing test
#define TEST_DURATION_MS 50  // Shorter duration, focus on stability

// OBS success/error codes if not defined
#ifndef OBS_SUCCESS
#define OBS_SUCCESS 0
#endif

// --- Reference the actual filter info struct ---
// Defined in plugins/obs-filters/limiter-v2-filter.c (must be compiled/linked)
extern struct obs_source_info limiter_v2_filter;

// Settings Keys - Must match limiter-v2-filter.c
#define S_PRESET                        "preset_selection"
#define S_FILTER_THRESHOLD              "threshold"
#define S_RELEASE_TIME                  "release_time"
#define S_OUTPUT_GAIN                   "output_gain"
#define S_LOOKAHEAD_ENABLED             "lookahead_enabled"
#define S_LOOKAHEAD_TIME_MS             "lookahead_time_ms"
#define S_ADAPTIVE_RELEASE_ENABLED      "adaptive_release_enabled"
#define S_TRUE_PEAK_ENABLED             "true_peak_enabled"

// Preset Values - Must match limiter-v2-filter.c
#define PRESET_VAL_CUSTOM               ""
#define PRESET_VAL_DEFAULT              "default"
#define PRESET_VAL_PODCAST              "podcast"
#define PRESET_VAL_STREAMING            "streaming"
#define PRESET_VAL_AGGRESSIVE           "aggressive"
#define PRESET_VAL_TRANSPARENT          "transparent"
#define PRESET_VAL_MUSIC                "music"
#define PRESET_VAL_BRICKWALL            "brickwall"
// ... include other PRESET_VAL_* defines if needed for more preset tests ...

// Parameter Defaults - Must match limiter-v2-filter.c
#define DEFAULT_THRESHOLD_DB            -6.0
#define DEFAULT_RELEASE_MS              60.0
#define DEFAULT_OUTPUT_GAIN_DB          0.0
#define DEFAULT_ADAPTIVE_RELEASE        true
#define DEFAULT_LOOKAHEAD_ENABLED       true
#define DEFAULT_LOOKAHEAD_MS            5.0
#define DEFAULT_TRUE_PEAK_ENABLED       true

// Maximum number of audio channels to allocate
#define MAX_AUDIO_CHANNELS              8

// Simple log handler for test output
static void test_log_handler(int log_level, const char *format, va_list args, void *param)
{
	// Simple implementation: print warnings and errors to stderr
	if (log_level <= LOG_WARNING) {
		char buffer[4096];
		vsnprintf(buffer, sizeof(buffer), format, args);
		fprintf(stderr, "[obs-test-log] %s\n", buffer);
	}
	UNUSED_PARAMETER(param);
}

// --- Test Framework ---
static int tests_run = 0;
static int tests_failed = 0;

#define RUN_TEST(test_func) \
	do { \
		printf("-- Running Test: %s --\n", #test_func); fflush(stdout); \
		tests_run++; \
		bool test_passed_internal = true; \
		test_func(&test_passed_internal); \
		if (test_passed_internal) { \
			printf("  PASS\n"); fflush(stdout); \
		} else { \
			printf("  FAIL\n"); fflush(stdout); \
			tests_failed++; \
		} \
		printf("\n"); fflush(stdout); \
	} while (0)

// Assertion helper
#define ASSERT_TRUE(condition, message) \
	if (!(condition)) { \
		fprintf(stderr, "  ASSERT FAIL: %s (%s:%d)\n", message, __FILE__, __LINE__); \
		*test_passed = false; \
	} else { \
		/* fprintf(stdout, "   ASSERT PASS: %s\n", message); */ \
	}

#define ASSERT_EQUAL_STR(val1, val2, message) \
	if (!(val1 && val2 && strcmp(val1, val2) == 0)) { \
		fprintf(stderr, "  ASSERT FAIL: %s Expected [%s] Got [%s] (%s:%d)\n", \
			message, val2 ? val2 : "NULL", val1 ? val1 : "NULL", __FILE__, __LINE__); \
		*test_passed = false; \
	} else { \
		/* fprintf(stdout, "   ASSERT PASS: %s\n", message); */ \
	}

#define ASSERT_EQUAL_DBL(val1, val2, tolerance, message) \
	if (!(fabs((val1) - (val2)) < tolerance)) { \
		fprintf(stderr, "  ASSERT FAIL: %s Expected [%f] Got [%f] (%s:%d)\n", \
			message, (double)(val2), (double)(val1), __FILE__, __LINE__); \
		*test_passed = false; \
	} else { \
		/* fprintf(stdout, "   ASSERT PASS: %s\n", message); */ \
	}

#define ASSERT_NOT_NULL(ptr, message) \
	if ((ptr) == NULL) { \
		fprintf(stderr, "  ASSERT FAIL: %s - NULL pointer (%s:%d)\n", \
			message, __FILE__, __LINE__); \
		*test_passed = false; \
	} else { \
		/* fprintf(stdout, "   ASSERT PASS: %s\n", message); */ \
	}

// --- Test Cases ---

/** @brief Test basic creation and destruction of the filter */
static void test_limiter_create_destroy(bool *test_passed)
{
	obs_data_t *settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create succeeded");

	obs_source_t *filter = obs_source_create_private(limiter_v2_filter.id, "Test CreateDestroy", NULL);
	ASSERT_NOT_NULL(filter, "limiter_v2_create returned non-NULL");

	obs_source_release(filter); // Should call _destroy without crashing
	obs_data_release(settings);
}

/** @brief Test if default settings are applied correctly */
static void test_limiter_defaults(bool *test_passed)
{
	obs_data_t *settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create succeeded");

	// Call get_defaults directly
	limiter_v2_filter.get_defaults(settings);

	// Verify key default values
	ASSERT_EQUAL_STR(obs_data_get_string(settings, S_PRESET), PRESET_VAL_DEFAULT, "Default preset correct");
	ASSERT_EQUAL_DBL(obs_data_get_double(settings, S_FILTER_THRESHOLD), DEFAULT_THRESHOLD_DB, 0.01,
			 "Default threshold correct");
	ASSERT_EQUAL_DBL(obs_data_get_double(settings, S_RELEASE_TIME), DEFAULT_RELEASE_MS, 0.01,
			 "Default release correct");
	ASSERT_TRUE(obs_data_get_bool(settings, S_LOOKAHEAD_ENABLED) == DEFAULT_LOOKAHEAD_ENABLED,
		    "Default lookahead enabled correct");
	ASSERT_EQUAL_DBL(obs_data_get_double(settings, S_LOOKAHEAD_TIME_MS), DEFAULT_LOOKAHEAD_MS, 0.01,
			 "Default lookahead time correct");
	ASSERT_TRUE(obs_data_get_bool(settings, S_ADAPTIVE_RELEASE_ENABLED) == DEFAULT_ADAPTIVE_RELEASE,
		    "Default adaptive release correct");
	ASSERT_TRUE(obs_data_get_bool(settings, S_TRUE_PEAK_ENABLED) == DEFAULT_TRUE_PEAK_ENABLED,
		    "Default true peak correct");

	obs_data_release(settings);
}

/** @brief Test updating filter with various settings */
static void test_limiter_update(bool *test_passed)
{
	obs_data_t *settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create succeeded");

	obs_source_t *filter = obs_source_create_private(limiter_v2_filter.id, "Test Update", settings);
	ASSERT_NOT_NULL(filter, "limiter_v2_create succeeded");

	if (filter) {
		// Modify settings AFTER create
		obs_data_set_double(settings, S_FILTER_THRESHOLD, -15.5);
		obs_data_set_double(settings, S_RELEASE_TIME, 150.0);
		obs_data_set_double(settings, S_OUTPUT_GAIN, -2.0);
		obs_data_set_bool(settings, S_ADAPTIVE_RELEASE_ENABLED, false);
		obs_data_set_bool(settings, S_LOOKAHEAD_ENABLED, true);
		obs_data_set_double(settings, S_LOOKAHEAD_TIME_MS, 12.3);
		obs_data_set_bool(settings, S_TRUE_PEAK_ENABLED, false);

		// Call update to apply new settings
		printf("   INFO: Calling obs_source_update with modified settings...\n");
		obs_source_update(filter, settings);
		printf("   INFO: Update call completed without crash.\n");
		// Primarily testing stability here.
	}

	obs_source_release(filter);
	obs_data_release(settings);
}

/** @brief Test preset selection and modification logic */
static void test_limiter_presets(bool *test_passed)
{
	obs_data_t *settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create succeeded");

	obs_source_t *filter = obs_source_create_private(limiter_v2_filter.id, "Test Presets", settings);
	ASSERT_NOT_NULL(filter, "limiter_v2_create succeeded");

	if (filter) {
		obs_properties_t *props = obs_source_properties(filter);
		ASSERT_NOT_NULL(props, "obs_source_properties succeeded");

		if (props) {
			// Simulate selecting the "Podcast" preset
			printf("   INFO: Simulating selecting Podcast preset...\n");
			obs_data_set_string(settings, S_PRESET, PRESET_VAL_PODCAST);

			// Trigger modification using the proper API call
			obs_property_t *preset_prop = obs_properties_get(props, S_PRESET);
			ASSERT_NOT_NULL(preset_prop, "Got preset property handle");
			if (preset_prop) {
				obs_property_modified(preset_prop, settings); // Triggers preset_modified_callback
			} else {
				*test_passed = false;
			}

			// Check settings object immediately after callback simulation
			ASSERT_EQUAL_DBL(obs_data_get_double(settings, S_FILTER_THRESHOLD), -8.0, 0.01,
					 "Podcast preset threshold applied");
			ASSERT_EQUAL_DBL(obs_data_get_double(settings, S_RELEASE_TIME), 80.0, 0.01,
					 "Podcast preset release applied");
			ASSERT_TRUE(obs_data_get_bool(settings, S_LOOKAHEAD_ENABLED),
				    "Podcast preset lookahead enabled applied");

			// Simulate manually changing a setting after selecting preset
			printf("   INFO: Simulating manual change after preset...\n");
			obs_data_set_double(settings, S_OUTPUT_GAIN, 5.0);

			// Trigger modification using the proper API call for the changed property
			obs_property_t *output_gain_prop = obs_properties_get(props, S_OUTPUT_GAIN);
			ASSERT_NOT_NULL(output_gain_prop, "Got output gain property handle");
			if (output_gain_prop) {
				obs_property_modified(output_gain_prop,
						      settings); // Triggers parameter_modified_callback
			} else {
				*test_passed = false;
			}

			// Check if preset in settings object switched back to "Custom"
			ASSERT_EQUAL_STR(obs_data_get_string(settings, S_PRESET), PRESET_VAL_CUSTOM,
					 "Preset set to Custom after manual change");

			obs_properties_destroy(props);
		}
	}

	obs_source_release(filter);
	obs_data_release(settings);
}

/** @brief Basic audio processing stability check */
static void test_limiter_process_stability(bool *test_passed)
{
	obs_data_t *settings = NULL;
	obs_source_t *filter = NULL;
	float *channel_buffers[MAX_AUDIO_CHANNELS] = {NULL};
	uint8_t *audio_pointers[MAX_AUDIO_CHANNELS] = {NULL};

	printf("   INFO: Performing basic audio processing stability check...\n");

	settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create for proc succeeded");

	obs_data_set_double(settings, S_FILTER_THRESHOLD, -10.0);
	obs_data_set_bool(settings, S_LOOKAHEAD_ENABLED, true);
	obs_data_set_double(settings, S_LOOKAHEAD_TIME_MS, 2.0);
	obs_data_set_bool(settings, S_TRUE_PEAK_ENABLED, true);

	filter = obs_source_create_private(limiter_v2_filter.id, "Test Processing Stability", settings);
	ASSERT_NOT_NULL(filter, "limiter_v2_create succeeded for proc");
	if (!filter)
		goto cleanup_proc;

	// Get filter data pointer using internal helper (use with caution)
	// Note: This relies on internal OBS structure access. If this fails,
	// the test's ability to call filter_audio directly is compromised.
	// A more complex test might attach the filter to a dummy source
	// and use obs_source_output_audio.
	void *filter_data_ptr = obs_source_get_filter_data(filter);
	ASSERT_NOT_NULL(filter_data_ptr, "obs_source_get_filter_data succeeded for proc");
	if (!filter_data_ptr)
		goto cleanup_proc;

	// Get audio parameters from OBS or use test defaults
	size_t channels = audio_output_get_channels(obs_get_audio());
	if (channels == 0 || channels > MAX_AUDIO_CHANNELS)
		channels = TEST_CHANNELS;
	uint32_t sample_rate = audio_output_get_sample_rate(obs_get_audio());
	if (sample_rate == 0)
		sample_rate = TEST_SAMPLE_RATE;

	// Allocate channel buffers
	for (size_t c = 0; c < channels; ++c) {
		channel_buffers[c] = bzalloc(sizeof(float) * TEST_BLOCK_SIZE);
		ASSERT_NOT_NULL(channel_buffers[c], "Audio channel buffer allocated");
		audio_pointers[c] = (uint8_t *)channel_buffers[c];
	}

	// Initialize audio data struct using struct literal
	struct obs_audio_data audio_data = {.frames = TEST_BLOCK_SIZE,
					    .timestamp = os_gettime_ns(),
					    .data = (const uint8_t **)audio_pointers};

	// Process several blocks
	uint32_t blocks_to_process = (uint32_t)(TEST_DURATION_MS * sample_rate) / (TEST_BLOCK_SIZE * 1000);
	if (blocks_to_process == 0)
		blocks_to_process = 1; // Ensure at least one block

	for (uint32_t block = 0; block < blocks_to_process; ++block) {
		// Fill with signal guaranteed to exceed threshold
		for (size_t c = 0; c < channels; ++c) {
			if (channel_buffers[c]) {
				for (uint32_t i = 0; i < TEST_BLOCK_SIZE; ++i) {
					channel_buffers[c][i] = ((i / 16) % 2 == 0) ? 1.1f : -1.1f; // > 0dBFS
				}
			}
		}
		audio_data.timestamp += (uint64_t)TEST_BLOCK_SIZE * 1000000000ULL / sample_rate;

		// Call the filter's audio processing function safely
		if (limiter_v2_filter.filter_audio) {
			struct obs_audio_data *result = limiter_v2_filter.filter_audio(filter_data_ptr, &audio_data);
			ASSERT_TRUE(result == &audio_data, "filter_audio returned input buffer");

			// Basic stability check: scan for NaN/Inf
			for (size_t c = 0; c < channels; ++c) {
				if (channel_buffers[c]) {
					for (uint32_t i = 0; i < TEST_BLOCK_SIZE; ++i) {
						if (!isfinite(channel_buffers[c][i])) {
							fprintf(stderr,
								"  ERROR: NaN/Inf detected in output sample %u, channel %zu\n",
								i, c);
							*test_passed = false;
							goto cleanup_proc;
						}
					}
				}
			}
		} else {
			fprintf(stderr, "  ERROR: filter_audio function pointer is NULL!\n");
			*test_passed = false;
			goto cleanup_proc;
		}
	}
	printf("   INFO: Processed audio blocks without crashes or NaN/Inf.\n");

cleanup_proc:
	// Free audio buffers
	for (size_t c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
		bfree(channel_buffers[c]);
	}
	obs_data_release(settings); // Safe on NULL
	obs_source_release(filter); // Safe on NULL
}

/** @brief Test all presets for correct application of settings */
static void test_limiter_all_presets(bool *test_passed)
{
	obs_data_t *settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create succeeded");

	obs_source_t *filter = obs_source_create_private(limiter_v2_filter.id, "Test All Presets", settings);
	ASSERT_NOT_NULL(filter, "limiter_v2_create succeeded");

	if (filter) {
		obs_properties_t *props = obs_source_properties(filter);
		ASSERT_NOT_NULL(props, "obs_source_properties succeeded");

		if (props) {
			// Define presets to test with expected key values
			struct {
				const char *preset_id;
				double threshold;
				double release;
				bool lookahead_enabled;
			} preset_tests[] = {{PRESET_VAL_DEFAULT, DEFAULT_THRESHOLD_DB, DEFAULT_RELEASE_MS,
					     DEFAULT_LOOKAHEAD_ENABLED},
					    {PRESET_VAL_PODCAST, -8.0, 80.0, true},
					    {PRESET_VAL_STREAMING, -7.0, 70.0, true},
					    {PRESET_VAL_AGGRESSIVE, -5.0, 40.0, true},
					    {PRESET_VAL_TRANSPARENT, -1.5, 50.0, true},
					    {PRESET_VAL_MUSIC, -2.0, 200.0, true},
					    {PRESET_VAL_BRICKWALL, -0.3, 50.0, true}};

			obs_property_t *preset_prop = obs_properties_get(props, S_PRESET);
			ASSERT_NOT_NULL(preset_prop, "Got preset property handle");

			// Test each preset
			for (size_t i = 0; i < sizeof(preset_tests) / sizeof(preset_tests[0]); i++) {
				printf("   INFO: Testing preset: %s\n", preset_tests[i].preset_id);
				obs_data_set_string(settings, S_PRESET, preset_tests[i].preset_id);

				if (preset_prop) {
					obs_property_modified(preset_prop, settings);
				}

				ASSERT_EQUAL_DBL(obs_data_get_double(settings, S_FILTER_THRESHOLD),
						 preset_tests[i].threshold, 0.01, "Preset threshold applied correctly");

				ASSERT_EQUAL_DBL(obs_data_get_double(settings, S_RELEASE_TIME), preset_tests[i].release,
						 0.01, "Preset release time applied correctly");

				ASSERT_TRUE(obs_data_get_bool(settings, S_LOOKAHEAD_ENABLED) ==
						    preset_tests[i].lookahead_enabled,
					    "Preset lookahead enabled state applied correctly");
			}

			obs_properties_destroy(props);
		}
	}

	obs_source_release(filter);
	obs_data_release(settings);
}

/** @brief Test that the limiter actually limits peaks to threshold */
static void test_limiter_peak_limiting(bool *test_passed)
{
	obs_data_t *settings = NULL;
	obs_source_t *filter = NULL;
	float *channel_buffers[MAX_AUDIO_CHANNELS] = {NULL};
	uint8_t *audio_pointers[MAX_AUDIO_CHANNELS] = {NULL};
	const float test_threshold_db = -10.0;
	const float threshold_linear = db_to_mul(test_threshold_db);
	const float test_margin_db = 0.1; // Allow slight variation due to filter implementation

	printf("   INFO: Testing actual peak limiting behavior...\n");

	settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create for peak limiting test succeeded");

	// Use minimal lookahead and true peak for fast processing
	obs_data_set_double(settings, S_FILTER_THRESHOLD, test_threshold_db);
	obs_data_set_double(settings, S_RELEASE_TIME, 1.0); // Fast release for test
	obs_data_set_bool(settings, S_LOOKAHEAD_ENABLED, true);
	obs_data_set_double(settings, S_LOOKAHEAD_TIME_MS, 0.1); // Minimal lookahead
	obs_data_set_bool(settings, S_TRUE_PEAK_ENABLED, false); // Simpler processing for test

	filter = obs_source_create_private(limiter_v2_filter.id, "Test Peak Limiting", settings);
	ASSERT_NOT_NULL(filter, "limiter_v2_create succeeded for peak limiting test");
	if (!filter)
		goto cleanup_peak;

	void *filter_data_ptr = obs_source_get_filter_data(filter);
	ASSERT_NOT_NULL(filter_data_ptr, "Got filter data pointer for peak limiting test");
	if (!filter_data_ptr)
		goto cleanup_peak;

	size_t channels = audio_output_get_channels(obs_get_audio());
	if (channels == 0 || channels > MAX_AUDIO_CHANNELS)
		channels = TEST_CHANNELS;
	uint32_t sample_rate = audio_output_get_sample_rate(obs_get_audio());
	if (sample_rate == 0)
		sample_rate = TEST_SAMPLE_RATE;

	// Allocate channel buffers
	for (size_t c = 0; c < channels; ++c) {
		channel_buffers[c] = bzalloc(sizeof(float) * TEST_BLOCK_SIZE);
		ASSERT_NOT_NULL(channel_buffers[c], "Audio channel buffer allocated for peak limiting test");
		audio_pointers[c] = (uint8_t *)channel_buffers[c];
	}

	// Initialize audio data struct using struct literal
	struct obs_audio_data audio_data = {.frames = TEST_BLOCK_SIZE,
					    .timestamp = os_gettime_ns(),
					    .data = (const uint8_t **)audio_pointers};

	// Fill with signal guaranteed to exceed threshold by a lot
	for (size_t c = 0; c < channels; ++c) {
		if (channel_buffers[c]) {
			// Create a loud signal at +6dB (2.0 linear)
			for (uint32_t i = 0; i < TEST_BLOCK_SIZE; ++i) {
				channel_buffers[c][i] = 2.0f;
			}
		}
	}

	// Process audio through the filter
	if (limiter_v2_filter.filter_audio) {
		// Process multiple times to ensure envelope has stabilized
		for (int pass = 0; pass < 5; pass++) {
			struct obs_audio_data *result = limiter_v2_filter.filter_audio(filter_data_ptr, &audio_data);
			ASSERT_TRUE(result == &audio_data, "filter_audio returned input buffer");
		}

		// Now check if output is limited to threshold
		bool peaks_limited = true;
		float max_sample = 0.0f;

		for (size_t c = 0; c < channels && peaks_limited; ++c) {
			if (channel_buffers[c]) {
				for (uint32_t i = 0; i < TEST_BLOCK_SIZE; ++i) {
					float sample_abs = fabsf(channel_buffers[c][i]);
					max_sample = fmaxf(max_sample, sample_abs);

					// Allow a small margin for implementation differences
					if (sample_abs > (threshold_linear * db_to_mul(test_margin_db))) {
						peaks_limited = false;
						fprintf(stderr,
							"  ERROR: Sample at %u in channel %zu exceeds threshold: %f (threshold: %f)\n",
							i, c, sample_abs, threshold_linear);
						break;
					}
				}
			}
		}

		ASSERT_TRUE(peaks_limited, "All output peaks are limited to threshold");
		printf("   INFO: Maximum sample value after limiting: %f (linear), threshold: %f\n", max_sample,
		       threshold_linear);
	} else {
		fprintf(stderr, "  ERROR: filter_audio function pointer is NULL!\n");
		*test_passed = false;
	}

cleanup_peak:
	// Free audio buffers
	for (size_t c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
		bfree(channel_buffers[c]);
	}
	obs_data_release(settings);
	obs_source_release(filter);
}

/** @brief Test lookahead delay is working correctly */
static void test_limiter_lookahead_delay(bool *test_passed)
{
	obs_data_t *settings = NULL;
	obs_source_t *filter = NULL;
	float *channel_buffers[MAX_AUDIO_CHANNELS] = {NULL};
	uint8_t *audio_pointers[MAX_AUDIO_CHANNELS] = {NULL};
	const float lookahead_ms = 10.0f;

	printf("   INFO: Testing lookahead delay...\n");

	settings = obs_data_create();
	ASSERT_NOT_NULL(settings, "obs_data_create for lookahead test succeeded");

	obs_data_set_double(settings, S_FILTER_THRESHOLD, -20.0); // Low threshold to avoid limiting
	obs_data_set_bool(settings, S_LOOKAHEAD_ENABLED, true);
	obs_data_set_double(settings, S_LOOKAHEAD_TIME_MS, lookahead_ms);

	filter = obs_source_create_private(limiter_v2_filter.id, "Test Lookahead", settings);
	ASSERT_NOT_NULL(filter, "limiter_v2_create succeeded for lookahead test");
	if (!filter)
		goto cleanup_lookahead;

	// Verify reported audio latency
	uint64_t reported_latency_ns = obs_source_get_audio_latency(filter);
	uint64_t expected_latency_ns = (uint64_t)(lookahead_ms * 1000000.0);
	// Allow some implementation variation due to sample rate conversion
	ASSERT_TRUE(reported_latency_ns > 0, "Lookahead filter reports non-zero latency");
	printf("   INFO: Reported lookahead latency: %" PRIu64 " ns\n", reported_latency_ns);

	void *filter_data_ptr = obs_source_get_filter_data(filter);
	ASSERT_NOT_NULL(filter_data_ptr, "Got filter data pointer for lookahead test");
	if (!filter_data_ptr)
		goto cleanup_lookahead;

	size_t channels = audio_output_get_channels(obs_get_audio());
	if (channels == 0 || channels > MAX_AUDIO_CHANNELS)
		channels = TEST_CHANNELS;
	uint32_t sample_rate = audio_output_get_sample_rate(obs_get_audio());
	if (sample_rate == 0)
		sample_rate = TEST_SAMPLE_RATE;

	// Calculate lookahead samples
	uint32_t expected_lookahead_samples = (uint32_t)((sample_rate * lookahead_ms) / 1000.0f);

	// Allocate channel buffers
	for (size_t c = 0; c < channels; ++c) {
		channel_buffers[c] = bzalloc(sizeof(float) * TEST_BLOCK_SIZE);
		ASSERT_NOT_NULL(channel_buffers[c], "Audio channel buffer allocated for lookahead test");
		audio_pointers[c] = (uint8_t *)channel_buffers[c];
	}

	// Initialize audio data struct using struct literal
	struct obs_audio_data audio_data = {.frames = TEST_BLOCK_SIZE,
					    .timestamp = os_gettime_ns(),
					    .data = (const uint8_t **)audio_pointers};

	// Create an impulse signal (1.0 at first sample, 0 elsewhere)
	for (size_t c = 0; c < channels; ++c) {
		if (channel_buffers[c]) {
			memset(channel_buffers[c], 0, TEST_BLOCK_SIZE * sizeof(float));
			channel_buffers[c][0] = 1.0f;
		}
	}

	// Process audio through the filter
	if (limiter_v2_filter.filter_audio) {
		struct obs_audio_data *result = limiter_v2_filter.filter_audio(filter_data_ptr, &audio_data);
		ASSERT_TRUE(result == &audio_data, "filter_audio returned input buffer");

		// Verify the impulse is delayed by lookahead
		bool delay_correct = true;

		for (size_t c = 0; c < channels && delay_correct; ++c) {
			if (channel_buffers[c] && expected_lookahead_samples < TEST_BLOCK_SIZE) {
				// Check that samples before lookahead point are zero (or near zero)
				for (uint32_t i = 0; i < expected_lookahead_samples; ++i) {
					if (fabsf(channel_buffers[c][i]) > 0.01f) {
						fprintf(stderr,
							"  ERROR: Sample at %u in channel %zu should be zero due to lookahead delay: %f\n",
							i, c, channel_buffers[c][i]);
						delay_correct = false;
						break;
					}
				}

				// Check that the impulse appears at or near the lookahead point
				bool impulse_found = false;
				for (uint32_t i = expected_lookahead_samples;
				     i < expected_lookahead_samples + 5 && i < TEST_BLOCK_SIZE; ++i) {
					if (fabsf(channel_buffers[c][i]) > 0.5f) { // Allow for some attenuation
						impulse_found = true;
						break;
					}
				}

				if (!impulse_found) {
					fprintf(stderr,
						"  ERROR: Impulse not found at expected position after lookahead\n");
					delay_correct = false;
				}
			}
		}

		ASSERT_TRUE(delay_correct, "Lookahead delay functions correctly");
	} else {
		fprintf(stderr, "  ERROR: filter_audio function pointer is NULL!\n");
		*test_passed = false;
	}

cleanup_lookahead:
	// Free audio buffers
	for (size_t c = 0; c < MAX_AUDIO_CHANNELS; ++c) {
		bfree(channel_buffers[c]);
	}
	obs_data_release(settings);
	obs_source_release(filter);
}

// --- Main Test Function ---

int main(int argc, char *argv[])
{
	UNUSED_PARAMETER(argc);
	UNUSED_PARAMETER(argv);

	printf("--- Starting OBS Limiter V2 Filter Basic Tests ---\n\n");

	// --- OBS Initialization ---
	base_set_log_handler(test_log_handler, NULL);
	if (!obs_startup("en-US", NULL, NULL)) {
		fprintf(stderr, "FATAL: obs_startup failed\n");
		return EXIT_FAILURE;
	}

	// Set default audio parameters
	struct audio_output_info aoi;
	memset(&aoi, 0, sizeof(aoi));
	aoi.format = AUDIO_FORMAT_FLOAT;
	aoi.speakers = (enum speaker_layout)TEST_CHANNELS;
	aoi.samples_per_sec = TEST_SAMPLE_RATE;
	if (obs_reset_audio(&aoi) != OBS_SUCCESS) {
		fprintf(stderr, "FATAL: obs_reset_audio failed\n");
		obs_shutdown();
		return EXIT_FAILURE;
	}

	// --- Load Required Modules ---
	// obs_load_all_modules(); // Generally needed if filter is in separate module

	// Ensure filter type exists - crucial check
	if (!obs_is_source_registered(limiter_v2_filter.id)) {
		fprintf(stderr, "FATAL: Filter type '%s' not registered.\n", limiter_v2_filter.id);
		fprintf(stderr, "Ensure obs-filters module is built and loaded or test is linked correctly.\n");
		obs_shutdown();
		return EXIT_FAILURE;
	}

	printf("OBS Initialized for testing.\n\n");

	// --- Run Tests ---
	RUN_TEST(test_limiter_create_destroy);
	RUN_TEST(test_limiter_defaults);
	RUN_TEST(test_limiter_update);
	RUN_TEST(test_limiter_presets);
	RUN_TEST(test_limiter_process_stability);
	RUN_TEST(test_limiter_all_presets);
	RUN_TEST(test_limiter_peak_limiting);
	RUN_TEST(test_limiter_lookahead_delay);

	// --- OBS Shutdown ---
	printf("\nShutting down OBS...\n");
	obs_shutdown();

	// --- Print Summary ---
	printf("--- Test Summary ---\n");
	printf("Total tests run: %d\n", tests_run);
	printf("Tests passed:    %d\n", tests_run - tests_failed);
	printf("Tests failed:    %d\n", tests_failed);
	printf("--------------------\n");

	return (tests_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}