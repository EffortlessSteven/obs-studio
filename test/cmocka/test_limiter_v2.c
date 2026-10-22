#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

/* Exercise the production DSP helpers without duplicating their implementation. */
#include "../../plugins/obs-filters/limiter-v2-filter.c"

/* The standalone executable has no module translation catalog. */
const char *obs_module_text(const char *text)
{
	return text;
}

static void test_defaults_and_presets(void **state)
{
	UNUSED_PARAMETER(state);
	obs_data_t *settings = obs_data_create();
	limiter_v2_filter.get_defaults(settings);
	assert_string_equal(obs_data_get_string(settings, S_PRESET), PRESET_VAL_DEFAULT);
	assert_float_equal(obs_data_get_double(settings, S_FILTER_THRESHOLD), -6.0, 0.001);
	assert_true(obs_data_get_bool(settings, S_LOOKAHEAD_ENABLED));

	obs_properties_t *props = limiter_v2_filter.get_properties(NULL);
	obs_property_t *preset = obs_properties_get(props, S_PRESET);
	assert_non_null(preset);
	const char *names[] = {PRESET_VAL_DEFAULT,     PRESET_VAL_PODCAST, PRESET_VAL_STREAMING, PRESET_VAL_AGGRESSIVE,
			       PRESET_VAL_TRANSPARENT, PRESET_VAL_MUSIC,   PRESET_VAL_BRICKWALL};
	const double thresholds[] = {-6.0, -8.0, -7.0, -5.0, -1.5, -2.0, -0.3};
	for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
		obs_data_set_string(settings, S_PRESET, names[i]);
		assert_true(obs_property_modified(preset, settings));
		assert_float_equal(obs_data_get_double(settings, S_FILTER_THRESHOLD), thresholds[i], 0.001);
	}
	obs_data_set_string(settings, S_PRESET, PRESET_VAL_CUSTOM);
	obs_data_set_double(settings, S_FILTER_THRESHOLD, -12.5);
	obs_property_modified(preset, settings);
	assert_float_equal(obs_data_get_double(settings, S_FILTER_THRESHOLD), -12.5, 0.001);
	obs_properties_destroy(props);
	obs_data_release(settings);
}

static void test_audio_thresholds(void **state)
{
	UNUSED_PARAMETER(state);
	const float thresholds[] = {-20.0f, -10.0f, -6.0f, -1.0f};
	for (size_t t = 0; t < sizeof(thresholds) / sizeof(thresholds[0]); t++) {
		struct limiter_v2_data cd = {0};
		cd.sample_rate = 48000;
		cd.num_channels = 2;
		cd.threshold_db = thresholds[t];
		cd.attack_coeff = gain_coefficient(cd.sample_rate, FIXED_ATTACK_TIME_MS);
		cd.release_coeff = gain_coefficient(cd.sample_rate, 60.0f);
		cd.output_gain = 1.0f;
		float left[512], right[512];
		struct obs_audio_data audio = {0};
		audio.frames = 512;
		audio.data[0] = (uint8_t *)left;
		audio.data[1] = (uint8_t *)right;
		/* A sustained signal lets the specified attack time settle. */
		for (size_t block = 0; block < 4; block++) {
			for (size_t i = 0; i < 512; i++) {
				left[i] = 2.0f;
				right[i] = -1.0f;
			}
			assert_ptr_equal(limiter_v2_filter.filter_audio(&cd, &audio), &audio);
		}
		float threshold = db_to_mul(thresholds[t]);
		for (size_t i = 0; i < 512; i++) {
			assert_float_equal(left[i], threshold, 0.001f);
			assert_float_equal(right[i], -threshold / 2.0f, 0.001f);
		}
		bfree(cd.envelope_buf);
	}
}

static void test_lookahead_delay(void **state)
{
	UNUSED_PARAMETER(state);
	struct limiter_v2_data cd = {0};
	cd.sample_rate = 48000;
	cd.num_channels = 1;
	cd.output_gain = 1.0f;
	cd.lookahead_enabled = true;
	cd.lookahead_samples = 6;
	assert_true(update_lookahead_buffers(&cd));
	float samples[4] = {0.25f, 0.0f, 0.0f, 0.0f};
	struct obs_audio_data audio = {0};
	audio.frames = 4;
	audio.data[0] = (uint8_t *)samples;
	limiter_v2_filter.filter_audio(&cd, &audio);
	for (size_t i = 0; i < 4; i++)
		assert_float_equal(samples[i], 0.0f, 0.00001f);
	limiter_v2_filter.filter_audio(&cd, &audio);
	assert_float_equal(samples[0], 0.0f, 0.00001f);
	assert_float_equal(samples[1], 0.0f, 0.00001f);
	assert_float_equal(samples[2], 0.25f, 0.00001f);
	assert_float_equal(samples[3], 0.0f, 0.00001f);
	cd.lookahead_enabled = false;
	assert_true(update_lookahead_buffers(&cd));
	assert_false(cd.lookahead_buffers_initialized);
	bfree(cd.envelope_buf);
}

static void test_empty_audio(void **state)
{
	UNUSED_PARAMETER(state);
	struct limiter_v2_data cd = {0};
	struct obs_audio_data audio = {0};
	assert_ptr_equal(limiter_v2_filter.filter_audio(NULL, &audio), &audio);
	assert_null(limiter_v2_filter.filter_audio(&cd, NULL));
	assert_ptr_equal(limiter_v2_filter.filter_audio(&cd, &audio), &audio);
	assert_float_equal(gain_coefficient(0, 1.0f), 0.0f, 0.00001f);
	assert_float_equal(gain_coefficient(48000, 0.0f), 0.0f, 0.00001f);
}

int main(void)
{
	const struct CMUnitTest tests[] = {
		cmocka_unit_test(test_defaults_and_presets),
		cmocka_unit_test(test_audio_thresholds),
		cmocka_unit_test(test_lookahead_delay),
		cmocka_unit_test(test_empty_audio),
	};
	return cmocka_run_group_tests(tests, NULL, NULL);
}
