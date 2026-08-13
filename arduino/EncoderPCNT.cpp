#include "EncoderPCNT.h"

static const int PCNT_HIGH_LIMIT = 32767;
static const int PCNT_LOW_LIMIT = -32768;
static uint8_t nextPcntUnit = 0;

void encoder_init(EncoderPCNT* enc, int pinA, int pinB)
{
  enc->pinA = pinA;
  enc->pinB = pinB;
  enc->unit = (pcnt_unit_t)nextPcntUnit++;

  pcnt_config_t channelA = {};
  channelA.pulse_gpio_num = pinA;
  channelA.ctrl_gpio_num = pinB;
  channelA.lctrl_mode = PCNT_MODE_KEEP;
  channelA.hctrl_mode = PCNT_MODE_REVERSE;
  channelA.pos_mode = PCNT_COUNT_DEC;
  channelA.neg_mode = PCNT_COUNT_INC;
  channelA.counter_h_lim = PCNT_HIGH_LIMIT;
  channelA.counter_l_lim = PCNT_LOW_LIMIT;
  channelA.unit = enc->unit;
  channelA.channel = PCNT_CHANNEL_0;

  pcnt_config_t channelB = {};
  channelB.pulse_gpio_num = pinB;
  channelB.ctrl_gpio_num = pinA;
  channelB.lctrl_mode = PCNT_MODE_KEEP;
  channelB.hctrl_mode = PCNT_MODE_REVERSE;
  channelB.pos_mode = PCNT_COUNT_INC;
  channelB.neg_mode = PCNT_COUNT_DEC;
  channelB.counter_h_lim = PCNT_HIGH_LIMIT;
  channelB.counter_l_lim = PCNT_LOW_LIMIT;
  channelB.unit = enc->unit;
  channelB.channel = PCNT_CHANNEL_1;

  ESP_ERROR_CHECK(pcnt_unit_config(&channelA));
  ESP_ERROR_CHECK(pcnt_unit_config(&channelB));
  ESP_ERROR_CHECK(pcnt_set_filter_value(enc->unit, 100));
  ESP_ERROR_CHECK(pcnt_filter_enable(enc->unit));
  ESP_ERROR_CHECK(pcnt_counter_pause(enc->unit));
  ESP_ERROR_CHECK(pcnt_counter_clear(enc->unit));
  ESP_ERROR_CHECK(pcnt_counter_resume(enc->unit));
}

int encoder_get_count(EncoderPCNT* enc)
{
  int16_t count = 0;
  ESP_ERROR_CHECK(pcnt_get_counter_value(enc->unit, &count));
  return count;
}

void encoder_clear(EncoderPCNT* enc)
{
  ESP_ERROR_CHECK(pcnt_counter_clear(enc->unit));
}
