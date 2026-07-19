#ifndef ENCODER_PCNT_H
#define ENCODER_PCNT_H

#include <Arduino.h>
#include "driver/pcnt.h"

typedef struct {
  int pinA;
  int pinB;
  pcnt_unit_t unit;
} EncoderPCNT;

void encoder_init(EncoderPCNT* enc, int pinA, int pinB);
int encoder_get_count(EncoderPCNT* enc);
void encoder_clear(EncoderPCNT* enc);
void encoder_start(EncoderPCNT* enc);
void encoder_stop(EncoderPCNT* enc);

#endif
