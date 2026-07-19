#ifndef PI_CONTROLLER_H
#define PI_CONTROLLER_H

#include <Arduino.h>

typedef struct {
  float kp;
  float ki;
  float ts;
  float integral;
  float outputMin;
  float outputMax;
  float error;
  float output;
} PIController;

void pi_init(PIController* controller, float kp, float ki, float ts, float outputMin, float outputMax);
float pi_update(PIController* controller, float reference, float measurement);
void pi_reset(PIController* controller);
void pi_set_gains(PIController* controller, float kp, float ki);
void pi_set_limits(PIController* controller, float outputMin, float outputMax);

#endif
