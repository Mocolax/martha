#include "PIController.h"

void pi_init(PIController* controller, float kp, float ki, float ts, float outputMin, float outputMax)
{
  controller->kp = kp;
  controller->ki = ki;
  controller->ts = ts;
  controller->integral = 0.0f;
  controller->outputMin = outputMin;
  controller->outputMax = outputMax;
  controller->error = 0.0f;
  controller->output = 0.0f;
}

float pi_update(PIController* controller, float reference, float measurement)
{
  controller->error = reference - measurement;
  controller->integral += controller->error * controller->ts;
  controller->output = controller->kp * controller->error + controller->ki * controller->integral;

  if (controller->output > controller->outputMax)
  {
    controller->output = controller->outputMax;
  }

  if (controller->output < controller->outputMin)
  {
    controller->output = controller->outputMin;
  }

  return controller->output;
}

void pi_reset(PIController* controller)
{
  controller->integral = 0.0f;
  controller->error = 0.0f;
  controller->output = 0.0f;
}

void pi_set_gains(PIController* controller, float kp, float ki)
{
  controller->kp = kp;
  controller->ki = ki;
}

void pi_set_limits(PIController* controller, float outputMin, float outputMax)
{
  controller->outputMin = outputMin;
  controller->outputMax = outputMax;

  if (controller->output > controller->outputMax)
  {
    controller->output = controller->outputMax;
  }

  if (controller->output < controller->outputMin)
  {
    controller->output = controller->outputMin;
  }
}
