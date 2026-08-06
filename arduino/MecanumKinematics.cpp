#include "MecanumKinematics.h"

static constexpr float RADS_TO_RPM = 60.0f / (2.0f * PI);

MecanumKinematics::MecanumKinematics(float wheelRadiusM,
                                     float centerProjectionSumM,
                                     float maxWheelRpm)
    : wheelRadiusM_(wheelRadiusM),
      centerProjectionSumM_(centerProjectionSumM),
      maxWheelRpm_(maxWheelRpm)
{
}

void MecanumKinematics::calculateWheelRpm(float velocityXMps,
                                          float velocityYMps,
                                          float angularVelocityRads,
                                          float wheelRpm[4]) const
{
  const float rotationVelocity = centerProjectionSumM_ * angularVelocityRads;
  const float linearToRpm = RADS_TO_RPM / wheelRadiusM_;

  wheelRpm[FRONT_LEFT] = (velocityXMps - velocityYMps - rotationVelocity) * linearToRpm;
  wheelRpm[FRONT_RIGHT] = (velocityXMps + velocityYMps + rotationVelocity) * linearToRpm;
  wheelRpm[REAR_LEFT] = (velocityXMps + velocityYMps - rotationVelocity) * linearToRpm;
  wheelRpm[REAR_RIGHT] = (velocityXMps - velocityYMps + rotationVelocity) * linearToRpm;

  limitWheelRpm(wheelRpm);
}

void MecanumKinematics::limitWheelRpm(float wheelRpm[4]) const
{
  if (maxWheelRpm_ <= 0.0f)
  {
    return;
  }

  float largestMagnitude = 0.0f;
  for (uint8_t i = 0; i < 4; ++i)
  {
    const float magnitude = fabsf(wheelRpm[i]);
    if (magnitude > largestMagnitude)
    {
      largestMagnitude = magnitude;
    }
  }

  if (largestMagnitude <= maxWheelRpm_)
  {
    return;
  }

  const float scale = maxWheelRpm_ / largestMagnitude;
  for (uint8_t i = 0; i < 4; ++i)
  {
    wheelRpm[i] *= scale;
  }
}
