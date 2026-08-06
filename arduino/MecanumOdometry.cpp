#include "MecanumOdometry.h"

#include "MecanumKinematics.h"

MecanumOdometry::MecanumOdometry(float wheelRadiusM,
                                 float centerProjectionSumM,
                                 float countsPerRevolution)
    : wheelRadiusM_(wheelRadiusM),
      centerProjectionSumM_(centerProjectionSumM),
      radiansPerCount_((2.0f * PI) / countsPerRevolution)
{
  for (uint8_t i = 0; i < 4; ++i)
  {
    data_.wheelPositionRad[i] = 0.0;
    data_.wheelVelocityRads[i] = 0.0f;
  }

  resetPose();
}

void MecanumOdometry::resetPose()
{
  data_.xM = 0.0f;
  data_.yM = 0.0f;
  data_.yawRad = 0.0f;
  data_.velocityXMps = 0.0f;
  data_.velocityYMps = 0.0f;
  data_.angularVelocityRads = 0.0f;
}

void MecanumOdometry::update(const int wheelCounts[4], float dtSeconds)
{
  if (dtSeconds <= 0.0f)
  {
    return;
  }

  float wheelDeltaRad[4];
  for (uint8_t i = 0; i < 4; ++i)
  {
    wheelDeltaRad[i] = wheelCounts[i] * radiansPerCount_;
    data_.wheelPositionRad[i] += wheelDeltaRad[i];
    data_.wheelVelocityRads[i] = wheelDeltaRad[i] / dtSeconds;
  }

  const float frontLeftRad = wheelDeltaRad[FRONT_LEFT];
  const float frontRightRad = wheelDeltaRad[FRONT_RIGHT];
  const float rearLeftRad = wheelDeltaRad[REAR_LEFT];
  const float rearRightRad = wheelDeltaRad[REAR_RIGHT];

  const float distanceFactor = wheelRadiusM_ * 0.25f;
  const float deltaXBody = distanceFactor *
      (frontLeftRad + frontRightRad + rearLeftRad + rearRightRad);
  const float deltaYBody = distanceFactor *
      (-frontLeftRad + frontRightRad + rearLeftRad - rearRightRad);
  const float deltaYaw = (distanceFactor / centerProjectionSumM_) *
      (-frontLeftRad + frontRightRad - rearLeftRad + rearRightRad);

  const float middleYaw = data_.yawRad + (deltaYaw * 0.5f);
  data_.xM += (deltaXBody * cosf(middleYaw)) - (deltaYBody * sinf(middleYaw));
  data_.yM += (deltaXBody * sinf(middleYaw)) + (deltaYBody * cosf(middleYaw));
  data_.yawRad = normalizeAngle(data_.yawRad + deltaYaw);

  data_.velocityXMps = deltaXBody / dtSeconds;
  data_.velocityYMps = deltaYBody / dtSeconds;
  data_.angularVelocityRads = deltaYaw / dtSeconds;
}

const MecanumOdometryData& MecanumOdometry::data() const
{
  return data_;
}

float MecanumOdometry::normalizeAngle(float angleRad)
{
  while (angleRad > PI)
  {
    angleRad -= 2.0f * PI;
  }

  while (angleRad < -PI)
  {
    angleRad += 2.0f * PI;
  }

  return angleRad;
}
