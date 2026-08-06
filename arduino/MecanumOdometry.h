#ifndef MECANUM_ODOMETRY_H
#define MECANUM_ODOMETRY_H

#include <Arduino.h>

struct MecanumOdometryData
{
  float xM;
  float yM;
  float yawRad;
  float velocityXMps;
  float velocityYMps;
  float angularVelocityRads;
  double wheelPositionRad[4];
  float wheelVelocityRads[4];
};

class MecanumOdometry
{
public:
  MecanumOdometry(float wheelRadiusM,
                  float centerProjectionSumM,
                  float countsPerRevolution);

  void resetPose();
  void update(const int wheelCounts[4], float dtSeconds);
  const MecanumOdometryData& data() const;

private:
  float wheelRadiusM_;
  float centerProjectionSumM_;
  float radiansPerCount_;
  MecanumOdometryData data_;

  static float normalizeAngle(float angleRad);
};

#endif
