#ifndef MECANUM_KINEMATICS_H
#define MECANUM_KINEMATICS_H

#include <Arduino.h>

enum WheelIndex : uint8_t
{
  FRONT_LEFT = 0,
  FRONT_RIGHT = 1,
  REAR_LEFT = 2,
  REAR_RIGHT = 3,
};

class MecanumKinematics
{
public:
  MecanumKinematics(float wheelRadiusM, float centerProjectionSumM, float maxWheelRpm);

  void calculateWheelRpm(float velocityXMps,
                         float velocityYMps,
                         float angularVelocityRads,
                         float wheelRpm[4]) const;

private:
  float wheelRadiusM_;
  float centerProjectionSumM_;
  float maxWheelRpm_;

  void limitWheelRpm(float wheelRpm[4]) const;
};

#endif
