#ifndef IMU_MPU6050_H
#define IMU_MPU6050_H

#include <Arduino.h>
#include <Wire.h>

struct ImuRawData
{
  int16_t accel_x;
  int16_t accel_y;
  int16_t accel_z;
  int16_t temperature;
  int16_t gyro_x;
  int16_t gyro_y;
  int16_t gyro_z;
};

struct ImuData
{
  float accel_x_mps2;
  float accel_y_mps2;
  float accel_z_mps2;
  float gyro_x_rads;
  float gyro_y_rads;
  float gyro_z_rads;
  float roll_rad;
  float pitch_rad;
  float yaw_rad;
  float temperature_c;
  bool stationary;
};

class ImuMpu6050
{
public:
  explicit ImuMpu6050(uint8_t address = 0x68);

  static void startI2c(int sda_pin = -1, int scl_pin = -1);

  bool begin(int sda_pin = -1, int scl_pin = -1);
  bool readRaw(ImuRawData &data);
  bool read(ImuData &data);
  bool calibrate(uint16_t samples = 600);
  bool calibrateGyro(uint16_t samples = 500);
  uint8_t whoAmI() const;

private:
  static const uint8_t ADDRESS_LOW = 0x68;
  static const uint8_t ADDRESS_HIGH = 0x69;

  static const uint8_t REG_PWR_MGMT_1 = 0x6B;
  static const uint8_t REG_SMPLRT_DIV = 0x19;
  static const uint8_t REG_CONFIG = 0x1A;
  static const uint8_t REG_GYRO_CONFIG = 0x1B;
  static const uint8_t REG_ACCEL_CONFIG = 0x1C;
  static const uint8_t REG_ACCEL_XOUT_H = 0x3B;
  static const uint8_t REG_WHO_AM_I = 0x75;
  static const uint8_t WHO_AM_I_MPU6050 = 0x68;
  static const uint8_t WHO_AM_I_MPU6500 = 0x70;
  static const uint8_t WHO_AM_I_MPU9250 = 0x71;

  static constexpr float GRAVITY_MPS2 = 9.80665f;
  static constexpr float ACCEL_LSB_PER_G = 8192.0f;
  static constexpr float GYRO_LSB_PER_DPS = 16.4f;
  static constexpr float DEG_PER_SEC_TO_RAD_PER_SEC = 0.017453292519943295f;
  static constexpr float FILTER_ALPHA = 0.98f;
  static constexpr float GYRO_DEADBAND_RADS = 0.02f;
  static constexpr float STATIONARY_GYRO_THRESHOLD_RADS = 0.08f;
  static constexpr float STATIONARY_ACCEL_TOLERANCE_MPS2 = 1.5f;
  static constexpr float ORIENTATION_ACCEL_TOLERANCE_MPS2 = 2.0f;
  static constexpr float BIAS_LEARNING_RATE = 0.02f;
  static const uint32_t I2C_CLOCK_HZ = 100000;

  uint8_t address_;
  uint8_t who_am_i_ = 0;
  float accel_x_bias_ = 0.0f;
  float accel_y_bias_ = 0.0f;
  float accel_z_bias_ = 0.0f;
  float gyro_x_bias_ = 0.0f;
  float gyro_y_bias_ = 0.0f;
  float gyro_z_bias_ = 0.0f;
  float roll_rad_ = 0.0f;
  float pitch_rad_ = 0.0f;
  float yaw_rad_ = 0.0f;
  unsigned long last_update_ms_ = 0;

  bool writeRegister(uint8_t reg, uint8_t value);
  bool readRegisters(uint8_t start_reg, uint8_t *buffer, uint8_t length);
  bool readRegister(uint8_t reg, uint8_t &value);
  bool probeAddress(uint8_t address);
  bool isCompatibleWhoAmI(uint8_t who_am_i) const;
  bool isStationary(const ImuData &data) const;
  bool isAccelReliable(const ImuData &data) const;
  float applyDeadband(float value) const;
  void accelToRollPitch(const ImuData &data, float &roll, float &pitch) const;
  void updateOrientation(ImuData &data);
};

#endif
