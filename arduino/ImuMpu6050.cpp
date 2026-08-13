#include "ImuMpu6050.h"

ImuMpu6050::ImuMpu6050(uint8_t address) : address_(address)
{
}

void ImuMpu6050::startI2c(int sda_pin, int scl_pin)
{
#if defined(ARDUINO_ARCH_ESP32)
  if (sda_pin >= 0 && scl_pin >= 0)
  {
    Wire.begin(sda_pin, scl_pin);
  }
  else
  {
    Wire.begin();
  }
#else
  (void)sda_pin;
  (void)scl_pin;
  Wire.begin();
#endif

  Wire.setClock(I2C_CLOCK_HZ);
}

bool ImuMpu6050::begin(int sda_pin, int scl_pin)
{
  startI2c(sda_pin, scl_pin);
  delay(100);

  if (!probeAddress(address_))
  {
    if (address_ == ADDRESS_LOW && probeAddress(ADDRESS_HIGH))
    {
      address_ = ADDRESS_HIGH;
    }
    else if (address_ == ADDRESS_HIGH && probeAddress(ADDRESS_LOW))
    {
      address_ = ADDRESS_LOW;
    }
    else
    {
      return false;
    }
  }

  if (!writeRegister(REG_PWR_MGMT_1, 0x00))
  {
    return false;
  }

  delay(100);

  return writeRegister(REG_SMPLRT_DIV, 0x04) &&
         writeRegister(REG_CONFIG, 0x03) &&
         writeRegister(REG_GYRO_CONFIG, 0x18) &&
         writeRegister(REG_ACCEL_CONFIG, 0x08);
}

bool ImuMpu6050::readRaw(ImuRawData &data)
{
  uint8_t buffer[14];
  if (!readRegisters(REG_ACCEL_XOUT_H, buffer, sizeof(buffer)))
  {
    return false;
  }

  data.accel_x = (int16_t)((buffer[0] << 8) | buffer[1]);
  data.accel_y = (int16_t)((buffer[2] << 8) | buffer[3]);
  data.accel_z = (int16_t)((buffer[4] << 8) | buffer[5]);
  data.temperature = (int16_t)((buffer[6] << 8) | buffer[7]);
  data.gyro_x = (int16_t)((buffer[8] << 8) | buffer[9]);
  data.gyro_y = (int16_t)((buffer[10] << 8) | buffer[11]);
  data.gyro_z = (int16_t)((buffer[12] << 8) | buffer[13]);

  return true;
}

bool ImuMpu6050::read(ImuData &data)
{
  ImuRawData raw;
  if (!readRaw(raw))
  {
    return false;
  }

  data.accel_x_mps2 = (((float)raw.accel_x - accel_x_bias_) / ACCEL_LSB_PER_G) * GRAVITY_MPS2;
  data.accel_y_mps2 = (((float)raw.accel_y - accel_y_bias_) / ACCEL_LSB_PER_G) * GRAVITY_MPS2;
  data.accel_z_mps2 = (((float)raw.accel_z - accel_z_bias_) / ACCEL_LSB_PER_G) * GRAVITY_MPS2;
  data.gyro_x_rads = (((float)raw.gyro_x - gyro_x_bias_) / GYRO_LSB_PER_DPS) * DEG_PER_SEC_TO_RAD_PER_SEC;
  data.gyro_y_rads = (((float)raw.gyro_y - gyro_y_bias_) / GYRO_LSB_PER_DPS) * DEG_PER_SEC_TO_RAD_PER_SEC;
  data.gyro_z_rads = (((float)raw.gyro_z - gyro_z_bias_) / GYRO_LSB_PER_DPS) * DEG_PER_SEC_TO_RAD_PER_SEC;
  data.temperature_c = ((float)raw.temperature / 340.0f) + 36.53f;
  data.stationary = isStationary(data);

  if (data.stationary)
  {
    gyro_x_bias_ = ((1.0f - BIAS_LEARNING_RATE) * gyro_x_bias_) + (BIAS_LEARNING_RATE * (float)raw.gyro_x);
    gyro_y_bias_ = ((1.0f - BIAS_LEARNING_RATE) * gyro_y_bias_) + (BIAS_LEARNING_RATE * (float)raw.gyro_y);
    gyro_z_bias_ = ((1.0f - BIAS_LEARNING_RATE) * gyro_z_bias_) + (BIAS_LEARNING_RATE * (float)raw.gyro_z);
    data.gyro_x_rads = 0.0f;
    data.gyro_y_rads = 0.0f;
    data.gyro_z_rads = 0.0f;
  }
  else
  {
    data.gyro_x_rads = applyDeadband(data.gyro_x_rads);
    data.gyro_y_rads = applyDeadband(data.gyro_y_rads);
    data.gyro_z_rads = applyDeadband(data.gyro_z_rads);
  }

  updateOrientation(data);

  return true;
}

bool ImuMpu6050::calibrate(uint16_t samples)
{
  if (samples == 0)
  {
    return false;
  }

  long accel_x_sum = 0;
  long accel_y_sum = 0;
  long accel_z_sum = 0;
  long gyro_x_sum = 0;
  long gyro_y_sum = 0;
  long gyro_z_sum = 0;

  for (uint16_t i = 0; i < samples; ++i)
  {
    ImuRawData raw;
    if (!readRaw(raw))
    {
      return false;
    }

    accel_x_sum += raw.accel_x;
    accel_y_sum += raw.accel_y;
    accel_z_sum += raw.accel_z;
    gyro_x_sum += raw.gyro_x;
    gyro_y_sum += raw.gyro_y;
    gyro_z_sum += raw.gyro_z;
    delay(3);
  }

  const float accel_x_avg = (float)accel_x_sum / samples;
  const float accel_y_avg = (float)accel_y_sum / samples;
  const float accel_z_avg = (float)accel_z_sum / samples;
  const float expected_z = accel_z_avg >= 0.0f ? ACCEL_LSB_PER_G : -ACCEL_LSB_PER_G;

  accel_x_bias_ = accel_x_avg;
  accel_y_bias_ = accel_y_avg;
  accel_z_bias_ = accel_z_avg - expected_z;
  gyro_x_bias_ = (float)gyro_x_sum / samples;
  gyro_y_bias_ = (float)gyro_y_sum / samples;
  gyro_z_bias_ = (float)gyro_z_sum / samples;
  roll_rad_ = 0.0f;
  pitch_rad_ = 0.0f;
  yaw_rad_ = 0.0f;
  last_update_ms_ = 0;

  return true;
}

uint8_t ImuMpu6050::whoAmI() const
{
  return who_am_i_;
}

bool ImuMpu6050::writeRegister(uint8_t reg, uint8_t value)
{
  Wire.beginTransmission(address_);
  Wire.write(reg);
  Wire.write(value);
  return Wire.endTransmission() == 0;
}

bool ImuMpu6050::readRegisters(uint8_t start_reg, uint8_t *buffer, uint8_t length)
{
  Wire.beginTransmission(address_);
  Wire.write(start_reg);
  if (Wire.endTransmission(false) != 0)
  {
    return false;
  }

  const size_t received = Wire.requestFrom(address_, length);
  if (received != length)
  {
    return false;
  }

  for (uint8_t i = 0; i < length; ++i)
  {
    buffer[i] = Wire.read();
  }

  return true;
}

bool ImuMpu6050::readRegister(uint8_t reg, uint8_t &value)
{
  return readRegisters(reg, &value, 1);
}

bool ImuMpu6050::probeAddress(uint8_t address)
{
  const uint8_t previous_address = address_;
  address_ = address;

  uint8_t who_am_i = 0;
  const bool ok = readRegister(REG_WHO_AM_I, who_am_i) && isCompatibleWhoAmI(who_am_i);
  if (ok)
  {
    who_am_i_ = who_am_i;
  }

  address_ = previous_address;
  return ok;
}

bool ImuMpu6050::isCompatibleWhoAmI(uint8_t who_am_i) const
{
  return who_am_i == WHO_AM_I_MPU6050 ||
         who_am_i == WHO_AM_I_MPU6500 ||
         who_am_i == WHO_AM_I_MPU9250;
}

bool ImuMpu6050::isStationary(const ImuData &data) const
{
  const float accel_norm = sqrtf(
      (data.accel_x_mps2 * data.accel_x_mps2) +
      (data.accel_y_mps2 * data.accel_y_mps2) +
      (data.accel_z_mps2 * data.accel_z_mps2));
  const float gyro_norm = sqrtf(
      (data.gyro_x_rads * data.gyro_x_rads) +
      (data.gyro_y_rads * data.gyro_y_rads) +
      (data.gyro_z_rads * data.gyro_z_rads));

  return fabsf(accel_norm - GRAVITY_MPS2) < STATIONARY_ACCEL_TOLERANCE_MPS2 &&
         gyro_norm < STATIONARY_GYRO_THRESHOLD_RADS;
}

bool ImuMpu6050::isAccelReliable(const ImuData &data) const
{
  const float accel_norm = sqrtf(
      (data.accel_x_mps2 * data.accel_x_mps2) +
      (data.accel_y_mps2 * data.accel_y_mps2) +
      (data.accel_z_mps2 * data.accel_z_mps2));

  return fabsf(accel_norm - GRAVITY_MPS2) < ORIENTATION_ACCEL_TOLERANCE_MPS2;
}

float ImuMpu6050::applyDeadband(float value) const
{
  if (fabsf(value) < GYRO_DEADBAND_RADS)
  {
    return 0.0f;
  }

  return value;
}

void ImuMpu6050::accelToRollPitch(const ImuData &data, float &roll, float &pitch) const
{
  roll = atan2f(data.accel_y_mps2, data.accel_z_mps2);
  pitch = atan2f(
      -data.accel_x_mps2,
      sqrtf((data.accel_y_mps2 * data.accel_y_mps2) + (data.accel_z_mps2 * data.accel_z_mps2)));
}

void ImuMpu6050::updateOrientation(ImuData &data)
{
  const unsigned long now_ms = millis();
  float accel_roll = 0.0f;
  float accel_pitch = 0.0f;
  accelToRollPitch(data, accel_roll, accel_pitch);

  if (last_update_ms_ == 0)
  {
    roll_rad_ = accel_roll;
    pitch_rad_ = accel_pitch;
    yaw_rad_ = 0.0f;
    last_update_ms_ = now_ms;
  }
  else
  {
    float dt = (float)(now_ms - last_update_ms_) / 1000.0f;
    last_update_ms_ = now_ms;

    if (dt < 0.0f)
    {
      dt = 0.0f;
    }
    else if (dt > 0.1f)
    {
      dt = 0.1f;
    }

    const float gyro_roll = data.stationary ? roll_rad_ : roll_rad_ + (data.gyro_x_rads * dt);
    const float gyro_pitch = data.stationary ? pitch_rad_ : pitch_rad_ + (data.gyro_y_rads * dt);
    if (!data.stationary)
    {
      yaw_rad_ += data.gyro_z_rads * dt;
    }

    if (isAccelReliable(data))
    {
      roll_rad_ = (FILTER_ALPHA * gyro_roll) + ((1.0f - FILTER_ALPHA) * accel_roll);
      pitch_rad_ = (FILTER_ALPHA * gyro_pitch) + ((1.0f - FILTER_ALPHA) * accel_pitch);
    }
    else
    {
      roll_rad_ = gyro_roll;
      pitch_rad_ = gyro_pitch;
    }
  }

  data.roll_rad = roll_rad_;
  data.pitch_rad = pitch_rad_;
  data.yaw_rad = yaw_rad_;
}
