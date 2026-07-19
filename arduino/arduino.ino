#include "EncoderPCNT.h"
#include "ImuMpu6050.h"
#include "PIController.h"
#include "esp_arduino_version.h"

// ======================================================
// IMU CONFIG
// ======================================================
static const int I2C_SDA_PIN = 21;
static const int I2C_SCL_PIN = 22;
static const unsigned long IMU_PERIOD_MS = 20;

// ======================================================
// MOTOR CONTROL CONFIG
// ======================================================
static const int MOTOR_SLEEP_PIN = 2;
static const uint8_t MOTOR_COUNT = 4;
static const float COUNTS_PER_REV = 3200.0f;
static const float CONTROL_TS = 0.01f;
static const unsigned long CONTROL_PERIOD_MS = 10;
static const unsigned long RPM_REPORT_PERIOD_MS = 50;

static const int PWM_FREQ_HZ = 10000;
static const int PWM_RESOLUTION_BITS = 8;

static const float KP = 2.0f;
static const float KI = 1.6f;
static const float OUTPUT_MIN = -255.0f;
static const float OUTPUT_MAX = 255.0f;
static const float DEFAULT_REFERENCE_RPM = 0.0f;

// ======================================================
// PINOUT
// Ajustar estos pines segun el cableado real.
// ======================================================
struct EncoderPins
{
  int pinA;
  int pinB;
};

struct MotorPins
{
  int pwm1;
  int pwm2;
  uint8_t channel1;
  uint8_t channel2;
};

EncoderPins encoderPins[MOTOR_COUNT] = {
    {34, 35},
    {36, 39},
    {32, 33},
    {18, 19},
};

MotorPins motorPins[MOTOR_COUNT] = {
    {25, 26, 0, 1},
    {27, 14, 2, 3},
    {16, 17, 4, 5},
    {23, 13, 6, 7},
};

// ======================================================
// STATE
// ======================================================
ImuMpu6050 imu;
EncoderPCNT encoders[MOTOR_COUNT];
PIController controllers[MOTOR_COUNT];

float referencesRpm[MOTOR_COUNT];
float rpmValues[MOTOR_COUNT];
float pwmValues[MOTOR_COUNT];
int8_t motorDirections[MOTOR_COUNT];

unsigned long lastImuReadMs = 0;
unsigned long lastControlMs = 0;
unsigned long lastRpmReportMs = 0;

char serialBuffer[64];
size_t serialBufferLength = 0;

// ======================================================
// SETUP / LOOP
// ======================================================
void setup()
{
  Serial.begin(115200);
  const unsigned long serialStartMs = millis();
  while (!Serial && millis() - serialStartMs < 2000)
  {
  }

  initImu();
  initMotors();
  initEncoders();
  initControllers();
  startMotorControl();

  Serial.println("MOTOR_READY");
}

void loop()
{
  readSerialReference();
  updateMotorControl();
  publishImu();
}

// ======================================================
// IMU
// ======================================================
void initImu()
{
  ImuMpu6050::startI2c(I2C_SDA_PIN, I2C_SCL_PIN);
  printI2cScan();

  if (!imu.begin(I2C_SDA_PIN, I2C_SCL_PIN))
  {
    Serial.println("IMU_ERROR: compatible IMU not detected");
    while (true)
    {
      delay(500);
    }
  }

  Serial.print("IMU_WHO_AM_I: 0x");
  if (imu.whoAmI() < 16)
  {
    Serial.print("0");
  }
  Serial.println(imu.whoAmI(), HEX);

  Serial.println("IMU_CALIBRATION: keep the sensor still and flat");
  delay(1000);

  if (!imu.calibrate(600))
  {
    Serial.println("IMU_ERROR: calibration failed");
    while (true)
    {
      delay(500);
    }
  }

  Serial.println("IMU_CALIBRATION_DONE");
  Serial.println("IMU_READY");
}

void publishImu()
{
  const unsigned long now = millis();
  if (now - lastImuReadMs < IMU_PERIOD_MS)
  {
    return;
  }

  lastImuReadMs = now;

  ImuData data;
  if (!imu.read(data))
  {
    Serial.println("IMU_ERROR: read failed");
    return;
  }

  Serial.print("imu,");
  Serial.print(data.accel_x_mps2, 4);
  Serial.print(",");
  Serial.print(data.accel_y_mps2, 4);
  Serial.print(",");
  Serial.print(data.accel_z_mps2, 4);
  Serial.print(",");
  Serial.print(data.gyro_x_rads, 6);
  Serial.print(",");
  Serial.print(data.gyro_y_rads, 6);
  Serial.print(",");
  Serial.print(data.gyro_z_rads, 6);
  Serial.print(",");
  Serial.print(data.roll_rad, 6);
  Serial.print(",");
  Serial.print(data.pitch_rad, 6);
  Serial.print(",");
  Serial.print(data.yaw_rad, 6);
  Serial.print(",");
  Serial.println(data.stationary ? 1 : 0);
}

void printI2cScan()
{
  Serial.println("I2C_SCAN_START");

  uint8_t devicesFound = 0;
  for (uint8_t address = 1; address < 127; ++address)
  {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission();

    if (error == 0)
    {
      Serial.print("I2C_DEVICE: 0x");
      if (address < 16)
      {
        Serial.print("0");
      }
      Serial.println(address, HEX);

      if (address == 0x68 || address == 0x69)
      {
        Wire.beginTransmission(address);
        Wire.write(0x75);
        if (Wire.endTransmission(false) == 0 && Wire.requestFrom(address, (uint8_t)1) == 1)
        {
          const uint8_t whoAmI = Wire.read();
          Serial.print("I2C_WHO_AM_I: 0x");
          if (whoAmI < 16)
          {
            Serial.print("0");
          }
          Serial.println(whoAmI, HEX);
        }
      }

      devicesFound++;
    }
  }

  if (devicesFound == 0)
  {
    Serial.println("I2C_SCAN_EMPTY");
  }

  Serial.println("I2C_SCAN_END");
}

// ======================================================
// SERIAL REFERENCE
// Enviar una linea con un numero RPM, por ejemplo: 120
// ======================================================
void readSerialReference()
{
  while (Serial.available() > 0)
  {
    const char c = (char)Serial.read();

    if (c == '\n' || c == '\r')
    {
      if (serialBufferLength > 0)
      {
        serialBuffer[serialBufferLength] = '\0';
        setAllReferences(atof(serialBuffer));
        serialBufferLength = 0;
      }
      continue;
    }

    if (serialBufferLength < sizeof(serialBuffer) - 1)
    {
      serialBuffer[serialBufferLength++] = c;
    }
    else
    {
      serialBufferLength = 0;
      Serial.println("REF_ERROR: line too long");
    }
  }
}

void setAllReferences(float referenceRpm)
{
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    referencesRpm[i] = referenceRpm;
  }

  Serial.print("ref_rpm,");
  Serial.println(referenceRpm, 3);
}

// ======================================================
// MOTOR SETUP
// ======================================================
void initMotors()
{
  pinMode(MOTOR_SLEEP_PIN, OUTPUT);
  digitalWrite(MOTOR_SLEEP_PIN, LOW);

  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    attachMotorPwm(motorPins[i].pwm1, motorPins[i].channel1);
    attachMotorPwm(motorPins[i].pwm2, motorPins[i].channel2);
    writeMotorPin(i, 0, 0);
  }

  digitalWrite(MOTOR_SLEEP_PIN, HIGH);
}

void attachMotorPwm(int pin, uint8_t channel)
{
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttachChannel(pin, PWM_FREQ_HZ, PWM_RESOLUTION_BITS, channel);
#else
  ledcSetup(channel, PWM_FREQ_HZ, PWM_RESOLUTION_BITS);
  ledcAttachPin(pin, channel);
#endif
}

void initEncoders()
{
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    encoder_init(&encoders[i], encoderPins[i].pinA, encoderPins[i].pinB);
  }
}

void initControllers()
{
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    referencesRpm[i] = DEFAULT_REFERENCE_RPM;
    rpmValues[i] = 0.0f;
    pwmValues[i] = 0.0f;
    motorDirections[i] = 0;
    pi_init(&controllers[i], KP, KI, CONTROL_TS, OUTPUT_MIN, OUTPUT_MAX);
  }
}

void startMotorControl()
{
  stopAllMotors();
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    encoder_clear(&encoders[i]);
    pi_reset(&controllers[i]);
  }

  lastControlMs = millis();
  lastRpmReportMs = millis();
}

// ======================================================
// MOTOR OUTPUT
// ======================================================
void motorStop(uint8_t motorIndex)
{
  brakeMotor(motorIndex);
  motorDirections[motorIndex] = 0;
}

void brakeMotor(uint8_t motorIndex)
{
  writeMotorPin(motorIndex, 0, 0);
}

uint8_t dutyFromControl(float controlOutput)
{
  float magnitude = controlOutput < 0.0f ? -controlOutput : controlOutput;
  if (magnitude > 255.0f)
  {
    magnitude = 255.0f;
  }

  return (uint8_t)magnitude;
}

int8_t directionFromControl(float controlOutput)
{
  if (controlOutput > 0.0f)
  {
    return 1;
  }

  if (controlOutput < 0.0f)
  {
    return -1;
  }

  return 0;
}

void setMotorPWM(uint8_t motorIndex, float controlOutput)
{
  const int8_t nextDirection = directionFromControl(controlOutput);

  if (motorDirections[motorIndex] != 0 &&
      nextDirection != 0 &&
      motorDirections[motorIndex] != nextDirection)
  {
    brakeMotor(motorIndex);
    motorDirections[motorIndex] = 0;
    return;
  }

  const uint8_t duty = dutyFromControl(controlOutput);

  if (nextDirection > 0)
  {
    writeMotorPin(motorIndex, duty, 0);
  }
  else if (nextDirection < 0)
  {
    writeMotorPin(motorIndex, 0, duty);
  }
  else
  {
    brakeMotor(motorIndex);
  }

  motorDirections[motorIndex] = nextDirection;
}

void writeMotorPin(uint8_t motorIndex, uint8_t pwm1Duty, uint8_t pwm2Duty)
{
  writeMotorPwm(motorPins[motorIndex].pwm1, motorPins[motorIndex].channel1, pwm1Duty);
  writeMotorPwm(motorPins[motorIndex].pwm2, motorPins[motorIndex].channel2, pwm2Duty);
}

void writeMotorPwm(int pin, uint8_t channel, uint8_t duty)
{
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)channel;
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(channel, duty);
#endif
}

void stopAllMotors()
{
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    motorStop(i);
  }
}

// ======================================================
// PI CONTROL
// ======================================================
void updateMotorControl()
{
  const unsigned long now = millis();
  if (now - lastControlMs < CONTROL_PERIOD_MS)
  {
    return;
  }

  lastControlMs = now;

  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    const int count = encoder_get_count(&encoders[i]);
    encoder_clear(&encoders[i]);

    rpmValues[i] = ((float)count / COUNTS_PER_REV) * (60.0f / CONTROL_TS);
    pwmValues[i] = pi_update(&controllers[i], referencesRpm[i], rpmValues[i]);
    setMotorPWM(i, pwmValues[i]);
  }

  if (now - lastRpmReportMs >= RPM_REPORT_PERIOD_MS)
  {
    lastRpmReportMs = now;
    publishRpm();
  }
}

void publishRpm()
{
  Serial.print("rpm");
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    Serial.print(",");
    Serial.print(rpmValues[i], 2);
  }
  Serial.println();
}
