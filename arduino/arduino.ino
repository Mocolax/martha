#include "EncoderPCNT.h"
#include "ImuMpu6050.h"
#include "MecanumKinematics.h"
#include "MecanumOdometry.h"
#include "PIController.h"
#include "esp_arduino_version.h"

// ======================================================
// IMU CONFIG
// ======================================================
static const int I2C_SDA_PIN = 25;
static const int I2C_SCL_PIN = 26;
static const unsigned long IMU_PERIOD_MS = 20;

// ======================================================
// BATTERY MONITOR CONFIG
// R5 = 39k from battery to GPIO34, R6 = 10k to GND.
// ======================================================
static const int BATTERY_VOLTAGE_PIN = 34;
static const float BATTERY_DIVIDER_GAIN = 4.9f;
static const float BATTERY_CALIBRATION_FACTOR = 1.0f;
static const float BATTERY_LOW_THRESHOLD_V = 11.0f;
static const float BATTERY_RESET_THRESHOLD_V = 11.5f;
static const uint8_t BATTERY_SAMPLE_COUNT = 16;
static const uint8_t BATTERY_LOW_CONFIRMATION_COUNT = 3;
static const unsigned long BATTERY_CHECK_PERIOD_MS = 100;
static const unsigned long BATTERY_REPORT_PERIOD_MS = 1000;

// ======================================================
// MOTOR CONTROL CONFIG
// ======================================================
static const int MOTOR_SLEEP_PIN = 17;
static const int MOTOR_OVERCURRENT_PIN = 23;
static const int MOTOR_OVERCURRENT_ACTIVE_LEVEL = LOW;
static const uint8_t MOTOR_COUNT = 4;
static const float COUNTS_PER_REV = 3200.0f;
static const unsigned long CONTROL_PERIOD_US = 10000;
static const float CONTROL_TS = CONTROL_PERIOD_US * 0.000001f;
static const unsigned long JOINT_STATE_REPORT_PERIOD_MS = 50;
static const unsigned long ODOM_REPORT_PERIOD_MS = 40;
static const unsigned long CMD_VEL_TIMEOUT_MS = 500;

static const float WHEEL_RADIUS_M = 0.075f;
static const float CENTER_PROJECTION_SUM_M = 0.385f;
static const float MAX_WHEEL_RPM = 200.0f;

static const int PWM_FREQ_HZ = 10000;
static const int PWM_RESOLUTION_BITS = 8;

static const float KP = 2.0f;
static const float KI = 1.6f;
static const float OUTPUT_MIN = -255.0f;
static const float OUTPUT_MAX = 255.0f;
static const float DEFAULT_REFERENCE_RPM = 0.0f;

// ======================================================
// PINOUT
// Sincronizado con martha_circuits.kicad_sch.
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
    {12, 13},
    {36, 39},
    {4, 16},
    {21, 22},
};

MotorPins motorPins[MOTOR_COUNT] = {
    {27, 14, 0, 1},
    {32, 33, 2, 3},
    {15, 2, 4, 5},
    {18, 19, 6, 7},
};

// Ajustan la polaridad fisica al orden logico FL, FR, RL, RR.
// Cambiar solamente el elemento de la rueda que gire o cuente al reves.
static const int8_t MOTOR_OUTPUT_SIGN[MOTOR_COUNT] = {1, 1, 1, 1};
static const int8_t ENCODER_COUNT_SIGN[MOTOR_COUNT] = {1, 1, 1, 1};

// ======================================================
// STATE
// ======================================================
ImuMpu6050 imu;
MecanumKinematics mecanumKinematics(WHEEL_RADIUS_M,
                                    CENTER_PROJECTION_SUM_M,
                                    MAX_WHEEL_RPM);
MecanumOdometry mecanumOdometry(WHEEL_RADIUS_M,
                                CENTER_PROJECTION_SUM_M,
                                COUNTS_PER_REV);
EncoderPCNT encoders[MOTOR_COUNT];
PIController controllers[MOTOR_COUNT];

float referencesRpm[MOTOR_COUNT];
float rpmValues[MOTOR_COUNT];
float pwmValues[MOTOR_COUNT];
int8_t motorDirections[MOTOR_COUNT];
bool motorOvercurrentLatched = false;
bool batteryLowLatched = false;
bool cmdVelActive = false;
float batteryVoltage = 0.0f;
uint8_t batteryLowConfirmationCount = 0;

unsigned long lastImuReadMs = 0;
unsigned long lastBatteryCheckMs = 0;
unsigned long lastBatteryReportMs = 0;
unsigned long lastControlUs = 0;
unsigned long lastJointStateReportMs = 0;
unsigned long lastOdometryReportMs = 0;
unsigned long lastCmdVelMs = 0;

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

  initBatteryMonitor();
  initImu();
  initMotors();
  initEncoders();
  initControllers();
  startMotorControl();

  Serial.println((motorOvercurrentLatched || batteryLowLatched)
                     ? "MOTOR_PROTECTION_ACTIVE"
                     : "MOTOR_READY");
}

void loop()
{
  readSerialReference();
  updateBatteryMonitor();
  checkMotorProtection();
  checkCmdVelTimeout();
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
// BATTERY MONITOR
// ======================================================
void initBatteryMonitor()
{
  pinMode(MOTOR_SLEEP_PIN, OUTPUT);
  digitalWrite(MOTOR_SLEEP_PIN, LOW);
  pinMode(BATTERY_VOLTAGE_PIN, INPUT);
  analogReadResolution(12);
  analogSetPinAttenuation(BATTERY_VOLTAGE_PIN, ADC_11db);

  delay(10);
  batteryVoltage = readBatteryVoltage();
  lastBatteryCheckMs = millis();
  lastBatteryReportMs = lastBatteryCheckMs;
  publishBatteryVoltage();

  if (batteryVoltage < BATTERY_LOW_THRESHOLD_V)
  {
    batteryLowLatched = true;
    publishBatteryLowEvent();
  }
}

float readBatteryVoltage()
{
  analogReadMilliVolts(BATTERY_VOLTAGE_PIN);
  delayMicroseconds(250);

  uint32_t millivoltSum = 0;
  for (uint8_t i = 0; i < BATTERY_SAMPLE_COUNT; ++i)
  {
    millivoltSum += analogReadMilliVolts(BATTERY_VOLTAGE_PIN);
    delayMicroseconds(250);
  }

  const float adcVoltage =
      ((float)millivoltSum / BATTERY_SAMPLE_COUNT) * 0.001f;
  return adcVoltage * BATTERY_DIVIDER_GAIN * BATTERY_CALIBRATION_FACTOR;
}

void updateBatteryMonitor()
{
  const unsigned long now = millis();
  if (now - lastBatteryCheckMs < BATTERY_CHECK_PERIOD_MS)
  {
    return;
  }

  lastBatteryCheckMs = now;
  batteryVoltage = readBatteryVoltage();

  if (now - lastBatteryReportMs >= BATTERY_REPORT_PERIOD_MS)
  {
    lastBatteryReportMs = now;
    publishBatteryVoltage();
  }

  if (batteryLowLatched)
  {
    return;
  }

  if (batteryVoltage >= BATTERY_LOW_THRESHOLD_V)
  {
    batteryLowConfirmationCount = 0;
    return;
  }

  if (batteryLowConfirmationCount < BATTERY_LOW_CONFIRMATION_COUNT)
  {
    batteryLowConfirmationCount++;
  }

  if (batteryLowConfirmationCount >= BATTERY_LOW_CONFIRMATION_COUNT)
  {
    latchBatteryLow();
  }
}

void publishBatteryVoltage()
{
  Serial.print("battery,");
  Serial.println(batteryVoltage, 2);
}

void publishBatteryLowEvent()
{
  Serial.print("battery_too_low,");
  Serial.println(batteryVoltage, 2);
}

// ======================================================
// SERIAL COMMANDS
// cmd_vel,vx_mps,vy_mps,wz_rad_s
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
        processSerialCommand(serialBuffer);
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

void processSerialCommand(const char* command)
{
  if (strcmp(command, "reset") == 0 || strcmp(command, "RESET") == 0)
  {
    resetMotorProtection();
    return;
  }

  if (strcmp(command, "odom_reset") == 0 || strcmp(command, "ODOM_RESET") == 0)
  {
    resetOdometry();
    return;
  }

  float velocityXMps = 0.0f;
  float velocityYMps = 0.0f;
  float angularVelocityRads = 0.0f;
  if (sscanf(command,
             "cmd_vel,%f,%f,%f",
             &velocityXMps,
             &velocityYMps,
             &angularVelocityRads) == 3)
  {
    setCmdVel(velocityXMps, velocityYMps, angularVelocityRads);
    return;
  }

  Serial.println("cmd_error: expected cmd_vel,vx,vy,wz, reset or odom_reset");
}

void setCmdVel(float velocityXMps, float velocityYMps, float angularVelocityRads)
{
  if (motorOvercurrentLatched || batteryLowLatched)
  {
    Serial.println("cmd_vel_blocked: motor protection active");
    return;
  }

  mecanumKinematics.calculateWheelRpm(velocityXMps,
                                      velocityYMps,
                                      angularVelocityRads,
                                      referencesRpm);
  lastCmdVelMs = millis();
  cmdVelActive = true;

  publishWheelReferences();
}

void checkCmdVelTimeout()
{
  if (!cmdVelActive || millis() - lastCmdVelMs <= CMD_VEL_TIMEOUT_MS)
  {
    return;
  }

  cmdVelActive = false;
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    referencesRpm[i] = 0.0f;
    pi_reset(&controllers[i]);
  }
  stopAllMotors();
  Serial.println("cmd_vel_timeout");
}

void publishWheelReferences()
{
  Serial.print("wheel_ref_rpm");
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    Serial.print(",");
    Serial.print(referencesRpm[i], 2);
  }
  Serial.println();
}

// ======================================================
// MOTOR SETUP
// ======================================================
void initMotors()
{
  pinMode(MOTOR_OVERCURRENT_PIN, INPUT_PULLUP);
  pinMode(MOTOR_SLEEP_PIN, OUTPUT);
  digitalWrite(MOTOR_SLEEP_PIN, LOW);

  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    attachMotorPwm(motorPins[i].pwm1, motorPins[i].channel1);
    attachMotorPwm(motorPins[i].pwm2, motorPins[i].channel2);
    writeMotorPin(i, 0, 0);
  }

  if (batteryLowLatched)
  {
    return;
  }

  if (isMotorOvercurrentActive())
  {
    motorOvercurrentLatched = true;
    Serial.println("motor_overcurrent");
    return;
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

  lastControlUs = micros();
  lastJointStateReportMs = millis();
  lastOdometryReportMs = millis();
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

bool isMotorOvercurrentActive()
{
  return digitalRead(MOTOR_OVERCURRENT_PIN) == MOTOR_OVERCURRENT_ACTIVE_LEVEL;
}

void checkMotorProtection()
{
  if (motorOvercurrentLatched)
  {
    return;
  }

  if (!isMotorOvercurrentActive())
  {
    return;
  }

  latchMotorOvercurrent();
}

void latchMotorOvercurrent()
{
  motorOvercurrentLatched = true;
  cmdVelActive = false;
  stopAllMotors();
  digitalWrite(MOTOR_SLEEP_PIN, LOW);
  Serial.println("motor_overcurrent");
}

void latchBatteryLow()
{
  batteryLowLatched = true;
  batteryLowConfirmationCount = 0;
  cmdVelActive = false;

  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    referencesRpm[i] = 0.0f;
  }

  stopAllMotors();
  digitalWrite(MOTOR_SLEEP_PIN, LOW);
  publishBatteryLowEvent();
}

void resetMotorProtection()
{
  batteryVoltage = readBatteryVoltage();
  publishBatteryVoltage();

  if (batteryVoltage < BATTERY_RESET_THRESHOLD_V)
  {
    Serial.print("battery_reset_blocked,");
    Serial.println(batteryVoltage, 2);
    return;
  }

  if (isMotorOvercurrentActive())
  {
    Serial.println("motor_overcurrent_reset_blocked");
    return;
  }

  motorOvercurrentLatched = false;
  batteryLowLatched = false;
  batteryLowConfirmationCount = 0;
  digitalWrite(MOTOR_SLEEP_PIN, HIGH);
  startMotorControl();
  Serial.println("motor_protection_reset");
}

// ======================================================
// PI CONTROL
// ======================================================
void updateMotorControl()
{
  if (motorOvercurrentLatched || batteryLowLatched)
  {
    return;
  }

  const unsigned long nowUs = micros();
  const unsigned long elapsedUs = nowUs - lastControlUs;
  if (elapsedUs < CONTROL_PERIOD_US)
  {
    return;
  }

  lastControlUs = nowUs;
  const float dtSeconds = elapsedUs * 0.000001f;
  int wheelCounts[MOTOR_COUNT];

  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    const int rawCount = encoder_get_count(&encoders[i]);
    encoder_clear(&encoders[i]);
    wheelCounts[i] = rawCount * ENCODER_COUNT_SIGN[i];

    rpmValues[i] = ((float)wheelCounts[i] / COUNTS_PER_REV) *
        (60.0f / dtSeconds);
    controllers[i].ts = dtSeconds;
    pwmValues[i] = pi_update(&controllers[i], referencesRpm[i], rpmValues[i]);
    setMotorPWM(i, pwmValues[i] * MOTOR_OUTPUT_SIGN[i]);
  }

  mecanumOdometry.update(wheelCounts, dtSeconds);

  const unsigned long nowMs = millis();
  if (nowMs - lastJointStateReportMs >= JOINT_STATE_REPORT_PERIOD_MS)
  {
    lastJointStateReportMs = nowMs;
    publishJointState();
  }

  if (nowMs - lastOdometryReportMs >= ODOM_REPORT_PERIOD_MS)
  {
    lastOdometryReportMs = nowMs;
    publishOdometry();
  }
}

void publishJointState()
{
  const MecanumOdometryData& odometry = mecanumOdometry.data();

  Serial.print("joint");
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    Serial.print(",");
    Serial.print(odometry.wheelPositionRad[i], 5);
  }

  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    Serial.print(",");
    Serial.print(odometry.wheelVelocityRads[i], 5);
  }
  Serial.println();
}

void publishOdometry()
{
  const MecanumOdometryData& odometry = mecanumOdometry.data();

  Serial.print("odom,");
  Serial.print(odometry.xM, 6);
  Serial.print(",");
  Serial.print(odometry.yM, 6);
  Serial.print(",");
  Serial.print(odometry.yawRad, 6);
  Serial.print(",");
  Serial.print(odometry.velocityXMps, 6);
  Serial.print(",");
  Serial.print(odometry.velocityYMps, 6);
  Serial.print(",");
  Serial.println(odometry.angularVelocityRads, 6);
}

void resetOdometry()
{
  mecanumOdometry.resetPose();
  for (uint8_t i = 0; i < MOTOR_COUNT; ++i)
  {
    encoder_clear(&encoders[i]);
  }
  lastControlUs = micros();
  Serial.println("odom_reset");
}
