enum Order 
{
  HELLO = 0,
  ALREADY_CONNECTED = 1,
  ERROR = 2,
  MOTOR = 3,
  RECEIVED = 4,
  STOP = 5,
};

typedef enum Order Order;

struct Direction 
{
  int8_t Vx;
  int8_t Vy;
  int8_t th;
};

struct Driver
{
  byte PWM1;
  byte PWM2;
  bool EN;
  bool ENB;
};

Direction direction;

Driver MotorFL;
Driver MotorFR;
Driver MotorBL;
Driver MotorBR;

bool is_connected = false; // Changed when Arduino connects succesfully with PC
bool DEBUG = true; // Used while testing (Ideally set to false once everything works to avoid sending unnecesary data)


void setup() 
{
  Serial.begin(115200);
  // LED pins
  pinMode(11, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode( 9, OUTPUT);
  // Add Motor pins

  // Wait until Arduino can connect to ROS 
  
  while(is_connected == false)
  {
    get_serial();
    delay(50);
  }
}

void loop() 
{
  get_serial();
  update_motors();
}

void get_serial()
{
  if(Serial.available() > 0)
  {
    Order order_received = read_order();

    if (order_received == HELLO)
    {
      if(!is_connected)
      {
        is_connected = true;
        write_order(HELLO);
      }

      else
      {
        // If we are already connected do not send "hello" to avoid infinite loop
        write_order(ALREADY_CONNECTED);
      }
      return;
    }

    else if(order_received == ALREADY_CONNECTED)
    {
      is_connected = true;
    }
    else
    {
      switch(order_received)
      {
        case STOP:
        {
          stop();
          if(DEBUG)
          {
            write_order(STOP);
          }
          break;
        }

        case MOTOR:
        {
          read_direction();
          break;
        }

        //In case of Unknown order:
        default:
          write_order(ERROR);
          return;
      } //End Switch
    }
    write_order(RECEIVED);
  }
}

Order read_order()
{
  return (Order) Serial.read();
}

void wait_for_bytes(int num_bytes, unsigned long timeout)
{
  unsigned long startTime = millis();
  while((Serial.available() < num_bytes) && ((millis() - startTime) < timeout))
  {}
}

void read_signed_bytes(int8_t* buffer, size_t n)
{
  size_t i = 0;
  int c;
  while (i < n)
  {
    c = Serial.read();
    if (c < 0) break;
    *buffer++ = (int8_t) c;
    i++;
  }
}

int32_t read_direction()
{
  int8_t buffer[3];
  wait_for_bytes(3, 200);
  read_signed_bytes(buffer, 3);
  direction.Vx = buffer[0];
  direction.Vy = buffer[1];
  direction.th = buffer[2];
}

void update_motors()
{
  analogWrite(11, direction.Vx);
  analogWrite(10, direction.Vy);
  analogWrite(9, direction.th);
}

void stop()
{
  //TO DO: Detener motores
}

void write_order(enum Order myOrder)
{
  uint8_t* Order = (uint8_t*) &myOrder;
  Serial.write(Order, sizeof(uint8_t));
}

void write_direction()
{
    int8_t buffer[3] = {
        direction.Vx,
        direction.Vy,
        direction.th
    };

    Serial.write((uint8_t*)buffer, 3);
}