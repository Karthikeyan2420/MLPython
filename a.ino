
#include <Servo.h>

// Front wheel
const int wheelPin = 10;
Servo myWheel;

// motor pins
const int leftBackward = 3;
const int leftForward = 5;
const int rightBackward = 6;
const int rightForward = 11;


// Pins used for the sonic sensor - the eyes
const int pingPin = 7;
const int echoPin = 8;

const int FULL_SPEED = 255;  // 255 is maximum speed, you can lower this number to slow your bot down
const int WHEEL_CENTER = 55;  // CHANGE THIS DEPENDING ON HOW YOU ATTACHED YOUR FRONT WHEEL - IN THEORY 90 IS MIDDLE
const int TURN_AMOUNT = 20; // CHANGE THIS TO MAKE THE ROBOT TURN A SHARPER CORNER


void setup() {
  Serial.begin(9600);
  myWheel.attach(wheelPin);
  
  pinMode(pingPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  pinMode(leftForward, OUTPUT);
  pinMode(leftBackward, OUTPUT);
  pinMode(rightForward, OUTPUT);
  pinMode(rightBackward, OUTPUT);

}

void loop() {
   
   

   // comment this line in to run a drive mode with front servo wheel
  runDriveMode();

 
}



void runDriveMode()
{
   // basic drive loop
   int distance = getDistanceToWall();
  Serial.println(distance);
   if (distance < 20)
   {
      turnRandom(2);
      moveBackward();  
      delay(random(1500));  
   }
   else if (distance < 50)
   {
      turnRandom(2);
   }
   else if (distance <80)
   {
      turnRandom(1);
   }
   else
   {
      myWheel.write(WHEEL_CENTER);
      moveForward();
   }
 
  delay(200);
}



// Motor movement functions
void moveForward(int s)
{
  analogWrite(leftBackward, 0);
  analogWrite(rightBackward, 0);
  analogWrite(leftForward, s);
  analogWrite(rightForward, s);  
}

void moveBackward(int s)
{
  analogWrite(leftForward, 0);
  analogWrite(rightForward, 0);  
  analogWrite(leftBackward, s);
  analogWrite(rightBackward, s);  
}



// turn using front wheel only
void turnRight(int factor)
{
  myWheel.write(WHEEL_CENTER+TURN_AMOUNT* factor);
 
}

void turnLeft(int factor)
{
  myWheel.write(WHEEL_CENTER-TURN_AMOUNT* factor);
 
}

void turnRight()
{
  turnRight(1);
}

void turnLeft()
{
  turnLeft(1);
}

void turnRandom(int factor)
{
  if (random(2) == 0)
    turnLeft(factor);
  else
    turnRight(factor);
}

void moveForward()
{
   moveForward(FULL_SPEED);
}
void moveBackward()
{
  moveBackward(FULL_SPEED);
}


// Returns the number centimeters to the closest object in front of our robot.
int getDistanceToWall()
{
  digitalWrite(pingPin, LOW);
  delayMicroseconds(2);
  digitalWrite(pingPin, HIGH);
  delayMicroseconds(5);
  digitalWrite(pingPin, LOW);
  
  // read the duration in time
  long duration = pulseIn(echoPin, HIGH);
  
   // convert the time into a distance
  float cm = microsecondsToCentimeters(duration);
  
   
  return cm;
}

int microsecondsToCentimeters(long duration)
{  
  int microseconds = duration/29/2 ; // distance = speed * time ( or you could go time divided by microseconds per centimeter - have a think about it.....  
  return microseconds;
}




