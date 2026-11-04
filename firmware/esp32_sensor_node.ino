/*
  Aqua-Swachh: ESP32-S3 Sensor Node Firmware
  =========================================
  Target: ESP32-S3 DevKit
  Sensors:
  - TDS In (Raw Water): Analog ADC (e.g. GPIO 1)
  - TDS Out (Permeate): Analog ADC (e.g. GPIO 2)
  - Temperature: DS18B20 1-Wire (e.g. GPIO 4)
  - Flow Sensor: YF-S201 Hall-Effect (e.g. GPIO 5)

  Serial Output Format:
  tds_in,tds_out,temperature,flow_rate
  e.g.: 420.5,32.1,26.4,1.82
*/

#include <OneWire.h>
#include <DallasTemperature.h>

// --- Pin Definitions ---
const int PIN_TDS_IN     = 1;   // ADC1 Channel for Feed TDS
const int PIN_TDS_OUT    = 2;   // ADC1 Channel for Permeate TDS
const int PIN_ONE_WIRE   = 4;   // 1-Wire bus for DS18B20
const int PIN_FLOW       = 5;   // Interrupt pin for YF-S201 flow sensor

// --- Sensor Objects ---
OneWire oneWire(PIN_ONE_WIRE);
DallasTemperature tempSensors(&oneWire);

// --- Flow Sensor Variables ---
volatile unsigned long pulseCount = 0;
unsigned long lastFlowTime = 0;
float flowRateLPM = 0.0;
// YF-S201 calibration factor: ~7.5 pulses per second = 1 L/min
const float FLOW_CALIBRATION_FACTOR = 7.5;

void IRAM_ATTR pulseCounter() {
  pulseCount++;
}

// Helper: Convert ADC voltage to TDS (ppm) with temperature compensation
float readTDS(int pin, float temperatureC) {
  int analogVal = analogRead(pin);
  // ESP32 ADC: 12-bit (0-4095) with reference voltage ~3.3V
  float voltage = (analogVal / 4095.0) * 3.3;
  
  // Standard Gravity TDS temperature compensation formula:
  // V_comp = V / (1.0 + 0.02 * (T - 25.0))
  float compensationCoefficient = 1.0 + 0.02 * (temperatureC - 25.0);
  float compensationVoltage = voltage / compensationCoefficient;
  
  // Typical empirical polynomial curve for analog TDS meters (Xcluma / Gravity)
  float tdsValue = (133.42 * pow(compensationVoltage, 3) 
                  - 255.86 * pow(compensationVoltage, 2) 
                  + 857.39 * compensationVoltage) * 0.5;
  if (tdsValue < 0) tdsValue = 0;
  return tdsValue;
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Initialize DS18B20
  tempSensors.begin();

  // Initialize Flow Sensor Interrupt
  pinMode(PIN_FLOW, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW), pulseCounter, FALLING);
  lastFlowTime = millis();

  // Configure ADC
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db); // Full range 0 - 3.3V
}

void loop() {
  unsigned long now = millis();
  
  // Sample every 1000 ms (1 Hz)
  if (now - lastFlowTime >= 1000) {
    // 1. Calculate Flow Rate
    detachInterrupt(digitalPinToInterrupt(PIN_FLOW));
    // flowRate (L/min) = (pulseCount / 7.5) * (1000 / dt_ms)
    float dt = (now - lastFlowTime) / 1000.0;
    flowRateLPM = (pulseCount / (FLOW_CALIBRATION_FACTOR * dt)) / 60.0 * 60.0; // in L/min
    pulseCount = 0;
    lastFlowTime = now;
    attachInterrupt(digitalPinToInterrupt(PIN_FLOW), pulseCounter, FALLING);

    // 2. Read Water Temperature
    tempSensors.requestTemperatures();
    float waterTemp = tempSensors.getTempCByIndex(0);
    // Fallback if sensor disconnected
    if (waterTemp < -50 || waterTemp > 85) {
      waterTemp = 25.0; // Default reference
    }

    // 3. Read TDS Sensors
    float tdsIn = readTDS(PIN_TDS_IN, waterTemp);
    float tdsOut = readTDS(PIN_TDS_OUT, waterTemp);

    // 4. Output CSV formatted stream over Serial
    // Format: tds_in,tds_out,temperature,flow_rate
    Serial.print(tdsIn, 1);
    Serial.print(",");
    Serial.print(tdsOut, 1);
    Serial.print(",");
    Serial.print(waterTemp, 1);
    Serial.print(",");
    Serial.println(flowRateLPM, 2);
  }
}
