#include <WiFi.h>
#include <PubSubClient.h>
#include "DHTesp.h"
#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <BH1750.h>
#include <ArduinoJson.h>

#define DHT_PIN 4

// -------- WIFI --------
const char* ssid = "iPhone de luffy";
const char* password = "ileff12345";

// -------- MQTT --------
const char* mqtt_server = "172.20.10.4";   // IP de ton PC Node-RED
const int mqtt_port = 1883;
const char* mqtt_topic = "iot/sante/capteurs";

// -------- OBJETS --------
WiFiClient espClient;
PubSubClient client(espClient);

DHTesp dht;

// Second I2C bus
TwoWire I2CBH = TwoWire(1);

Adafruit_MLX90614 mlx = Adafruit_MLX90614();
BH1750 lightMeter;


// -------- WIFI CONNECTION --------
void setup_wifi() {

  Serial.println();
  Serial.print("Connecting to WiFi...");

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}


// -------- MQTT RECONNECT --------
void reconnectMQTT() {

  while (!client.connected()) {
    

    Serial.print("Connecting to MQTT...");

    if (client.connect("ESP32Client")) {
      Serial.println("connected");
    }
    else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retry in 5 seconds");
      delay(5000);
    }
  }
}


// -------- SETUP --------
void setup() {

  Serial.begin(115200);

  dht.setup(DHT_PIN, DHTesp::DHT22);

  // MLX90614 I2C
  Wire.begin(21,22);

  // BH1750 I2C
  I2CBH.begin(27,26);

  if (!mlx.begin(0x5A, &Wire)) {
    Serial.println("Error MLX90614");
    while(1);
  }

  if (!lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE,0x23,&I2CBH)) {
    Serial.println("Error BH1750");
    while(1);
  }

  Serial.println("Sensors ready");

  // WiFi
  setup_wifi();
  Serial.println(WiFi.localIP());

  // MQTT
  client.setServer(mqtt_server, mqtt_port);
  if (client.connected()) 
    Serial.println("MQTT connected");
}


// -------- LOOP --------
void loop() {

Serial.println(WiFi.localIP());
  if (!client.connected()) {
    reconnectMQTT();
  }

  client.loop();

  // Read sensors
  TempAndHumidity data = dht.getTempAndHumidity();
  float ambient = mlx.readAmbientTempC();
  float objectT = mlx.readObjectTempC();
  float lux = lightMeter.readLightLevel();

  // JSON creation
  StaticJsonDocument<256> doc;

  doc["humidity"] = data.humidity;
  doc["temperature"] = data.temperature;
  doc["ambientTemp"] = ambient;
  doc["objectTemp"] = objectT;
  doc["light"] = lux;

  char buffer[256];
  serializeJson(doc, buffer);

  // Publish MQTT
  client.publish(mqtt_topic, buffer);

  // Debug
  Serial.print("Data sent: ");
  Serial.println(buffer);


  delay(3000);
}
