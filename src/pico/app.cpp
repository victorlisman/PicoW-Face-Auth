#include <WiFi.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include <LittleFS.h>
#include <Adafruit_GFX.h>              
#include <Adafruit_ST7735.h>           
#include <SPI.h>
#include <ArduinoJson.h>

#define TFT_CS      9     
#define TFT_RST     8
#define TFT_DC      7
#define TFT_SCK     10
#define TFT_MOSI    11
#define TFT_MISO
#define BUZZER_PIN  6
#define BUTTON_PIN  17

Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCK, TFT_RST);
int textxoffset = 5;     
int textyoffset = 7;    

int tft_line1 = 0;

const char* ssid = "pass";
const char* password = "pass";

const char* serverUrl = "http://192.168.0.72:5000/predict";
const char* filePath = "/test.jpg";

WiFiClient client;
HTTPClient http;

char responseMessage[512];

unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;
int lastButtonState = LOW;
int buttonState = LOW;

void setup() {
    Serial.begin(115200);
    delay(1000); 
    Serial.println("Starting setup...");

    tft.initR(INITR_BLACKTAB);                   
    tft.fillScreen(ST7735_BLACK);                     
    tft.setRotation(0);                             
    tft.setTextWrap(true);   
    tft.setTextColor(ST7735_WHITE);
    tft.setTextSize(1);
    tft.setCursor(20, 20);

    pinMode(BUTTON_PIN, INPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    responseMessage[0] = '\0';

    if (!LittleFS.begin()) {
        Serial.println("An Error has occurred while mounting LittleFS");
        return;
    } else {
        Serial.println("LittleFS mounted successfully");
    }

    Serial.print("Connecting to ");
    Serial.println(ssid);
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("Connected to WiFi");

    if (sendImage(serverUrl, filePath)) {
        Serial.println("File uploaded and response received successfully");

        if (responseMessage[0] != '\0') {
            tft.fillRect(10, 10, 100, 50, ST7735_GREEN);
            tft.setCursor(20, 20);
            tft.print(responseMessage);
        }
    } else {
        Serial.println("File upload failed");
        tft.setCursor(20, 20);
        tft.fillRect(10, 10, 100, 50, ST7735_BLUE);
    }
}

void loop() {
    int reading = digitalRead(BUTTON_PIN);

    if (reading != lastButtonState) {
        lastDebounceTime = millis();
    }

    if ((millis() - lastDebounceTime) > debounceDelay) {
        if (reading != buttonState) {
            buttonState = reading;

            if (buttonState == HIGH) {
                Serial.println("Button pressed, resending image...");
                if (sendImage(serverUrl, filePath)) {
                    Serial.println("File uploaded and response received successfully");

                    if (responseMessage[0] != '\0') {
                        tft.fillRect(10, 10, 100, 50, ST7735_GREEN);
                        tft.setCursor(20, 20);
                        tft.print(responseMessage);
                    }
                } else {
                    Serial.println("File upload failed");
                    tft.setCursor(20, 20);
                    tft.fillRect(10, 10, 100, 50, ST7735_BLUE);
                }
            }
        }
    }

    lastButtonState = reading;
}

bool sendImage(const char* serverUrl, const char* filePath) {
    Serial.println("Opening file...");
    File imageFile = LittleFS.open(filePath, "r");
    if (!imageFile) {
        Serial.println("Failed to open file for reading");
        return false;
    }

    int fileSize = imageFile.size();
    Serial.print("File size: ");
    Serial.println(fileSize);

    uint8_t* imageData = (uint8_t*)malloc(fileSize);
    imageFile.read(imageData, fileSize);
    imageFile.close();

    http.begin(client, serverUrl);
    http.addHeader("Content-Type", "image/jpeg");

    Serial.println("Sending HTTP POST request...");
    int httpResponseCode = http.POST(imageData, fileSize);
    free(imageData);

    delay(10000);

    if (httpResponseCode > 0) {
        Serial.print("HTTP Response code: ");
        Serial.println(httpResponseCode);
        String response = http.getString();
        Serial.println("Response from server:");
        Serial.println(response);

        Serial.print("Response length: ");
        Serial.println(response.length());

        if (response.length() == 0) {
            Serial.println("Received empty response from server.");
            return false;
        }

        DynamicJsonDocument doc(1024);
        DeserializationError error = deserializeJson(doc, response);
        if (error) {
            Serial.print("Failed to parse JSON: ");
            Serial.println(error.c_str());
            return false;
        }

        const char* prediction = doc["prediction"];
        float confidence = doc["confidence"];
        if (prediction) {
            if (confidence > 0.5) {
              snprintf(responseMessage, sizeof(responseMessage), "Prediction: \n %s, \n Confidence: %.2f", prediction, confidence);
            }
            else {
              snprintf(responseMessage, sizeof(responseMessage), "Person not recognized");
            }
        } else {
            Serial.println("No 'prediction' field in JSON response");
            return false;
        }

        digitalWrite(BUZZER_PIN, HIGH);
        http.end();
        return true;
    } else {
        Serial.print("Error code: ");
        Serial.println(httpResponseCode);
        http.end();
        return false;
    }
}
