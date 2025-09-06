# 06_simple_class.py

# A simple class representing a sensor
class Sensor:
    def __init__(self, name, location):
        self.name = name
        self.location = location
        self.reading = 0.0

    def take_reading(self, value):
        """Simulates taking a new reading."""
        self.reading = value
        print(f"{self.name} at {self.location} reads {self.reading}")

# Create an instance of the Sensor class
sensor1 = Sensor("TempSensor1", "Lab A")

# Use the object's method
sensor1.take_reading(22.5)
