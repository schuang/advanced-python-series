class Animal:
        def __init__(self, age):
            self.age = age
        
        def report_age(self):
            return f"I am {self.age} years old."
        
class Mammal(Animal):
        def __init__(self, age, fur_color):
            # Initialize the parent class to set the age
            super().__init__(age)
            self.fur_color = fur_color
            
        def describe_fur(self):
            return f"I have {self.fur_color} fur."
        
class Dog(Mammal):
        def __init__(self, age, fur_color, breed):
            # Initialize the parent class (Mammal)
            # The Mammal's __init__ will in turn initialize the Animal class.
            super().__init__(age, fur_color)
            self.breed = breed
            
        def bark(self):
            return "Woof!"
        
if __name__ == "__main__":

    my_dog = Dog(age=5, fur_color='brown', breed='Golden Retriever')

    # Method from the Dog class
    print(my_dog.bark())

    # Method from the Mammal class
    print(my_dog.describe_fur())

    # Method from the Animal class
    print(my_dog.report_age())

    # Attributes from all levels
    print(f"Breed: {my_dog.breed}, Fur: {my_dog.fur_color}, Age: {my_dog.age}")  