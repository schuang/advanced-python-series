import numpy as np

class Particle:
    """The parent class for all simulation particles."""
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def advance_position(self, dt):
        """Advances the particle's position based on its velocity."""
        self.position += self.velocity * dt

    def calculate_kinetic_energy(self):
        """Calculates the kinetic energy."""
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)
    
    def describe(self):
        return f"A particle with mass {self.mass:.2f}."


class Atom(Particle):
    """An Atom is a type of Particle. It inherits everything."""
    # We don't need to write anything here yet!
    # It automatically gets __init__, advance_position, etc. from Particle.
    pass

class Ion(Particle):
    """An Ion is a Particle that also has a charge."""
    def __init__(self, mass, position, velocity, charge):
        # --- Calling the Parent's Constructor ---
        # It's crucial to initialize the parent class first.
        # super() refers to the parent class (Particle).
        super().__init__(mass, position, velocity)
        
        # Now, add the new attribute that is unique to the Ion.
        self.charge = charge

    def describe(self):
        # But we can still call the parent's version first using super()
        # This is called EXTENDING the method.
        parent_description = super().describe()
        return f"{parent_description} And it has a charge of {self.charge}."

# Create an instance of the Atom child class
argon_atom = Atom(mass=39.9, position=[0,0,0], velocity=[1,0,0])

# Create an instance of the Ion child class
sodium_ion = Ion(mass=22.9, position=[5,5,5], velocity=[0,1,0], charge=1)

# We can call the method defined in the PARENT class on both objects
argon_atom.advance_position(0.1)
sodium_ion.advance_position(0.1)

print(f"Argon atom new position: {argon_atom.position}")
print(f"Sodium ion new position: {sodium_ion.position}")

# The sodium_ion object has the extra attribute
print(f"Sodium ion charge: {sodium_ion.charge}")