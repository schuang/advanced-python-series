from abc import ABC, abstractmethod

# 1. Inherit from ABC to mark this as an abstract base class.
class ForceField(ABC):
    """
    An abstract base class (a contract) for all force fields.
    It cannot be instantiated directly.
    """
    def __init__(self, parameters):
        self.parameters = parameters

    # 2. Use the @abstractmethod decorator.
    # This declares that any concrete child class MUST implement this method.
    @abstractmethod
    def calculate_energy(self, positions):
        """Calculates the potential energy of the system."""
        pass


class LennardJones(ForceField):
    """A concrete implementation for the Lennard-Jones potential."""
    
    # We MUST implement this method, or Python will raise an error.
    def calculate_energy(self, positions):
        # In a real scenario, this would be a complex calculation.
        print("Calculating energy using the Lennard-Jones potential...")
        return np.sum(positions**2) # Placeholder

class Coulomb(ForceField):
    """A concrete implementation for the Coulomb potential."""
    
    # We also MUST implement this method.
    def calculate_energy(self, positions):
        print("Calculating energy using the Coulomb potential...")
        return np.sum(np.abs(positions)) # Placeholder
    
def run_simulation(force_field: ForceField, positions):
    """Runs a simulation step using a given force field."""
    print(f"\n--- Running simulation with {force_field.__class__.__name__} ---")
    # It doesn't know the specific type, it just trusts the contract.
    energy = force_field.calculate_energy(positions)
    print(f"Calculated Energy: {energy:.2f}")


if __name__ == "__main__":
    import numpy as np

    lj = LennardJones(parameters={'sigma': 1.0, 'epsilon': 0.1})
    coulomb = Coulomb(parameters={})
    # Imagine we wrote a third class, AI_Potential, that also inherits from ForceField

    run_simulation(lj, np.random.rand(10,3))
    run_simulation(coulomb, np.random.rand(10,3))

    