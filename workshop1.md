
# Workshop 1: Foundations for Sustainable Research Software

**Part of the series:** *From Scripts to Software: Practical Python for Reproducible Research*

This hands-on workshop will introduce you to the principles of modern software development and show you how to apply them to your scientific projects. We will move beyond simple scripting and learn how to write Python code that is robust, reusable, and ready for the challenges of modern research. While the examples are in Python, the underlying principles of software design are universal and apply to any programming language.

### Learning Objectives

*   Structure scientific code using classes and objects.
*   Apply inheritance and polymorphism to create flexible and reusable code.
*   Use decorators to add functionality like timing and logging.

## Part 1: The "Why" of OOP: From Chaos to Clarity

We often start with simple scripts that grow into unmanageable chaos. This section will motivate the need for a more structured approach to programming.

*   **The Problem:** We will look at examples of chaotic research code, which are often hard to read, maintain, and reproduce.
    *   See examples: [01_chaotic_research_code.py](workshop-1-examples/01_chaotic_research_code.py), [03_graduate_student_chaos.py](workshop-1-examples/03_graduate_student_chaos.py)

*   **The Solution:** We will see how Object-Oriented Programming (OOP) can help us organize our code into logical, reusable components.
    *   See examples: [02_organized_research_code.py](workshop-1-examples/02_organized_research_code.py), [04_class_solution.py](workshop-1-examples/04_class_solution.py)

## Part 2: The "How" of OOP: Core Concepts with the Heat Equation

Now, let's dive into the core concepts of OOP using our first "golden example": a 2D heat equation solver.

*   **Base Classes:** We will start with a base class that defines a common interface for all our solvers.
    *   See example: [19_pde_solver_base.py](workshop-1-examples/19_pde_solver_base.py)

*   **Inheritance:** We will create specialized solver classes that inherit functionality from our base class.
    *   See example: [20_pde_solver_inheritance.py](workshop-1-examples/20_pde_solver_inheritance.py)

*   **Polymorphism:** We will see how we can use different solver objects interchangeably, making our code more flexible.
    *   See example: [21_pde_solver_polymorphism.py](workshop-1-examples/21_pde_solver_polymorphism.py)

## Part 3: OOP in Practice: A Deep Learning Example

To show the versatility of OOP, we will now apply the same principles to our second "golden example": a simple neural network for image classification.

*   **The Problem:** First, we'll look at a non-OOP implementation of a neural network, which is hard to modify and reuse.
    *   See example: [22_deep_learning_chaos.py](workshop-1-examples/22_deep_learning_chaos.py)

*   **The Solution:** Then, we will refactor the code into an object-oriented structure with classes for the network and its layers.
    *   See examples: [23_deep_learning_oop.py](workshop-1-examples/23_deep_learning_oop.py), [24_deep_learning_training.py](workshop-1-examples/24_deep_learning_training.py)

## Part 4: Pythonic OOP: Writing Elegant Code

Python provides several features that make writing OOP code more elegant and concise.

*   **Dataclasses:** We will learn how to use dataclasses to reduce boilerplate code in our classes.
    *   See examples: [06_traditional_class_boilerplate.py](workshop-1-examples/06_traditional_class_boilerplate.py), [07_dataclass_basic.py](workshop-1-examples/07_dataclass_basic.py), [08_dataclass_advanced.py](workshop-1-examples/08_dataclass_advanced.py)

*   **Decorators:** We will see how decorators can add functionality like timing and logging to our functions without cluttering the core logic.
    *   See examples: [16_repetitive_code_problem.py](workshop-1-examples/16_repetitive_code_problem.py), [17_decorator_solution.py](workshop-1-examples/17_decorator_solution.py), [18_research_decorators.py](workshop-1-examples/18_research_decorators.py)

## Part 5: High-Performance Python: A Glimpse into `ctypes`

For performance-critical sections of your code, you can integrate Python with lower-level languages like C or C++.

*   **`ctypes`:** We will briefly introduce the `ctypes` library, which allows you to call functions in shared libraries directly from Python. This is an important tool for leveraging existing high-performance libraries.

## Next Steps

You are now ready to proceed to **Workshop 2: Scaling Your Science with Parallel Computing** to learn about scaling up your code to multiple processors using MPI (Massively passing interface).
