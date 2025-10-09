# Python Packaging: Turning Your Code into an Installable Tool

So far, we've focused on structuring code using classes to make it robust and reproducible. But how do you share that code with a colleague or the wider scientific community? Emailing a `.py` file is a start, but not a professional or reliable solution. Recipient won't know what dependencies are needed, what versions are compatible, and they'll have to manage the file manually.

This is where packaging comes in. Goal: bundle your Python code into a standardized format that can be easily installed, managed, and distributed. Gold standard: make your tool installable with a simple command: `pip install your-tool-name`.

This section walks you through this process in three parts:

1. Part 1: Creating Your First Package. Turn a local script into a properly structured project you can install and test on your own machine
2. Part 2: Sharing Your Package. Two common ways to distribute your package: via private GitHub repository or public Python Package Index (PyPI)
3. Part 3: The Development Cycle. How to release new versions of your package as you make improvements


### Recall: Classes vs. Modules in Python

Classes:

- A class defines a blueprint for creating objects that bundle data (attributes) and behavior (methods).

- Ideal for representing complex entities or workflows where state and operations are tightly coupled (e.g., a simulation, a data processor).

- Supports encapsulation, inheritance, and polymorphism, enabling more advanced software design.

- Classes are defined within modules and can be reused across projects.

- Classes bundle related data and methods for more complex tasks.
  
Modules:

- A module is a single Python file, or a collection of files, that groups related functions, variables, and classes.

- Useful for organizing code by topic, workflow, or functionality (e.g., `math.py`, `io_utils.py`).

- Promotes code reuse and separation of concerns.

- Modules are imported using the `import` statement, making code easy to share and maintain.





## Part 1: Creating Your First Package (v0.1.0)

Our first goal is to take a standalone Python file and turn it into a real, installable package. We will start by creating version `0.1.0`.

### Step 1.1: A Standard Project Structure

The Python community has adopted a standard layout for projects that is recognized by all modern packaging tools. The most important convention is to place your actual source code inside a dedicated directory.

Here is the recommended structure:

```
heat1d-project/
├── src/
│   └── heat1d/
│       ├── __init__.py
│       └── heat1d_class.py
├── tests/
│   └── test_heat1d.py
├── pyproject.toml
└── README.md
```

Let's break this down:

*   `heat1d-project/`: The root directory of your project.
*   `src/`: A directory that contains your Python source code. This separation is a modern best practice that prevents many common packaging problems.
*   `src/heat1d/`: This is your actual **Python package**. The name of this directory is what users will use when they `import` your code.
*   `src/heat1d/__init__.py`: This file is often empty. Its presence tells Python that the `heat1d` directory is a package.
*   `src/heat1d/heat1d_class.py`: Your actual Python code file.
*   `pyproject.toml`: This is the most important file. It's the modern, standardized file for configuring your project, telling Python's build tools everything they need to know to package your code.

### Step 1.2: The `pyproject.toml` Configuration File

This file is the heart of your package. It contains all the metadata about your project. Using the TOML (Tom's Obvious, Minimal Language) format, it is designed to be easy for humans to read and write.

Here is a simple but complete `pyproject.toml` for our project:

```toml
# pyproject.toml

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "heat1d-simulation"
version = "0.1.0"
authors = [
  { name="Your Name", email="your.email@example.com" },
]
description = "A simple 1D heat equation solver."
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "numpy",
    "matplotlib",
]

[project.urls]
"Homepage" = "https://github.com/your-username/heat1d-project"
"Bug Tracker" = "https://github.com/your-username/heat1d-project/issues"
```

Let's break down the key sections:

*   `[build-system]`: This section is mostly boilerplate. It tells `pip` what tools are needed to build your package (in this case, `setuptools`). You can usually just copy-paste this.
*   `[project]`: This is where you define your project's metadata.
    *   `name`: The name that will be used on PyPI (the Python Package Index). This is what people will type when they run `pip install ...`.
    *   `version`: The current version of your package. You will update this number whenever you release a new version.
    *   `dependencies`: This is a critical field. It lists all the other packages that your code needs to run (like `numpy` and `matplotlib`). When a user installs your package, `pip` will automatically find and install these dependencies for them.

### Step 1.3: Building Your Package

Once you have your project structure and your `pyproject.toml` file, you are ready to build the package. This process creates the actual files that will be distributed.

First, you need to install the standard Python build tool:
```bash
pip install build
```

Now, from the root of your project directory (`heat1d-project/`), run the build command:
```bash
python -m build
```

This command creates a new directory called `dist/`. Inside, you'll find two important files:

- `heat1d_simulation-0.1.0-py3-none-any.whl`: wheel file. Pre-built, binary distribution format that makes installation very fast for end-user. Preferred format
- `heat1d_simulation-0.1.0.tar.gz`: source distribution (sdist). Compressed archive of your source code. `pip` can use this as fallback if there isn't a compatible wheel file available

### Step 1.4: Installing and Testing Your Package Locally

Before you share your package with the world, you should test it on your own machine. You can install the wheel file you just created directly using `pip`.

Navigate out of your project directory and install it:
```bash
cd ..
pip install heat1d-project/dist/heat1d_simulation-0.1.0-py3-none-any.whl
```

To overwrite an existing installation, use
```
pip install --force-reinstall dist/heat1d_simulation-0.1.0-py3-none-any.whl 
```


Your package is now installed in your Python environment, just like any other package. You can now open a Python interpreter or a script from any directory on your system and use it:

```python
python run-heat.py
```

At this point, you have a fully working local package. The next step is to share it with others.

