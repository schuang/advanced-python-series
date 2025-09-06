# Python Packaging: Turning Your Code into an Installable Tool

So far, we have focused on how to structure your code using classes to make it robust and reproducible. But how do you share that code with a colleague, or with the wider scientific community? Emailing a `.py` file is a start, but it's not a professional or reliable solution. The recipient won't know what dependencies are needed, what versions are compatible, and they will have to manage the file manually.

This is where **packaging** comes in. The goal of packaging is to bundle your Python code into a standardized format that can be easily installed, managed, and distributed. The gold standard is making your tool installable with a simple command: `pip install your-tool-name`.

This tutorial will walk you through this process in three parts:
1.  **Part 1: Creating Your First Package.** We will turn a local script into a properly structured project that you can install and test on your own machine.
2.  **Part 2: Sharing Your Package.** We will cover two common ways to distribute your package: via a private GitHub repository or on the public Python Package Index (PyPI).
3.  **Part 3: The Development Cycle.** We will show you how to release new versions of your package as you make improvements.

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

This command will create a new directory called `dist/`. Inside, you will find two important files:
*   `heat1d_simulation-0.1.0-py3-none-any.whl`: This is a **wheel** file. It's a pre-built, binary distribution format that makes installation very fast for the end-user. This is the preferred format.
*   `heat1d_simulation-0.1.0.tar.gz`: This is a **source distribution** (sdist). It's a compressed archive of your source code. `pip` can use this as a fallback if there isn't a compatible wheel file available.

### Step 1.4: Installing and Testing Your Package Locally

Before you share your package with the world, you should test it on your own machine. You can install the wheel file you just created directly using `pip`.

Navigate out of your project directory and install it:
```bash
cd ..
pip install heat1d-project/dist/heat1d_simulation-0.1.0-py3-none-any.whl
```

That's it! Your package is now installed in your Python environment, just like any other package. You can now open a Python interpreter or a script from any directory on your system and use it:

```python
from heat1d.heat1d_class import Heat1D
sim = Heat1D(nx=50)
print(sim)
```

At this point, you have a fully working local package. The next step is to share it with others.

## Part 2: Sharing Your Package

Once your package is built and tested, you need to host the distribution files (`.whl` and `.tar.gz`) somewhere others can access them. We will cover two common methods.

### Option A: Distributing via GitHub Releases

For many projects, especially internal tools or research code, a full release to PyPI is unnecessary. A great alternative is to use GitHub Releases. This workflow ties your package versions directly to your source control, which is excellent for reproducibility.

The process integrates `git` with the build process.

1.  **Commit Your Changes:** Make sure all your code changes for the new version are committed to your repository.

2.  **Tag the Version:** Create a `git` tag that matches the version number in your `pyproject.toml`. This permanently marks that specific commit as that version. It's common to use tags like `v0.1.0`.
    ```bash
    # Create a tag for version 0.1.0
    git tag v0.1.0
    
    # Push the tag to GitHub
    git push origin v0.1.0
    ```

3.  **Create a GitHub Release:**
    *   Go to your repository on GitHub.
    *   Click on the "Releases" link on the right-hand side.
    *   Click "Draft a new release".
    *   In the "Choose a tag" dropdown, select the tag you just pushed (e.g., `v0.1.0`).
    *   Give the release a title (e.g., "Version 0.1.0") and write a description of the changes.
    *   In the "Attach binaries" section, drag and drop the `.whl` and `.tar.gz` files you created in the `dist/` directory.
    *   Click "Publish release".

Your package is now available on GitHub.

### How Users Install from GitHub

A user can now install your package directly from your GitHub release using `pip`. They just need the URL to the wheel file.

1.  The user goes to your GitHub Releases page.
2.  They find the release they want (e.g., `v0.1.0`).
3.  They right-click on the `.whl` file and copy its URL.
4.  They use that URL with `pip install`:
    ```bash
    pip install https://github.com/your-username/heat1d-project/releases/download/v0.1.0/heat1d_simulation-0.1.0-py3-none-any.whl
    ```
`pip` will download the file and install it along with any dependencies, just as if it were coming from PyPI. This method is a powerful way to share installable versions of your code without needing a public package index.

### Option B: Distributing on the Python Package Index (PyPI)

The final step for sharing your work with the entire community is to upload it to the Python Package Index (PyPI). This is the official central repository where `pip` looks for packages.

1.  **Create an Account:** Register for an account on [pypi.org](https://pypi.org/).
2.  **Install the Upload Tool:**
    ```bash
    pip install twine
    ```
3.  **Upload Your Distribution Files:**
    ```bash
    twine upload dist/*
    ```
    `twine` will prompt you for your PyPI username and password.

Once the upload is complete, your package is live! Anyone in the world can now install it by simply running:
```bash
pip install heat1d-simulation
```

## Part 3: The Development Cycle: Releasing a New Version

Once your package is released, you will inevitably want to improve it. The process of releasing an update is a cycle of coding, versioning, and re-distributing.

Let's say you want to release version `0.2.0`.

1.  **Make Your Code Changes:** Add your new features or fix bugs in the source code.
2.  **Update the Version Number:** This is a critical step. Open your `pyproject.toml` file and increment the `version` number.
    ```diff
    - version = "0.1.0"
    + version = "0.2.0"
    ```
    It's common to follow a system called [Semantic Versioning](https://semver.org/) (Major.Minor.Patch).
3.  **Re-run the Build:**
    ```bash
    python -m build
    ```
    This will create new `heat1d_simulation-0.2.0-py3-none-any.whl` and `heat1d_simulation-0.2.0.tar.gz` files in your `dist/` directory.
4.  **Distribute the New Files:** Repeat the steps from Part 2 to share your new version.
    *   **For GitHub:** Create a new git tag (`v0.2.0`), push it, and create a new GitHub Release, uploading the new distribution files.
    *   **For PyPI:** Run `twine upload dist/*` again to upload the new version. PyPI will automatically handle the version update.

This cycle ensures that users can reliably access specific versions of your code and understand what has changed between releases.

## Key Takeaways

*   **Packaging** is the process of turning your Python code into a standardized, installable format.
*   The process can be broken down into three stages: **creating** the package locally, **sharing** it via a platform like GitHub or PyPI, and **updating** it by releasing new versions.
*   A modern Python project should use a `src/` layout and be configured with a `pyproject.toml` file, which defines its name, version, and dependencies.
*   Use the `build` library (`python -m build`) to create distribution files (wheels and source distributions).
*   Always test your package locally with `pip` before distributing it.
*   To release a new version, you must update the `version` number in `pyproject.toml`, re-build, and then re-distribute the new files.

By following these steps, you can transform your scientific scripts into professional, reusable, and easily shareable software packages. This is a critical skill for ensuring your computational research is reproducible and accessible to others.
