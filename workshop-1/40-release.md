## Releasing Your Package

Once your package is built and tested, you need to host the distribution files (`.whl` and `.tar.gz`) somewhere others can access them. Two common methods:

### Option A: Distributing via GitHub Releases

For many projects, especially internal tools or research code, full release to PyPI is unnecessary. Great alternative: use GitHub Releases. This workflow ties your package versions directly to your source control—excellent for reproducibility.

Process integrates `git` with the build process.

1. Commit Your Changes: make sure all your code changes for the new version are committed to your repository

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
