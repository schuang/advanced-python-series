# Python Packaging

- **Why Package Your Code?**
  - Share code professionally and reliably
  - Avoid manual file management and dependency confusion
  - Make your tool installable with `pip install ...`

- **Packaging Workflow: 3 Parts**
  1. **Create Your First Package**
     - Use a standard project structure: `src/`, `tests/`, `pyproject.toml`, `README.md`
     - Place code in `src/your_package/`
     - `pyproject.toml` defines name, version, dependencies
  2. **Share Your Package**
     - Distribute via GitHub Releases (private or public)
     - Or upload to PyPI for global access
     - Users install with `pip install <URL or package-name>`
  3. **Release New Versions**
     - Update code and version in `pyproject.toml`
     - Rebuild and redistribute
     - Use semantic versioning: Major.Minor.Patch

- **Key Files & Directories**
  - `src/your_package/`: Source code
  - `__init__.py`: Marks a directory as a Python package
  - `pyproject.toml`: Project metadata and dependencies
  - `tests/`: Unit tests
  - `dist/`: Built distribution files (`.whl`, `.tar.gz`)

- **Building & Installing Locally**
  - Install build tool: `pip install build`
  - Build package: `python -m build`
  - Install locally: `pip install dist/your_package-<version>-py3-none-any.whl`

- **Sharing Options**
  - **GitHub Releases:**
    - Tag version with `git tag vX.Y.Z`
    - Upload built files to release
    - Users install via direct URL
  - **PyPI:**
    - Register at pypi.org
    - Upload with `twine upload dist/*`
    - Users install with `pip install your-package-name`

- **Development Cycle**
  - Make changes, update version, rebuild, redistribute
  - Always test locally before sharing

- **Key Takeaways**
  - Packaging = reproducible, installable, shareable code
  - Use `src/` layout and `pyproject.toml` for modern projects
  - Distribute via GitHub or PyPI
  - Update version for every release
  - Critical for reproducible and accessible scientific software
