---
title: "**Foundations for Sustainable Research Software**"
author: "Shao-Ching Huang"
date: 2025-10-09
---



Workshop series:

1. Foundations for sustainable research software (October 9)

2. Scaling your science with parallel computing (October 16)

3. Accelerating your code with GPUs (October 30)




Resources:

- Workshop materials [[link]](https://github.com/schuang/advanced-python-series)

- UCLA Office of Advanced Research Computing [[link]](https://oarc.ucla.edu/contact)

- Hoffman2 Cluster [[link]](https://www.hoffman2.idre.ucla.edu)



# Introduction

## Observation

- Python has become a dominant language for scientific computing and machine learning

- Widespread use of Notebooks has led to proliferation of "linear" scripts that are difficult to reuse

- Advanced Python features for software reuse remain underutilized in research code

- Many Python codebases are limited to single-core execution

- Multi-core CPUs are standard, and GPU resources are increasingly accessible


## Software Sustainability

- Keeps research code usable and maintainable over time

- Facilitates collaboration and sharing across research teams

- Prevents code from becoming obsolete or unusable

- Enables efficient updates, bug fixes, and feature enhancements

- Ensures reproducibility and transparency in scientific workflows

- Distinguishes sustainable software from disposable one-time scripts


## General Goals of This Workshop Series

- Transform ad-hoc scripts into well-structured, reusable software

- Teach best practices for writing maintainable research code

- Demonstrate how to package and distribute code effectively

- Emphasize reproducibility in computational workflows

- Provide hands-on examples for scientific computing applications

- Leverage modern hardware: multi-core CPUs and GPUs

- Empower researchers to build robust, sustainable software


## Why Learn This in the LLM Era?

Even with powerful LLM-based code-generating assistants, understanding software design principles remains critical:

- You provide the human intelligence

  - LLMs need clear, structured prompts that reflect good design thinking
  
  - You must specify what architecture, patterns, and structure you want
  
  - Without understanding concepts like classes, inheritance, or packaging, you can't guide the LLM effectively

- You must read and validate generated code
  
  - LLMs make mistakes, introduce bugs, and sometimes generate outdated or inefficient patterns
  
  - You need to recognize when code follows best practices vs. anti-patterns
  
- You maintain and evolve the codebase
  
  - Generated code needs to be integrated, refactored, and maintained over time
  
  - You make architectural decisions that LLMs cannot: "Should this be a class or a function?" "How should components interact?"
  
  - Long-term project success depends on human oversight of structure and design



## Transitioning from Scripts to Software

A Staged Approach (using Python as an example)

- **Refactor linear scripts into functions**
  
  - Each function does one thing (single responsibility principle)
  
  - Use descriptive, meaningful function names
  
  - Add clear comments and documentation

- **Group related functions into classes**

- **Organize code into modules** by topic or workflow

- **Practice is essential** — mastering this skill takes time


Bottom line: for sustainable code, organize logic into functions rather than linear scripts.



