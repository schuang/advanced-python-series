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

## Software Sustainability

- Ensures research code remains usable and maintainable over time

- Facilitates collaboration and sharing with other scientists

- Reduces risk of code becoming obsolete

- Enables easier updates, bug fixes, and feature additions

- Supports reproducibility and transparency in scientific results

- Distinguish one time "thrown-away" script from sustainable software


## General Goals of This Workshop Series

- Transition from ad-hoc scripts to well-structured, reusable software

- Teach (or encourage) best practices for writing maintainable code

- Introduce tools for version control, testing, and documentation

- Demonstrate how to package code for sharing and reuse

- Emphasize reproducibility in computational workflows

- Provide hands-on examples relevant to scientific computing

- Embrace modern computer architecture with parallel computing (multi-core, GPU)

- Empower researchers to build robust, sustainable software for science


## Why Learn This in the LLM Era?

Even with powerful LLM-based coding assistants, understanding software design principles remains critical:

- **You provide the human intelligence**
  - LLMs need clear, structured prompts that reflect good design thinking
  - You must specify what architecture, patterns, and structure you want
  - Without understanding concepts like classes, inheritance, or packaging, you can't guide the LLM effectively

- **You must read and validate generated code**
  - LLMs make mistakes, introduce bugs, and sometimes generate outdated or inefficient patterns
  - You need to recognize when code follows best practices vs. anti-patterns
  - Understanding the principles lets you spot problems before they become technical debt

- **You maintain and evolve the codebase**
  - Generated code needs to be integrated, refactored, and maintained over time
  - You make architectural decisions that LLMs cannot: "Should this be a class or a function?" "How should components interact?"
  - Long-term project success depends on human oversight of structure and design

- **Scientific integrity requires understanding**
  - You are responsible for the correctness of your computational research
  - Blindly trusting generated code without understanding it is as risky as using a statistical method you don't comprehend
  - Peer reviewers and collaborators expect you to explain and justify your code design choices

- **LLMs amplify your expertise, not replace it**
  - With strong fundamentals, you can use LLMs to code 10x faster
  - Without fundamentals, LLMs produce code you can't debug, maintain, or trust
  - The goal: use AI as a force multiplier for your knowledge, not a substitute


## Transitioning from Scripts to Software

A Staged Approach (using Python as an example)

- Group "linear" scripts into functions 
  - Each function should do one thing (single responsibility principle).
  - Name the functions properly
  - Code comments + documentation

- Group related functions together into classes

- Modules help organize functions and classes by topic or workflow.

- It takes a lot of practice to get it right.



### Classes vs. Modules in Python

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


