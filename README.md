# Recitation 2 - Main

This repository contains code and examples for Recitation 2 of the CS521 course.

## Overview

The main focus of this recitation is to demonstrate the integration of Python with MATLAB using the `matlab.engine` API. This allows for seamless execution of MATLAB scripts and functions from within Python.

## Requirements

- Python 3.10
- MATLAB
- `matlab.engine` for Python

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://gitlab.oit.duke.edu/cs521_24fa/recitation2-main.git
    cd recitation2-main
    ```

2. **Set up a virtual environment:**

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the required Python packages:**

    ```sh
    pip install matlab.engine
    ```

## Troubleshooting

### Common Errors

- **Undefined function or variable:** Ensure that the directory containing the MATLAB script is correctly added to the MATLAB path using `eng.addpath`.

- **MATLAB engine not starting:** Verify that MATLAB is installed and properly configured to work with the `matlab.engine` API.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.