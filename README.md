# CS 521, 321, Math 462
## Recitation #2 - (more) MATLAB

### Objective:
In this recitation, you will deepen your hands-on experience with MATLAB, focusing on practicing basic operations and learning how to integrate MATLAB with Python using the `matlab.engine` package. This approach will allow you to run MATLAB commands from Python, enhancing flexibility in technical computing. You’ll gain practical skills necessary for more complex tasks in upcoming projects and homework, with an alternative option to work fully in Python if preferred.

Complete as much of this as you can during recitation. If you run out of time, please complete the rest at home.

> **Note (to repeat one more time):** The auto-magic power of VS code or JetBrains will not be here to help you. But it does not mean you need to remember every detailed command line we used ---- LLMs are always ready to help you. But what you do need is understanding from the top level (to give clear instructions as “leader”).

---

### 1. First, let’s get a repo for you.

1. **Fork the repository:**

    - Go to the [Recitation 2 - Main](https://gitlab.oit.duke.edu/cs521_24fa/recitation2-main) repository on GitLab.
    - Click the **Fork** button in the top-right corner.
    - Select your personal namespace as the destination for the fork.

2. **Clone the repository:**

    - Go to your forked repository on GitLab.
    - Click the **Clone** button and copy the URL.
    - Open a terminal and run the following commands:

        ```sh
        git clone <URL>  # Replace <URL> with the copied URL
        cd recitation2-main
        ```

3. **Set up a virtual environment:**

    - Run the following commands in the terminal:

        ```sh
        python -m venv .venv
        source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
        ```
      
    - **Note:** The python version used in the virtual environment should be 3.10.
      
4. **Install the required Python packages:**

    - Run the following command in the terminal:

        ```sh
        pip install -r requirements.txt
        ```

---

### 2. Get the course files (Mcode and datasets).

1. **Make `update_cf.sh` executable:**

    - Run the following command in the terminal:

        ```sh
        chmod +x update_cf.sh
        ```

2. **Run the script:**

    - Run the following command in the terminal:

        ```sh
        ./update_cf.sh
        ```

    - This script will download the course files and place them in the `Mcode` and `data-sets` directory.
    - **Note:** Later you can rerun this script to get the latest course files.

---

### 3. Test your `matlab.engine` installation.

1. **Open a jupyter notebook (Optional):**

    - Run the following command in the terminal:

        ```sh
        jupyter notebook
        ```

    - This will open a new tab in your default web browser.
    - Navigate to the `main.ipynb` file and open it.

2. **Run the notebook (or run `main.py`):**

    - Execute the cell, you should see `MATLAB Engine for Python successfully started` as the output.

---

### 4. Complete the tasks in the notebook.

1. **Follow the instructions in the notebook:**

    - Complete the tasks in the notebook by running the cells.
    - Make sure to understand the code and the output.
    - Feel free to experiment with the code and ask questions.

---

