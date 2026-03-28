Description

This project implements different methods for solving Linear Programming (LP) problems, including:

Graphical Method
Simplex Method
Two-Phase Method

It is designed as an academic tool to understand optimization techniques used in operations research and systems engineering.

Features
Graphical solution for 2-variable problems
Simplex method with iterations (tables)
Two-phase method for infeasible problems
Visualization using graphs
Python-based implementation

Technologies Used
Python 3
NumPy
Matplotlib
Pandas
Version control with GitHub

Installation Guide

1. Clone the Repository

Open your terminal (CMD, PowerShell, or VS Code terminal) and run:
git clone https://github.com/your-username/linear-programming-solver.git

Then enter the project folder:
cd linear-programming-solver

2. Create a Virtual Environment (Recommended)
python -m venv venv

Activate the environment:

On Windows:
venv\Scripts\activate

On Mac/Linux:
source venv/bin/activate

3. Install Required Libraries

Install all dependencies with:

pip install -r requirements.txt

 If you don’t have a requirements.txt, install manually:
 pip install numpy matplotlib pandas

 4. Run the Project
    python main.py

  5. Project Structure

     linear-programming-solver/
│── main.py
│── simplex.py
│── two_phase.py
│── graphical_method.py
│── utils.py
│── README.md

