# CS3600 Deep Learning: <br> Homework Assignments

This repo contains the homework assignments for cs3600 - Deep Learning

All rights reserve to Aviv Rosenverg


This document will help you get started with the course homework assignments. Please read it carefully as it contains crucial information.

General
---
he course homework assignments are mandatory and a large part of the grade. They entail writing code in python using popular third-party machine-learning libraries and also theoretical questions.

The assignments are implemented in part on a platform called Jupyter notebooks. Jupyter is a widely-used tool in the machine-learning ecosystem which allows us to create interactive notebooks containing live code, equations and text. We’ll use jupyter notebooks to guide you through the assignments, explain concepts, test your solutions and visualize their outputs.

To install and manage all the necessary packages and dependencies for the assignments, we use conda, a popular package-manager for python. The homework assignments come with an environment.yml file which defines what third-party libraries we depend on. Conda will use this file to create a virtual environment for you. This virtual environment includes python and all other packages and tools we specified, separated from any preexisting python installation you may have. Detailed installation instructions are below. We will not support any other installation method other than the one described.


For working on the code itself, we recommend using PyCharm, however you can use any other editor or IDE if you prefer.

Homework structure
---
Each assignment’s root directory contains the following files and folders:

cs3600: Python package containing course utilities and helper functions. You do not need to edit anything here.

hw#: Python package containing the assignment code. All your solutions will be implemented here, including answers to questions.

Part#_XYZ.ipynb where XYZ is some name: A set of jupyter notebooks that contain the instructions that will guide you through the assignment. You do not need to edit these. However, you should write your name(s) at the beginning of Part0.

main.py: A script providing some utilities via a CLI. You’ll run it to create your submission after completing the assignment.

environment.yml: A file for conda, specifying the third-party packages it should install into the virtual environment it creates. Note that every assignment could define a different environment, but if not speficied, you can use the original one that provided in the main dir.

Environment set-up
---
1. install python3 of miniconda and follow the instructions here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

2. create a virtual env by:

```bash
conda env update -f environment.yml
```


3. activate the enviroment by 
```bash
conda activate cs3600
```


Activating an environment simply means that the path to its python binaries (and packages) is placed at the beginning of your $PATH shell variable. Therefore, running programs installed into the conda env (e.g. python) will run the version from the env since it appears in the $PATH before any other installed version.

To check what conda environments you have and which is active, run
```bash
 conda env list
```

### General Notes:

* You should to do steps 1 (installing conda) once, not for each assignment.

* However, the third-party package dependencies (in the environment.yml file) might slightly change from one assignment to the next. To make sure you have the correct versions, always install the environment again (step 2 above) from the assignment root directory every time a new assignment is published and then activate the environment with the assignment number.

* Always make sure the correct environment is active! It will revert to its default each new terminal session. If you want to change the default env you can add a conda activate in your ~/.bashrc.

* If you use PyCharm or any other IDE, you should configure the interpreter path of the IDE to the path of the python executable within the conda env folder. For example, point the interpreter path to ~/miniconda3/envs/cs236781-hwN/bin/python. This is under Settings -> Project -> Project Interpreter.

* You’ll need to install the conda env within your user folder on the course server. The installation procedure is exactly the same, just follow the instructions for linux.

#### Notes for Windows Users
* On Windows, you can run these commands from the Anaconda Prompt program that is installed with miniconda. If you also add the conda installation to the Windows PATH variable, you can run these commands from the regular windows command prompt.

* Also on Windows, you need to install (Microsoft’s Build Tools for Visual Studio)[https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019] before the conda environment. Make sure “C++ Build Tools” is selected during installation. This only needs to be done once.


Working on the assignment
---
run
```bash
 jupyter lab
```
This will start a jupyter lab server and open your browser at the local server’s url. You can now start working. Open the first notebook (Part0) and follow the instructions.


Implementing your solution and answering questions
The assignment is comprised of a set of notebooks and accompanying code packages.
You only need to edit files in the code package corresponding to the assignment number, e.g. hw1, hw2, etc.
The notebooks contain material you need to know, instructions about what to do and also code blocks that will test and visualize your implementations.
Within the notebooks, anything you need to do is marked with a TODO beside it. It will explain what to implement and in which file.
Within the assignment code package, all locations where you need to write code are marked with a special marker (YOUR CODE). Additionally, implementation guidelines, technical details and hints are in some cases provided in a comment above.
Sometimes there are open questions to answer. Your answers should also be written within the assignment package, not within the notebook itself. The notebook will specify where to write each answer.

Notes:

You should think of the code blocks in the notebooks as tests. They test your solutions and they will fail if something is wrong. As such, if you implement everything and the notebook runs without error, you can be confident about your solution.

You may edit any part of the code, not just the sections marked with YOUR CODE. However, note that there is always a solution which requires editing only within these markers.

When we check your submission, we’ll run the original notebook files of the assignment, together with your submitted code (from the hwN) package. Therefore, any changes you do to the notebook files (such as changing the tests) will not affect the results of our grading. If you rely on notebook modifications to pass the tests, the tests will fail when we grade your work and you will lose marks.

Please don’t put other files in the assignment directory. If you do, they will be added to your submission which is automatically generated from the contents of the assignment folder.

Always make sure the active conda env is cs236781-hwN (where N is the assignment number). If you get strange errors or failing import statements, this is probably the reason. Note that if you close your terminal session you will need to re-activate since conda will use it’s default base environment.


Submitting the assignment
---
What you’ll submit:

All notebooks, after running them clean from start to end, with all outputs present.
An html file containing the merged content of all notebooks.
The code package (hwN), with all your solutions present.
You don’t need to do this manually; we provide you with a helper CLI program to run all the notebooks and combine them into a single file for submission.

Generating your submission file
To generate your submission, run (obviously with different id’s):
```bash
python main.py prepare-submission --id 123456789 --id 987654321
```


The above command will:

1. Execute all the notebooks cleanly, from start to end, regenerating all outputs.
2. Merge the notebook contents into a single html file.
2. Create a zip file with all of the above and also with your code.
If there are errors when running your notebooks, it means there’s a problem with your solution or that you forgot to implement something.

Additionally, you can use the --skip-run flag to skip running your notebooks (and just merge them) in case you already ran everything and you’re sure that all outputs are present:
```bash
python main.py prepare-submission --skip-run --id 123456789 --id 987654321
```


Note however that if some of the outputs are missing from your submission you’ll lose marks.

Note: The submission script must also be run from within the same conda env as the assignment. Don’t forget to activate the env before running the submission script!

#### Submitting a partial solution
If you are unable to solve the entire assignment and wish to submit a partial solution you can create a submission with errors by adding an allow-errors flag, like so:
```bash
python main.py prepare-submission --allow-errors --id 123456789 --id 987654321
```


Uploading the solution
---
simpy upload the zip file to the moodle section of the homework

only a submission generated by the course script is considered valid. Any other submissions, e.g. submitting only the notebooks or the code files will not be graded.

thanks
Course stuff
