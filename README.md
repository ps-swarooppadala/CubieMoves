# CubieMoves

This project solves a Rubik's Cube using OpenCV libraries. 

1.Clone the repository
```
git clone https://github.com/simbo1905/CubieMoves
```
2.  Create an virtual environment using
```
python3 -m venv  virtual environment name.
```
3.  Run the following to install the requirements
```
pip3 install -r requirements.txt
```
4.  Run the following to solve using the 'U.png', 'D.png', etc files in the 'solve0' subdirectory
```
python3 main.py solve0
```

If it is possible to solve then it will print out the solution and popup some image windows snowing how it found the colors. On my mac these windows don't seem to foreground I have to clock on the python icon in my task bar and select 'Show All Windows' and I can see all the windows. The program awaits any key being pressed into any window. So click on one window and hit the spacebar to quit the app. 
