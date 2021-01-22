# :mortar_board: Eplogic Web Service

Web Service for Anti-HLA antibody target prediction via machine learning.


## :gear: Installation

Run the command to install the packages:

`pip install -r requirements.txt`

## :computer: How to get .exe

1º Use command:

`pyi-makespec -F eplogic_local.py`

this will create a file by name eplogic_local.spec

2º Find line with:

`hidden imports=[]`

in the .spec file.

3º Replace it with:

`hiddenimports = ['sklearn.ensemble._forest']`

4º Add these two lines at beginning of the .spec file:

`import sys`
`sys.setrecursionlimit(5000)`

to increase recursion limit

5º Run:

`pyinstaller eplogic_local.spec`

At the end of all these steps, there will be three new folders:

`__pycache__`
`build`
`dist`

the .exe will be in the dist folder.