# MA-4625-Final

## Making a virtual environment
`python` OR `python3` `-m venv <name of venv> ` 

## Activating the Virtual Environment

POSIX (Mac/Linux): 
`source <name of venv>/bin/activate`

Windows:
`<name of venv>/Scripts/activate`

## Installing Requirements

`pip install -r requirements.txt`
 
## Setup Script (fetches the data and splits it into train, test, and split):

POSIX (Mac/Linux):
`sh` or `.` `setup.py`

Windows:
`setup.bat`

## Notes
Please add your venv to the gitignore if it's not already present. If you don't, when you push to git it will send all the packages in the venv, which is something we don't want. 
