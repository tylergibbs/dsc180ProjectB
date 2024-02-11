import subprocess
import sys
from glob import glob

#For Paper Replications Synthetic data
import logging

#For Real World Datasets
import cdt

#For Early Stopping Analyses
import argparse



def clean():
    subprocess.run(['rm', 'output/*'], stdout = sys.stdout, check=True, text=True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('cmd', choices=commands.keys())

    return parser.parse_args()

def all():
    """Dynamicaly gets and runs all comands in comands"""
    for k in commands:
        if (k != 'all'):
           print("running:{}".format(k))
           commands[k]()

def comareisons():
    pass

def replication():
    pass

def real_dataset():
    pass


commands = {
'all': all,
'compare':comareisons,
'replicate':replication,
'real_data':real_dataset,
}

def run():
    parser = get_args()

    cmd = parser.cmd
    print(cmd)
    commands[cmd]()



if __name__ == '__main__':
   run()
