import argparse
from report.snp100 import snp100
from report.graphs import graphs

import report.testing_dagmanl as testing_dagmanl
import report.testing_two as testing_two
import report.testing_one as testing_one
import report.testing as testing
import report.testing_dagmats as testing_dagmats

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

def generate_data():
    testing.test_all_methods(output_dir='testing01.jsonl')
    testing_one.test_all_methods(output_dir='testing02.jsonl')
    testing_dagmats.test_all_methods(output_dir='testing06.jsonl')
    testing_two.test_all_methods(output_dir='results/testing07.jsonl')
    testing_dagmanl.test_all_methods(output_dir='testing08.jsonl')

commands = {
'all': all,
'snp100':snp100,
'data':generate_data,
'graphs':graphs,

}

def run():
    parser = get_args()

    cmd = parser.cmd
    print(cmd)
    commands[cmd]()



if __name__ == '__main__':
   run()
