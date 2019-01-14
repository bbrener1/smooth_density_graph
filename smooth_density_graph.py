import numpy as np
import sys
import os
import tempfile as tmp
from pathlib import Path
import io
import argparse
from time import sleep
import subprocess as sp
from threading import Thread
from queue import Queue, Empty
# import asyncio as aio
import numpy as np
# import matplotlib.pyplot as plt

def main():
    print("Running main")
    counts = np.loadtxt(sys.argv[1])
    fit_predict(counts,scaling=.1,sample_sub=10)

def fit_predict(targets,command,distance=None,verbose=False,steps=None,subsample=None,k=None,processors=None,backtrace=False):

    # np.array(targets)
    targets = targets.astype(dtype=float)

    targets = "\n".join(["\t".join([str(y) for y in x]) for x in targets]) + "\n"


    input_temp = tmp.NamedTemporaryFile()
    progress_temp = tmp.NamedTemporaryFile()
    # final_pos_temp = tmp.NamedTemporaryFile()

    input_writer = open(input_temp.name,mode='w')
    input_writer.write(targets)
    input_writer.close()

    # print(targets)

    # table_pipe = io.StringIO(targets)
    #
    # table_pipe.seek(0)

    path_to_rust = (Path(__file__).parent / "target/release/smooth_density_graph").resolve()

    print("Running " + str(path_to_rust))

    arg_list = []
    if backtrace:
        arg_list.append("RUST_BACKTRACE=1")
    arg_list.extend([str(path_to_rust),command])
    arg_list.extend(["-c",input_temp.name])
    # arg_list.extend(["-stdin"])
    # arg_list.extend(["-stdout"])
    # if verbose:
    #     arg_list.extend(["-verbose"])
    if steps is not None:
        arg_list.extend(["-steps",str(steps)])
    if subsample is not None:
        arg_list.extend(["-ss",str(subsample)])
    if k is not None:
        arg_list.extend(["-k",str(k)])
    if distance is not None:
        arg_list.extend(["-d",str(distance)])
    arg_list.extend(["2>"+progress_temp.name])

    print("Command: " + " ".join(arg_list))

    # print("Peek at input:")
    # print(open(input_temp.name,mode='r').read())

    # cp = sp.Popen(arg_list,stdin=sp.PIPE,stdout=sp.PIPE,stderr=sp.PIPE)
    cp = sp.Popen(" ".join(arg_list),stdout=sp.PIPE,universal_newlines=True,shell=True)

    # cp.stdin.write(targets.encode())

    # cp = sp.run(arg_list,input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True)
    # cp = sp.run(" ".join(arg_list),input=targets,stdout=sp.PIPE,stderr=sp.PIPE,universal_newlines=True,shell=True)

    # while True:
    #     # sleep(0.1)
    #     rc = cp.poll()
    #     if rc is not None:
    #         for line in cp.stderr.readlines():
    #             sleep(.001)
    #             print(line)
    #         break
    #     output = cp.stderr.readline()
    #     # print("Read line")
    #     print(output.strip())
    #

    progress_counter = 0

    while cp.poll() is None:
        sleep(.01)
        line = progress_temp.readline()
        if verbose and line != b"":
            print(line,flush=True)
            # print(line.count(b's:'),flush=True)
        #     progress_counter += line.count(b's:')
        #     print(f"Points descended:{progress_counter}",flush=True)
        # else:
        # if not verbose:
        #     progress_counter += line.count(b's:')
        #     if b"Clusters" not in line:
        #         print(f"Points descended:{progress_counter}",end="\r",flush=True)
        #     else:
        #         print(str(line.strip), end='\r')
        # if line != b"":
        #     print(line)
        # else:
        #     print(cp.returncode)

    print("Broke loop")

    for line in progress_temp.readlines():
        print(line)

    # print(cp.stdout.read())

    # return(list([float(x) for x in cp.stdout.read().split()]))

    return(list(map(lambda x: float(x),cp.stdout.read().split())))
