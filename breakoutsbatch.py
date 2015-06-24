""" Launches statistics.py script in batch mode. Statistics is calculated
    for several (min_change, max_pullback) values and is stored into separate
    appropriately named files.
"""

import os
import sys
import shutil
import argparse
import subprocess
import logging
from itertools import product


logging.basicConfig(filename="statistics.log", level=logging.DEBUG)


def generate_command(change_range, pullback_range, args):
    """ Yields commands for statistics.py script with different values
        of pullback and change parameters
    """
    template = "python statistics.py -i {input} -o {output} --min-change " \
               "{change} --max-pullback {pullback}{threaded} --breakouts-only"

    multiplier = 0.0001

    for (min_change, max_pullback) in product(change_range, pullback_range):
        logging.debug("[.] Statistics calculation for min_change=%s / "
                      "max_pullback=%s was launched", min_change, max_pullback)
        fn = args["output"]
        fn = fn.replace("xx", str(min_change)).replace("yy", str(max_pullback))
        min_change *= multiplier
        max_pullback *= multiplier
        yield template.format(input=args["input"], output=fn,
                              change=min_change, pullback=max_pullback,
                              threaded=" -t" if args["threaded"] else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--max-change-low", type=int, default=10,
                        help="start point for maximum price change range")
    parser.add_argument("--max-change-high", type=int, default=20,
                        help="end point for maximum price change range")
    parser.add_argument("--min-pullback-low", type=int, default=10,
                        help="start point for pullback range")
    parser.add_argument("--min-pullback-high", type=int, default=20,
                        help="end point for pullback")
    parser.add_argument("-i", "--input", type=str, default='linked.csv')
    parser.add_argument("-o", "--output", type=str,
                        default='breakout_minchange_xx_pullback_yy.csv')
    parser.add_argument("-t", "--threaded", action="store_true", default='true')
    args = vars(parser.parse_args())

    max_change_low = args["max_change_low"]
    max_change_high = args["max_change_high"]
    min_pullback_low = args["min_pullback_low"]
    min_pullback_high = args["min_pullback_high"]

    change_range = list(range(max_change_low, max_change_high + 1))
    pullback_range = list(range(min_pullback_low, min_pullback_high + 1))
    commands = generate_command(change_range, pullback_range, args)

    if os.path.exists("img"):
        logging.warning("[!] Warning: folder with images will be rewritten")
        shutil.rmtree("img")

    for cmd in commands:
        logging.debug("[.] Calculation for command: %s" % cmd)
        try:
            p = subprocess.Popen(cmd)
        except WindowsError as e:
            print("[!] Cannot launch command. Trying another interpreter...")
            try:
                cmd = cmd.replace("python", "python3")
                p = subprocess.Popen(cmd)
            except Exception as e:
                print("[-] Unexpected error: %s" % str(e))
                print("[-] Failed command: %s" % cmd)
                sys.exit(1)
        p.wait(timeout=10000)
        logging.debug("[+] Successful")

    logging.debug("[!] Processing ended")

