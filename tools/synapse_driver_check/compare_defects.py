#!/usr/bin/env python

import argparse
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('files', type=argparse.FileType('r'), nargs="+", help="at least one defects summary file (result of ana_defects.py)")
    args = parser.parse_args()

    file_a_dict = json.load(args.files[0])
    bad_a = set(file_a_dict["bad_drivers"])

    print "file A: {}".format(args.files[0].name)
    print
    print "bad in {}:\n{}".format("A", sorted(list(bad_a)))

    for f in args.files[1:]:

        file_b_dict = json.load(f)

        bad_b = set(file_b_dict["bad_drivers"])

        print
        print "file B: {}".format(f.name)
        print
        print "\tbad in {}:\n\t{}".format("B", sorted(list(bad_b)))
        print
        print "\tbad in {} but not in {}:\n\t{}".format("A", "B", sorted(list(bad_a.difference(bad_b))))
        print
        print "\tbad in {} but not in {}:\n\t{}".format("B", "A", sorted(list(bad_b.difference(bad_a))))
        print
        print "\tbad in A and in B:\n\t", sorted(list(bad_a.intersection(bad_b)))
        print
        print "\tbad in A or B:\n\t", sorted(list(bad_a.union(bad_b)))
