#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import random
import string
import os
from pycake.helpers.misc import mkdir_p


class TestMkdir(unittest.TestCase):
    """Create random directory which did not exist,
    create it again to make sure this does not cause issues,
    remove it afterwards."""

    def setUp(self):
        """generate random directory name which does not exist"""
        def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
            return "".join(random.choice(chars) for _ in range(size))

        # generate random directory name
        randdir = os.path.join("/tmp", id_generator())
        while(os.path.exists(randdir)):
            # generate another one until it does not exist
            randdir = os.path.join("/tmp", id_generator())

        self.randdir = randdir

    def test_mkdir(self):
        """create directory twice"""
        randdir = self.randdir
        self.assertFalse(os.path.exists(randdir))
        mkdir_p(randdir)
        self.assertTrue(os.path.isdir(randdir))

        # assert this does not raise
        mkdir_p(randdir)

    def tearDown(self):
        # cleanup
        try:
            os.rmdir(self.randdir)
        except:
            pass


if __name__ == "__main__":
    unittest.main()
