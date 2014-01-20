#!/usr/bin/env python

import os
import copy

from waflib.extras import symwaf2ic
from waflib.extras.gtest import summary


def depends(ctx):
    ctx('sthal', branch='dev_polymorphic_handle')
    ctx('symap2ic', 'src/pylogging')
    ctx('calibtic', 'pycalibtic', branch='dev_sthal')  # sthal does not depend on pycalibtic
    ctx('redman', 'pyredman')
    ctx('redman', 'backends')


def options(opt):
    opt.load('post_task')


def configure(cfg):
    cfg.load('post_task')

    try:
        disable_bindings = cfg.options.disable_bindings
    except AttributeError:
        try:
            disable_bindings = not cfg.options.bindings
        except AttributeError:
            disable_bindings = not cfg.env.build_python_bindings

    if disable_bindings:
        cfg.fatal('Calibration depends on Python bindings.')


def build(bld):
    bld(
        target='pycake',
        source=bld.path.ant_glob('pycake/**/*.py'),
        features='post_task',
        post_task=['pycalibtic', '_pysthal', 'pylogging', 'pyredman', 'redman_xml', 'redman_mongo'],
        install_from='.',
        install_path='lib',
    )
