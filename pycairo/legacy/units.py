"""Units for variables, using unum.

Example:
    import pycake.units as units

    i = 5*units.uA
    print i.asUnit(units.nA)
    print i.asNumber()
    print i.asUnit(units.nA).asNumber()
"""

from unum import Unum
import unum.units

mV = Unum.unit('mV', 1e-3 * unum.units.V, 'millivolt')

nF = Unum.unit('nF', 1e-9 * unum.units.F, 'nanofarad')
pF = Unum.unit('pF', 1e-12 * unum.units.F, 'picofarad')

nS = Unum.unit('nS', 1e-9 * unum.units.S, 'nanosiemens')

ms = unum.units.ms
us = unum.units.us

uA = unum.units.uA
nA = unum.units.nA
