#!/usr/bin/env python

import unittest
import numpy as np
import pycake.helpers.peakdetect


class TestPeakDetect(unittest.TestCase):
    """Test the functionality of peak detection."""

    def test_peak(self):
        """Run peak detection on fake data, check that peaks are detected."""
        peakdet = pycake.helpers.peakdetect.peakdet
        series = [0, 0, 0, 2, 0, 0, 0, -2, 0, 0, 0, 2, 0, 0, 0, -2, 0]
        maxtab, mintab = peakdet(series, .3)

        # maximum at index 3 and 11, value 2
        np.testing.assert_array_equal(maxtab, np.array([[3, 2], [11, 2]]))

        # minimum at index 7 and 15, value -2
        np.testing.assert_array_equal(mintab, np.array([[7, -2], [15, -2]]))

    def test_exceptions(self):
        """Run peak detection using wrong arguments.
        Make sure that the correct exception is raised."""
        peakdet = pycake.helpers.peakdetect.peakdet

        v = [1, 2, 3, 4]
        x = [1, 2, 3]

        # different array size
        self.assertRaises(TypeError, peakdet, v, 0.1, x)

        # non-scalar delta
        self.assertRaises(TypeError, peakdet, v, (2, 3))

        # negative delta
        self.assertRaises(ValueError, peakdet, v, -0.1)


if __name__ == '__main__':
    unittest.main()
