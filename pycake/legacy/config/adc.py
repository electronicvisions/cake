import pycalibtic
import pyhalbe

ADC_BOARD_ID = 25  # needed to load calibration data for this board id
INPUT_CHANNEL = pyhalbe.Coordinate.ChannelOnADC(3)  # which pin on is connected on the ADC board?


class SampleTime(object):
    '''Define constant values for calibration sample times [us].

    These values were stripped out of the calibration ADC interface.'''

    min = 100
    max = 100
    mean = 100
    spikes = 10000


def init_adc_calibration_backend():
    '''Initialize the backend which contains ADC calibration data.

    The data will be used to convert raw ADC values to voltages.'''

    backend = pycalibtic.loadBackend(pycalibtic.loadLibrary("libcalibtic_mongo.so"))
    backend.config("host", "cetares")
    backend.config("collection", "adc")
    backend.init()

    adc_id = pyhalbe.Coordinate.ADC(ADC_BOARD_ID)
    adc_calib = pycalibtic.ADCCalibration()
    adc_calib.load(backend, adc_id)

    return adc_calib
