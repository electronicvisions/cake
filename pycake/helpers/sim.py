import os
import hashlib
import numpy
import scipy.interpolate
import pandas
import cPickle
from collections import defaultdict

import pylogging
import pyhalco_hicann_v2
from pyhalco_common import X
from pyhalco_hicann_v2 import NeuronOnHICANN

from pycake.helpers.sthal import StHALContainer
from pycake.helpers.sthal import FakeAnalogRecorder

from sims.sim_denmem_lib import NETS_AND_PINS
from sims.sim_denmem_lib import TBParameters
from sims.sim_denmem_lib import run_remote_simulation


class SimStHALContainer(StHALContainer):
    """Contains StHAL objects for hardware access. Multiple experiments can
    share one container.

    Attributes:
        remote_host: hostname of simulation server
        remote_port: port of simulation server
        hicann_version: hicann version to simulate (2,4)
        mc_seed: Monte carlo seed (None disables MC)
        maximum_spikes: abort simulation after this many spikes (default=200)
        spike_counter_offset: start counting spikes after this time (default
                              simulation_init_time)
    """
    logger = pylogging.get("pycake.helper.simsthal")

    def __init__(self,
                 config,
                 coord_analog=pyhalco_hicann_v2.AnalogOnHICANN(0),
                 recording_time=30.0e-6,
                 resample_output=True):
        """Initialize StHAL. kwargs default to vertical setup configuration.

        Args:
            config: Config instance
            recording_time: ADC recording time in seconds
            wafer_cfg: ?
            resample_output: bool
                if True, resample the result of read_adc to the adc
                sampling frequency
        """
        super(SimStHALContainer, self).__init__(
            config, coord_analog, recording_time)
        self.current_neuron = pyhalco_hicann_v2.NeuronOnHICANN()
        host, port = config.get_sim_denmem().split(':')
        self.remote_host = host
        self.remote_port = int(port)

        self.resample_output = resample_output

        # 10 times simulation reset
        self.simulation_init_time = 10.0e-07
        self.set_recording_time(recording_time, 1)

        self.logger.INFO("Using sim_denmem on {}:{}".format(
            self.remote_host, self.remote_port))

        self.simulation_cache = config.get_sim_denmem_cache()
        if self.simulation_cache and not os.path.isdir(self.simulation_cache):
            raise RuntimeError("simulation_cache must be a folder")
        self.mc_seed = config.get_sim_denmem_mc_seed()
        self.adc = FakeAnalogRecorder()

        maximum_spikes = config.get_sim_denmem_maximum_spikes()
        if maximum_spikes is None:
            self.maximum_spikes = 200
        else:
            self.maximum_spikes = maximum_spikes

        self.spike_counter_offset = self.simulation_init_time

        self.simulated_configurations = defaultdict(list)

    def connect(self):
        """Connect to the hardware."""
        pass

    def connect_adc(self, coord_analog=None):
        """Gets ADC handle.

        Args:
            coord_analog: pyhalco_hicann_v2.AnalogOnHICANN to override default behavior
        """
        pass

    def disconnect(self):
        """Free handles."""
        pass

        self.adc = None

    def write_config(self, configurator=None):
        """Write full configuration."""
        pass

    def switch_analog_output(self, coord_neuron, l1address=None):
        super(SimStHALContainer, self).switch_analog_output(
            coord_neuron, l1address)
        self.current_neuron = coord_neuron

    def resample_simulation_result(self, adc_result,
                                   adc_sampling_interval):
        """Re-sample the result of self.read_adc().

        adc_result: Dictionary as returned by self.read_adc(). Must
            contain the key "t". All values must be numpy arrays of
            same length.

        adc_sampling_interval: float. adc sampling interval in
            seconds.

        Returns a dictionary with the same keys as `adc_result`.
            self.resample_simulation_result(data)['t'] covers the
            linear range between min(data['t']) and max(data['t']) in
            steps corresponding to the adc sampling interval.

            For x != 't', self.resample_simulation_result(data)[x] is
            the interpolated value of data[x] at the corresponding
            time values.
        """
        time = numpy.arange(min(adc_result['t']), max(adc_result['t']),
                            adc_sampling_interval)

        if len(adc_result['t']) < 2:
            raise ValueError("simulated ADC output too short")

        self.logger.info(
            "resampling simulation adc_result from {} to {} samples".format(
                len(adc_result['t']), len(time)))
        self.logger.info(
            "new sampling interval is {} s".format(
                adc_sampling_interval))

        resampled = dict()
        signals = adc_result.keys()
        signals.remove('t')

        for signal in signals:
            resampled[signal] = scipy.interpolate.interp1d(
                adc_result['t'],
                adc_result[signal])(time)

        return pandas.DataFrame(resampled, index=time)

    def read_adc(self):
        """Fake ADC readout by evaluating a denmem_sim run."""

        param = self.build_TBParameters(self.current_neuron)
        json = param.to_json()
        result = self.run_simulation(self.current_neuron, param)

        if self.resample_output:
            return self.resample_simulation_result(
                result, adc_sampling_interval=1.0/self.ideal_adc_freq)
        else:
            return result

    def read_adc_and_spikes(self):
        return self.read_adc()

    def get_simulation_neurons(self, neuron):
        """Returns the two neurons, that will be simulated for a given neuron"""
        if int(neuron.x()) % 2 == 0:
            return neuron, NeuronOnHICANN(X(int(neuron.x()) + 1), neuron.y())
        else:
            return NeuronOnHICANN(X(int(neuron.x()) - 1), neuron.y()), neuron

    def build_TBParameters(self, neuron):
        """Returns the serialized TBParameters for self.current_neuron"""
        left, right = self.get_simulation_neurons(neuron)
        param = TBParameters.from_sthal(self.wafer, self.coord_hicann, left,
                                        self.simulation_init_time,
                                        self.spike_counter_offset)
        param.simulator_settings.simulation_time = self.recording_time
        param.simulator_settings.nets_to_save = NETS_AND_PINS.ALL
        param.simulator_settings.hicann_version = self.hicann_version
        param.simulator_settings.max_spike_count = self.maximum_spikes

        if self.mc_seed is not None:
            mc_run = int(neuron.toEnum())/2 + 1
            param.simulator_settings.set_mc_run(self.mc_seed, mc_run)

        # set the unused neuron to "harmless" parameters
        idx = (int(neuron.x()) + 1) % 2
        nparams = [
            ('El', 0.7),
            ('Vt', 1.2),
            ('Igl', 2.0e-6),
            ('Vreset', 0.7),
            ('Vconvoffi', 1.7),
            ('Vconvoffx', 1.7)
        ]
        fg = param.floating_gate_parameters
        for p, v in nparams:
            tmp = list(fg[p])
            tmp[idx] = v
            fg[p] = tuple(tmp)

        tmp = list(param.digital_parameters['activate_firing'])
        tmp[idx] = False
        param.digital_parameters['activate_firing'] = tuple(tmp)

        return param

    def run_simulation(self, neuron, param):
        """Execute a remote simulation for the given json set"""
        # TODO Error handling
        json = param.to_json()
        json_hash = None
        left, right = self.get_simulation_neurons(neuron)

        if self.simulation_cache:
            json_hash = hashlib.new('sha256')
            json_hash.update(json)
            json_hash = os.path.join(
                self.simulation_cache, json_hash.hexdigest())

        if json_hash and os.path.isfile(json_hash):
            self.logger.info("load result from cache: {}".format(json_hash))
            with open(json_hash) as infile:
                json_loaded, lresult, rresult = cPickle.load(infile)
                assert json_loaded == json
        else:
            self.logger.info("Run simulation on {}:{}".format(
                self.remote_host, self.remote_port))
            lresult, rresult = run_remote_simulation(
                param, self.remote_host, self.remote_port,
                init_time=self.simulation_init_time)

            if json_hash:
                self.logger.info("cache result in {}".format(json_hash))
                with open(json_hash, 'w') as outfile:
                    data = (json, lresult, rresult)
                    cPickle.dump(data, outfile, cPickle.HIGHEST_PROTOCOL)

        if neuron == left:
            self.simulated_configurations[left].append(param)
            return lresult
        else:
            self.simulated_configurations[right].append(param)
            return rresult

    def read_adc_status(self):
        return "FAKE ADC :P"

    def read_wafer_status(self):
        return self.wafer.status()

    def set_recording_time(self, recording_time, _=None):
        """Sets the recording time of the ADC.

        The recording_time should be as small as theoretical required for the
        measurement. And the repetitions factor should be the amount you need
        to cope with readout noise. This is to speed up simulations.

        To speed up simulations this implementation ignores repetitions factor!
        """
        self.recording_time = recording_time + self.simulation_init_time
        self.spike_counter_offset = self.simulation_init_time

    def set_neuron_size(self, n):
        self.logger.ERROR("Neuron size other than 1 not supported! Using size 1")

    def get_neuron_size(self):
        return 1

    def switch_current_stimulus_and_output(self, coord_neuron, l1address=None):
        def do_nothing(*args, **kwargs):
            pass
        self.wafer.configure = do_nothing
        super(SimStHALContainer, self).switch_current_stimulus_and_output(coord_neuron, l1address=l1address)

        # remove non-C++ entry for pickling
        del self.wafer.__dict__['configure']

    def read_floating_gates(self, parameter):
        return pandas.DataFrame()

    def is_hardware(self):
        return False
