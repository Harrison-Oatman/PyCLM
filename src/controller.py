"""
The controller is the brain of the feedback loop.

It is responsible for
- managing timing
- passing messages between processes
- scheduling microscope events
"""
from abc import ABCMeta, abstractmethod
from time import time
from .queues import AllQueues
from .events import AcquisitionEvent, UpdatePatternEvent
from .experiments import experiment_from_toml, Position, ExperimentSchedule, TimeCourse
from .datatypes import AcquisitionData, CameraPattern, EventSLMPattern
from .messages import Message, UpdatePatternEventMessage, AcquisitionEventMessage
from h5py import File
from pathlib import Path
import numpy as np
from cv2 import warpAffine


class DataPassingProcess(metaclass=ABCMeta):
    def __init__(self, aq: AllQueues):
        self.all_queues = aq

        self.message_history = []

        self.from_manager = None
        self.data_in = None

    def process(self):

        while True:

            if not self.from_manager.empty():
                msg = self.from_manager.get()

                must_break = self.handle_message(msg)

                if must_break:
                    break

            if not self.data_in.empty():
                data = self.data_in.get()

                if isinstance(data, Message):
                    must_break = self.handle_message(data)

                    if must_break:
                        break

                assert isinstance(data, AcquisitionData), \
                    f"Unexpected data type: {type(data)}, expected AcquisitionData"

                self.handle_data(data)

    @abstractmethod
    def handle_data(self, data):
        pass

    def handle_message(self, msg):

        self.message_history.append(msg)

        match msg.message:

            case "close":
                return True

            case _:
                raise ValueError(f"Unexpected message: {msg}")


class MicroscopeOutbox(DataPassingProcess):
    # grabs data from microscope, writes data to disk

    def __init__(self, aq: AllQueues, base_path: Path = Path().cwd()):
        super().__init__(aq)

        self.from_manager = aq.manager_to_outbox
        self.data_in = aq.acquisition_outbox

        self.manager = aq.outbox_to_manager

        self.seg_queue = aq.segmentation_queue
        self.pattern_queue = aq.raw_pattern

        self.base_path = base_path

    def handle_data(self, data):
        aq_event = data.event

        if aq_event.save_output:
            self.write_data(data)

        if aq_event.segment:
            self.seg_queue.put(data)

        if aq_event.updates_pattern:
            self.pattern_queue.put(data)

    def handle_message(self, msg):

        self.message_history.append(msg)

        match msg.message:

            case "close":
                return True

            case _:
                raise ValueError(f"Unexpected message: {msg}")

    def write_data(self, data: AcquisitionData):
        aq_event = data.event

        file_relpath, relpath = aq_event.get_rel_path()

        filepath = self.base_path / file_relpath
        with File(filepath, "w") as f:
            dset = f.create_dataset(relpath + r"\data", data=data.data)

            print(dset)

    def test_write_data(self):
        aq_event = AcquisitionEvent("test", Position(1, 2, 0), scheduled_time=0, exposure_time_ms=1,
                                    sub_axes=[0, "test"])
        data = AcquisitionData(aq_event, np.random.rand(100, 100))

        self.write_data(data)


class SLMBuffer(DataPassingProcess):

    def __init__(self, aq: AllQueues):
        super().__init__(aq)

        self.from_manager = aq.manager_to_slm_buffer
        self.data_in = aq.pattern_to_slm

        self.manager = aq.slm_buffer_to_manager

        self.to_microscope = aq.slm_to_microscope

        self.slm_patterns = {}

        self.slm_shape = None
        self.affine_transform = None

        self.initialized = False

    def initialize(self, shape, affine_transform, experiment_names):
        """
        :param shape: Tuple of (width, height) for the SLM pattern
        :param affine_transform: 2x3 array for affine transformation to apply to pattern
        :param experiment_names: List of experiment names to initialize patterns for
        """

        self.slm_shape = shape
        self.affine_transform = np.array(affine_transform)

        assert affine_transform.shape == (2, 3), "Affine transform must be a 2x3 matrix"

        # Initialize patterns for each experiment
        for name in experiment_names:
            slm_pattern = np.zeros(self.slm_shape, dtype=np.uint8)  # Initialize a blank pattern
            self.slm_patterns[name] = (0, slm_pattern)  # Store the pattern in the dictionary

        self.initialized = True

    def pattern_to_slm(self, pattern: np.ndarray):
        """
        This function takes a pattern and applies the stored affine transformation
        :param pattern: np array of type float scaled from 0-1, in coordinates of camera
        :return: at_slm_pattern: np array of type uint8, transformed to SLM coordinates
        """
        assert self.initialized, "SLMBuffer must be initialized before converting patterns"

        return warpAffine(np.round(pattern * 255).astype(np.uint8), self.affine_transform, self.slm_shape)

    def handle_data(self, data: CameraPattern):

        pattern = data.data
        pattern_id = data.pattern_id

        experiment_name = data.experiment

        slm_pattern = self.pattern_to_slm(pattern)

        # Store the pattern in the dictionary
        if experiment_name in self.slm_patterns:
            # set the current pattern and id
            self.slm_patterns[experiment_name] = (pattern_id, slm_pattern)
        else:
            print(f"Warning: Experiment name '{experiment_name}' not found in SLM patterns.")

    def handle_message(self, msg):
        """
        Handle messages sent to the SLMBuffer from the manager
        :param msg: Message object
        :return: bool indicating whether to close the process
        """
        self.message_history.append(msg)

        match msg.message:

            case "close":
                return True

            case "initialize_slm_queue":
                # Initialize the SLM buffer with provided parameters
                shape = msg.shape
                affine_transform = msg.affine_transform
                experiment_names = msg.experiment_names

                self.initialize(shape, affine_transform, experiment_names)

            case "update_pattern_event":
                event = msg.event

                update_pattern_event_id = event.id
                experiment_name = event.experiment_name

                pattern = self.slm_patterns[experiment_name]

                data = EventSLMPattern(update_pattern_event_id, pattern[1], pattern[0])

                self.to_microscope.put(data)

            case _:
                raise ValueError(f"Unexpected message: {msg}")


class Manager:

    def __init__(self, aq: AllQueues):
        self.msgout = {
            "microscope": aq.manager_to_microscope,
            "outbox": aq.manager_to_outbox,
            "slm_buffer": aq.manager_to_slm_buffer,
            "seg": aq.manager_to_seg,
            "pattern": aq.manager_to_pattern,
        }

        self.msgin = {
            "microscope": aq.microscope_to_manager,
            "outbox": aq.outbox_to_manager,
            "slm_buffer": aq.slm_buffer_to_manager,
            "seg": aq.seg_to_manager,
            "pattern": aq.pattern_to_manager,
        }

        self.initialized = False
        self.schedule = None
        self.experiments = None
        self.times = None
        self.positions = None

    def initialize(self, schedule: ExperimentSchedule):
        self.schedule = schedule
        self.experiments = schedule.experiments
        self.positions = schedule.positions
        self.times = schedule.times

        self.initialized = True

    def process(self):
        assert self.initialized, "manager must be initialized with an experiment schedule to start"

        times: TimeCourse = self.times
        start_time = time() + times.setup

        for t in range(times.count):
            print(f"t = {t}")

            while (time() - start_time) < (t * times.interval) - times.setup:
                pass

            for i, (name, experiment) in enumerate(self.experiments.items()):
                scheduled_time = start_time + (t * times.interval) + (i * times.between)

                stim = experiment.stimulation

                if t % stim.every_t == 0:
                    update_pattern = UpdatePatternEvent(name, stim.config_groups, stim.device_properties)
                    pattern_acquisition = AcquisitionEvent(name, self.positions[name],
                                                           scheduled_time=scheduled_time,
                                                           exposure_time_ms=stim.exposure,
                                                           needs_slm=True,
                                                           config_groups=stim.config_groups,
                                                           devices=stim.device_properties,
                                                           sub_axes=[t, "stim_aq"],
                                                           save_output=stim.save,
                                                           )

                    upmsg = UpdatePatternEventMessage(update_pattern)
                    aqmsg = AcquisitionEventMessage(pattern_acquisition)

                    self.msgout["slm_buffer"].put(upmsg)
                    self.msgout["microscope"].put(upmsg)
                    self.msgout["microscope"].put(aqmsg)

                for channel_name, channel in experiment.channels.items():
                    if t % channel.every_t == 0:
                        channel_acquisition = AcquisitionEvent(name, self.positions[name],
                                                               scheduled_time=scheduled_time,
                                                               exposure_time_ms=channel.exposure,
                                                               config_groups=channel.config_groups,
                                                               devices=channel.device_properties,
                                                               sub_axes=[t, f"channel_{channel_name}"],
                                                               save_output=channel.save,
                                                               )

                        aqmsg = AcquisitionEventMessage(channel_acquisition)
                        self.msgout["microscope"].put(aqmsg)

    def sample_experiment(self):
        t_interval = 5
        t_steps = 5
        t_setup = 2
        t_between = 1
        sample_experiment_toml_path = r"D:\FeedbackControl\SampleExperiment.toml"
        experiments = {name: experiment_from_toml(sample_experiment_toml_path, name) for name in ["a", "b"]}
        positions = {
            "a": Position(1, 2, 0),
            "b": Position(3, 4, 0)
        }

        print(experiments)

        start_time = time()

        for t in range(t_steps):
            print(f"t = {t}")

            while (time() - start_time) < (t * t_interval) - t_setup:
                pass

            for i, (name, experiment) in enumerate(experiments.items()):
                scheduled_time = start_time + (t * t_interval) + (i * t_between)

                stim = experiment.stimulation

                if t % stim.every_t == 0:
                    update_pattern = UpdatePatternEvent(name, stim.config_groups, stim.device_properties)
                    pattern_acquisition = AcquisitionEvent(name, positions[name],
                                                           scheduled_time=scheduled_time,
                                                           exposure_time_ms=stim.exposure,
                                                           needs_slm=True,
                                                           config_groups=stim.config_groups,
                                                           devices=stim.device_properties,
                                                           sub_axes=[t, "stim_aq"],
                                                           save_output=stim.save,
                                                           )

                    print(update_pattern)
                    print(pattern_acquisition)
                    print(pattern_acquisition.get_rel_path())

                for channel_name, channel in experiment.channels.items():
                    if t % channel.every_t == 0:
                        channel_acquisition = AcquisitionEvent(name, positions[name],
                                                               scheduled_time=scheduled_time,
                                                               exposure_time_ms=channel.exposure,
                                                               config_groups=channel.config_groups,
                                                               devices=channel.device_properties,
                                                               sub_axes=[t, f"channel_{channel_name}"],
                                                               save_output=channel.save,
                                                               )
                        print(channel_acquisition)
                        print(channel_acquisition.get_rel_path())

    # def make_update_pattern_event(self, experiment: Experiment, position: Position, scheduled_time: float, ):
    #     pass


if __name__ == "__main__":
    # Example usage
    all_queues = AllQueues()

    outbox = MicroscopeOutbox(all_queues)
    outbox.test_write_data()
    #
    # manager = Manager(aq)
    # manager.sample_experiment()
