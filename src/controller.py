"""
The controller is the brain of the feedback loop.

It is responsible for
- managing timing
- passing messages between processes
- scheduling microscope events
"""

from time import time
from .queues import AllQueues
from .events import AcquisitionEvent, UpdatePatternEvent, Position
from .experiments import Experiment, experiment_from_toml
from .datatypes import AcquisitionData
from .messages import Message
from h5py import File
from pathlib import Path
import numpy as np


class MicroscopeOutbox:

    # grabs data from microscope, writes data to disk

    def __init__(self, aq: AllQueues, base_path: Path = Path().cwd()):
        self.inbox = aq.manager_to_outbox
        self.manager = aq.outbox_to_manager
        self.outbox = aq.acquisition_outbox
        self.seg_queue = aq.segmentation_queue
        self.pattern_queue = aq.raw_pattern

        self.base_path = base_path

        self.message_history = []

    def process(self):

        while True:

            if not self.inbox.empty():
                msg = self.inbox.get()

                must_break = self.handle_message(msg)

                if must_break:
                    break

            if not self.outbox.empty():
                data = self.outbox.get()

                if isinstance(data, Message):
                    must_break = self.handle_message(data)

                    if must_break:
                        break

                assert isinstance(data, AcquisitionData), \
                    f"Unexpected data type: {type(data)}, expected AcquisitionData"

                self.write_data(data)

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


class SLMBuffer:

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_slm_buffer
        self.manager = aq.slm_buffer_to_manager
        self.slm_queue = aq.slm_to_microscope
        self.pattern_queue = aq.pattern_to_slm

    def process(self):
        pass


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

    def process(self):

        while True:
            pass

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
    aq = AllQueues()

    outbox = MicroscopeOutbox(aq)
    outbox.test_write_data()
    #
    # manager = Manager(aq)
    # manager.sample_experiment()
