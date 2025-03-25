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


class MicroscopeOutbox:

    # grabs data from microscope, writes data to disk

    def __init__(self, aq: AllQueues):
        self.inbox = aq.manager_to_outbox
        self.manager = aq.outbox_to_manager
        self.outbox = aq.acquisition_outbox
        self.seg_queue = aq.segmentation_queue
        self.pattern_queue = aq.raw_pattern

    def process(self):
        pass


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
    manager = Manager(aq)
    manager.sample_experiment()
