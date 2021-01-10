import logging
import numpy as np
from pyvf.strategy import *
from pyvf.strategy.Model import ConstantModel
from pyvf.strategy.Responder import PerfectResponder, RampResponder

_logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # for starting_threshold, true_threshold in zip(
    #     [[30.0, 30.0], [30.0, 30.0], [30.0, 30.0], [30.0, 30.0], [30.0, 30.0], [30.0, 30.0]],
    #     [[-10.0, -10.0], [0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [35.0, 35.0]]
    # ):
    for starting_threshold, true_threshold in zip(
        [[30.0, 30.0]],
        [[10.0, 30.0]]
    ):
        starting_threshold = np.array(starting_threshold)
        true_threshold = np.array(true_threshold)
        print(starting_threshold, true_threshold)
        responder = RampResponder(true_threshold=true_threshold, fp=0., fn=0., width=4, seed=0)
        model = ConstantModel(eval_pattern=PATTERN_DOUBLE,
                              mean=starting_threshold,
                              std=4)  # std no effect in this case

        strategy = ZestMSPStrategy(
            pattern=PATTERN_DOUBLE,
            blindspot=[],
            model=model,
            term_std=1.5,
            rng=0,
            growth_pattern=EmptyGrowthPattern()
        )

        collect_data = []
        collect_total_presentations = []
        collect_estimates = []

        for i in range(1000):
            counter = 0
            data = []
            while True:
                stimulus, threshold = strategy.get_stimulus_threshold(data)
                if stimulus is None:
                    break  # Test is finished
                else:  # isinstance(stimulus, Stimulus):  # Single stimulus perimetry
                    # _logger.debug("%3d: %s\t%s", counter, threshold, stimulus)
                    if isinstance(stimulus, Stimulus):
                        stimulus = stimulus.copy(**{TSDISP: counter})
                        stimulus = responder.get_response(stimulus)
                        data.append(stimulus)
                    elif isinstance(stimulus[0], Stimulus):
                        stimulus = [s.copy(**{TSDISP: counter}) for s in stimulus]
                        stimulus = responder.get_response(stimulus)
                        data.extend(stimulus)
                    else:
                        raise ValueError(f"Invalid stimulus object or list of stimuli: {stimulus}")
                counter += 1

            # _logger.info("%3d: %s\t%s", counter, threshold, stimulus)

            total_presentations = sum(map(lambda s: 1.0 / s.multi, data))
            # _logger.info("Presentations (total): %s", total_presentations)
            # _logger.info("Presentations (per location): %s", total_presentations / len(PATTERN_DOUBLE))

            collect_data.append(Stimulus.to_numpy(data))
            collect_total_presentations.append(total_presentations)
            collect_estimates.append(threshold)

        print("Presentations:")
        print(np.array(collect_total_presentations).mean())
        print(np.array(collect_total_presentations).std())
        print("Thresholds:")
        print(np.array(collect_estimates).mean(axis=0))
        print(np.array(collect_estimates).std(axis=0))
        print("Thresholds, aggregate:")
        print(np.array(collect_estimates).mean())
        print(np.array(collect_estimates).std())
