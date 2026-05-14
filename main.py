import numpy as np
import torch

from sched_dfr import SchedDFR
from melt_manager import MeltManager


def main():
    dfr = SchedDFR(down_sample_ratio=2, max_compression=4)
    T = 8
    d_h = 2
    fake_output = np.random.rand(T, d_h)
    encoded_output = dfr.optimal_down_sample(fake_output)
    decoded_output = dfr.up_sample_encoded(encoded_output)

    print(fake_output)
    print(encoded_output.encoding_lengths)
    print(decoded_output)

    melt_manager = MeltManager()

    fake_features = torch.randn(1, d_h, T)
    step_increment = int(np.floor(melt_manager.s_p / 2))
    for i in range(4 + 1):
        print('Generating some distributions (%d/4)' % i)
        step = i * step_increment
        for _ in range(5):
            print(melt_manager.melt(fake_features, step=step), '\n')
        print("\n" * 2)


if __name__ == "__main__":
    main()
