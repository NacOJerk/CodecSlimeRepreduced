import numpy as np
from sched_dfr import SchedDFR
from melt_manager import MeltManager

def main():
    dfr = SchedDFR(down_sample_ratio=2, max_compression=4)
    T = 4
    d_h = 2
    fake_output = np.ones((T, d_h))
    encoded_output = dfr.optimal_down_sample(fake_output)
    decoded_output = dfr.up_sample_encoded(encoded_output)

    print(fake_output)
    print(encoded_output.encoding_lengths)
    print(decoded_output)

    melt_manager = MeltManager()

    for i in range(4 + 1):
        print('Generating some distributions (%d/4)' % i)
        for _ in range(5):
            print('\t', melt_manager.generate_segment_lenght_propotations())
        print("Working hard to increase step count")
        for _ in range(np.floor(melt_manager.s_p / 2).astype(int)):
            melt_manager.generate_segment_lenght_propotations()
        print("\n" * 2)

if __name__ == "__main__":
    main()