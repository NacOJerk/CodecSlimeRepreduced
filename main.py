import numpy as np
from sched_dfr import SchedDFR
from melt_manager import MeltManager

def main():
    dfr = SchedDFR(down_sample_ratio=2, max_compression=4)
    T = 8
    d_h = 2
    fake_output = np.random.rand(*(4, T, d_h))
    # encoded_output = dfr.optimal_down_sample(fake_output)
    # decoded_output = dfr.up_sample_encoded(encoded_output)

    # print(fake_output)
    # print(encoded_output.encoding_lengths)
    # print(decoded_output)

    melt_manager = MeltManager()

    for i in range(4 + 1):
        print('Generating some distributions (%d/4)' % i)
        for _ in range(5):
            result = melt_manager.generate_segment_lenght_propotations()
            print('\t', result)
            if result is None:
                continue
            print(melt_manager.random_down_sample_with_proption_multi(fake_output, result), '\n')
        print("Working hard to increase step count")
        for _ in range(np.floor(melt_manager.s_p / 2).astype(int)):
            melt_manager.generate_segment_lenght_propotations()
        print("\n" * 2)



if __name__ == "__main__":
    main()