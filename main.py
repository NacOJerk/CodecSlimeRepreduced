import numpy as np
from sched_dfr import SchedDFR

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
    

if __name__ == "__main__":
    main()