import numpy as np
from sched_dfr import SchedDFR

def main():
    dfr = SchedDFR(down_sample_ratio=2, max_compression=4)
    T = 6
    d_h = 16
    fake_output = np.ones((T, d_h))
    encoded_output = dfr.down_sample(fake_output)
    print(encoded_output)
    decoded_output = dfr.up_sample(encoded_output)
    print(decoded_output)
    assert np.allclose(fake_output, decoded_output)
    

if __name__ == "__main__":
    main()