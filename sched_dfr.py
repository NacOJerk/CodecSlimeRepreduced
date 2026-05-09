import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class EncodedData:
    encoded_data: np.ndarray
    encoding_lengths: List[int]

class SchedDFR:
    def __init__(self, down_sample_ratio: float, max_compression: int):
        self.down_sample_ratio = down_sample_ratio
        self.max_compression = max_compression

    def down_sample(self, raw_outpout: np.ndarray) -> EncodedData:
        return EncodedData(raw_outpout, [1] * len(raw_outpout))
    
    def up_sample(self, encoded_output: EncodedData) -> np.ndarray:
        assert all([x == 1 for x in encoded_output.encoding_lengths])
        return encoded_output.encoded_data

    def encode_binary(self, encoded_data: EncodedData) -> bytes:
        raise NotImplementedError()
    
    def decode_binary(self, encoded_bytes: bytes) -> EncodedData:
        raise NotImplementedError()
    
    def encode(self, raw_output: np.ndarray) -> bytes:
        encoded_data = self.down_sample(raw_output)
        return self.encode_binary(encoded_data)
    
    def decode(self, encoded_data: bytes) -> np.ndarray:
        encoded_output = self.decode_binary(encoded_data)
        return self.up_sample(encoded_output)