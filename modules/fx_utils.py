# fx_utils.py

import random
from pathlib import Path
from typing import Union, List, Optional
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F  # NEW: for simple resampling

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform, EmptyPathException
# from torch_audiomentations.utils.convolution import convolve  # no longer used
from torch_audiomentations.utils.file import find_audio_files_in_paths
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.utils.object_dict import ObjectDict


class ApplyImpulseResponse(BaseWaveformTransform):
    """
    Apply circular convolution with impulse responses (IRs).
    - IRs are read at 8 kHz, trimmed there (time-accurate), then resampled to the working SR.
    - Output is max-normalized per example (across channels).
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # IRs are mixed to mono and applied to all channels
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = False
    requires_target = False

    def __init__(
        self,
        ir_paths: Union[List[Path], List[str], Path, str],
        convolve_mode: str = "full",  # kept for backward-compat; ignored (we always do circular)
        compensate_for_propagation_delay: bool = False,  # kept for API; ignored in circular mode
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
        trim_ms: int = 75,  # default trim window
    ):
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        self.ir_paths = find_audio_files_in_paths(ir_paths)
        if len(self.ir_paths) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        # Working audio loader (downmix to mono to match original behavior)
        if sample_rate is not None:
            self.audio = Audio(sample_rate=sample_rate, mono=True)

        # NEW: IRs are always preprocessed at 8 kHz before trimming
        self.ir_preprocess_rate = 8000
        self.ir_audio_8k = Audio(sample_rate=self.ir_preprocess_rate, mono=True)

        self.trim_ms = trim_ms
        self.convolve_mode = convolve_mode
        self.compensate_for_propagation_delay = compensate_for_propagation_delay

    def _trim_samples(self, sr: int) -> int:
        """Number of samples to keep for a given sr and configured trim_ms."""
        return max(1, int(round(self.trim_ms * sr / 1000)))

    @staticmethod
    def _resample_1d(wave_ct: Tensor, src_sr: int, dst_sr: int) -> Tensor:
        """
        Very lightweight resampler using linear interpolation.
        Input: (channels, time). Output: (channels, time_resampled)
        """
        if src_sr == dst_sr:
            return wave_ct
        c, t = wave_ct.shape
        t_dst = max(1, int(round(t * float(dst_sr) / float(src_sr))))
        wave_ct = wave_ct.unsqueeze(0)  # (1, C, T)
        wave_ct = F.interpolate(wave_ct, size=t_dst, mode="linear", align_corners=False)
        return wave_ct.squeeze(0)

    @staticmethod
    def _wrap_to_length(h: Tensor, N: int) -> Tensor:
        """
        Wrap IR h (B, M) into length N by modulo addition (circular kernel).
        """
        B, M = h.shape
        if M == N:
            return h
        wrapped = torch.zeros(B, N, device=h.device, dtype=h.dtype)
        # Add chunks of length N
        start = 0
        while start < M:
            end = min(start + N, M)
            seg_len = end - start
            wrapped[:, :seg_len] += h[:, start:end]
            start += N
        return wrapped

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size, _, _ = samples.shape

        # Working SR for the forward pass
        audio = self.audio if hasattr(self, "audio") else Audio(sample_rate, mono=True)
        max_len_8k = self._trim_samples(self.ir_preprocess_rate)

        random_ir_paths = random.choices(self.ir_paths, k=batch_size)
        trimmed_resampled_irs_tc = []

        for ir_path in random_ir_paths:
            # 1) Load at 8kHz and TRIM there (time-accurate per your requirement)
            ir_ct_8k = self.ir_audio_8k(ir_path)  # (C, T_8k)
            ir_ct_8k = ir_ct_8k[..., :max_len_8k]

            # Safety: ensure at least one sample
            if ir_ct_8k.shape[-1] == 0:
                ir_ct_8k = torch.zeros((ir_ct_8k.shape[0], 1), dtype=samples.dtype, device=samples.device)

            # 2) Resample the trimmed IR back to working SR for convolution
            ir_ct = self._resample_1d(ir_ct_8k.to(samples.device, dtype=samples.dtype),
                                      self.ir_preprocess_rate, audio.sample_rate)

            # (time, channels) for pad_sequence
            trimmed_resampled_irs_tc.append(ir_ct.transpose(0, 1))

        # (B, 1, M_max) after pad + transpose back to (B, C, T)
        self.transform_parameters["ir"] = pad_sequence(
            trimmed_resampled_irs_tc, batch_first=True, padding_value=0.0
        ).transpose(1, 2)

        self.transform_parameters["ir_paths"] = random_ir_paths

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        """
        Circular convolution:
          y[n] = sum_k x[(n-k) mod N] * h[k]
        Implemented via FFT with kernel wrapped to length N, per batch; then max-normalized.
        """
        B, C, N = samples.shape
        device = samples.device
        dtype = samples.dtype

        # (B, 1, M_max) -> (B, M_max)
        ir_padded = self.transform_parameters["ir"].to(device=device, dtype=dtype).squeeze(1)

        # Wrap IR to signal length (circular kernel)
        h_wrapped = self._wrap_to_length(ir_padded, N)  # (B, N)

        # FFT-based circular convolution for all channels
        X = torch.fft.rfft(samples, n=N, dim=2)           # (B, C, Nf)
        H = torch.fft.rfft(h_wrapped, n=N, dim=1)         # (B, Nf)
        Y = torch.fft.irfft(X * H.unsqueeze(1), n=N, dim=2)  # (B, C, N)

        # Max-normalize per example across channels and time
        max_val = Y.abs().amax(dim=(1, 2), keepdim=True)
        Y = torch.where(max_val > 0, Y / max_val, Y)

        # Note: delay compensation is not meaningful in circular mode; ignored.
        return ObjectDict(
            samples=Y,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )
