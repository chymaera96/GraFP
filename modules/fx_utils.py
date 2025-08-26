import random
from pathlib import Path
from typing import Union, List, Optional
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from torch_audiomentations.core.transforms_interface import BaseWaveformTransform, EmptyPathException
from torch_audiomentations.utils.convolution import convolve
from torch_audiomentations.utils.file import find_audio_files_in_paths
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.utils.object_dict import ObjectDict



class ApplyImpulseResponse(BaseWaveformTransform):
    """
    Convolve the given audio with impulse responses.
    This version trims each IR before use.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # Note: This transform has only partial support for multichannel audio. IRs that are not
    # mono get mixed down to mono before they are convolved with all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = False  # FIXME: some work is needed to support targets (see FIXMEs in apply_transform)
    requires_target = False

    def __init__(
        self,
        ir_paths: Union[List[Path], List[str], Path, str],
        convolve_mode: str = "full",
        compensate_for_propagation_delay: bool = False,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
    ):
        """
        :param ir_paths: Either a path to a folder with audio files or a list of paths to audio files.
        :param convolve_mode:
        :param compensate_for_propagation_delay: Convolving audio with a RIR normally
            introduces a bit of delay, especially when the peak absolute amplitude in the
            RIR is not in the very beginning. When compensate_for_propagation_delay is
            set to True, the returned slices of audio will be offset to compensate for
            this delay.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        :param target_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        self.ir_paths = find_audio_files_in_paths(ir_paths)

        if sample_rate is not None:
            # Ensure mono to match original behavior (downmix if needed)
            self.audio = Audio(sample_rate=sample_rate, mono=True)

        if len(self.ir_paths) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.convolve_mode = convolve_mode
        self.compensate_for_propagation_delay = compensate_for_propagation_delay

    def _trim_samples(self, sr: int, trim_ms: int = 75) -> int:
        
        # At least 1 sample to avoid empty tensors
        return max(1, int(round(trim_ms * sr)))

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size, _, _ = samples.shape

        audio = self.audio if hasattr(self, "audio") else Audio(sample_rate, mono=True)
        max_len = self._trim_samples(audio.sample_rate)

        random_ir_paths = random.choices(self.ir_paths, k=batch_size)
        trimmed_irs_tc = []
        for ir_path in random_ir_paths:
            ir_ct = audio(ir_path)  
            ir_ct = ir_ct[..., :max_len]
            # Fallback: if a weird file is shorter than 1 sample after read (shouldn't happen), ensure at least one zero
            if ir_ct.shape[-1] == 0:
                ir_ct = torch.zeros((ir_ct.shape[0], 1), dtype=ir_ct.dtype, device=ir_ct.device)
            trimmed_irs_tc.append(ir_ct.transpose(0, 1))  # (time, channels)

        self.transform_parameters["ir"] = pad_sequence(
            trimmed_irs_tc,
            batch_first=True,
            padding_value=0.0,
        ).transpose(1, 2)

        self.transform_parameters["ir_paths"] = random_ir_paths

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, 1, max_ir_length) — stays mono IR applied to all channels
        ir = self.transform_parameters["ir"].to(samples.device)

        convolved_samples = convolve(
            samples, ir.expand(-1, num_channels, -1), mode=self.convolve_mode
        )

        if self.compensate_for_propagation_delay:
            propagation_delays = ir.abs().argmax(dim=2, keepdim=False)[:, 0]
            convolved_samples = torch.stack(
                [
                    convolved_sample[
                        :, propagation_delay : propagation_delay + num_samples
                    ]
                    for convolved_sample, propagation_delay in zip(
                        convolved_samples, propagation_delays
                    )
                ],
                dim=0,
            )

            return ObjectDict(
                samples=convolved_samples,
                sample_rate=sample_rate,
                targets=targets,  # FIXME compensate targets as well?
                target_rate=target_rate,
            )

        else:
            return ObjectDict(
                samples=convolved_samples[..., :num_samples],
                sample_rate=sample_rate,
                targets=targets,  # FIXME crop targets as well?
                target_rate=target_rate,
            )
