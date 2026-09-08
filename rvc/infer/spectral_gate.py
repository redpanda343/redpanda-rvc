import torch
import torch.nn.functional as F


class SpectralGate(torch.nn.Module):
    def __init__(self, sample_rate, n_fft, prop_decrease=0.9):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.prop_decrease = prop_decrease
        self.register_buffer(
            "window", torch.hann_window(n_fft), persistent=False
        )
        self.register_buffer(
            "smoothing_filter",
            self._make_smoothing_filter(),
            persistent=False,
        )

    def _make_smoothing_filter(self):
        frequency_bins = max(
            1, int(500 / (self.sample_rate / (self.n_fft / 2)))
        )
        time_bins = max(
            1, int(50 / ((self.hop_length / self.sample_rate) * 1000))
        )
        frequency = torch.cat(
            (
                torch.linspace(0, 1, frequency_bins + 1)[:-1],
                torch.linspace(1, 0, frequency_bins + 2)[1:],
            )
        )
        time = torch.cat(
            (
                torch.linspace(0, 1, time_bins + 1)[:-1],
                torch.linspace(1, 0, time_bins + 2)[1:],
            )
        )
        kernel = torch.outer(frequency, time).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    @staticmethod
    def _amplitude_to_db(value):
        epsilon = torch.finfo(value.real.dtype).eps
        value_db = 20 * torch.log10(value.abs() + epsilon)
        floor = value_db.amax(dim=-1, keepdim=True) - 40
        return torch.maximum(value_db, floor)

    @torch.inference_mode()
    def forward(self, audio, noise=None):
        spectrum = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="constant",
        )
        reference = spectrum
        if noise is not None:
            reference = torch.stft(
                noise,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=self.window,
                return_complex=True,
                center=True,
                pad_mode="constant",
            )
        reference_db = self._amplitude_to_db(reference)
        spectrum_db = self._amplitude_to_db(spectrum)
        deviation, average = torch.std_mean(reference_db, dim=-1)
        threshold = average + deviation * 1.5
        mask = spectrum_db > threshold.unsqueeze(-1)
        mask = self.prop_decrease * (mask.float() - 1.0) + 1.0
        mask = F.conv2d(
            mask.unsqueeze(1),
            self.smoothing_filter.to(mask.dtype),
            padding="same",
        ).squeeze(1)
        filtered = spectrum * mask
        return torch.istft(
            filtered,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=audio.shape[-1],
        ).to(audio.dtype)
