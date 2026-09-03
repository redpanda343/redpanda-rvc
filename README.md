<p align="center">
  <img src="assets/applio_mascot.png" alt="Applio mascot" width="180">
</p>

# Applio Fork

This project is a fork of [Applio](https://github.com/IAHispano/Applio), with changes focused on dataset preprocessing, inference, training, normalization, and a cleaner WebUI experience.

## Changelog

- **AI-based automatic slicing** — The automatic slicer now uses [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) to detect speech/singing in the dataset.
- **Speech-aware post normalization** — Post normalization also uses FireRedVAD so that speech is boosted without unnecessarily amplifying background noise.
- **PM pitch extraction restored** — PM is available again as an option for both inference and training.
- **Audio normalization** — The previous volume envelope option has been replaced with audio normalization.
- **Bloat removed** — Unnecessary components and UI elements have been removed for a cleaner experience.
- **Updated WebUI colors** — The interface uses a different color scheme from upstream Applio.
- **Adjusted WebUI descriptions** — Some vague descriptions have been clarified.
- **FLAC support** — Supports training with FLAC files.
- **Small tweaks to algorithm** — Now fully match [Mainline RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) behavior.
- **Small tweaks to training script — D is frozen during G steps.
- **Deterministic inference — You can now use a deterministic seed for inference.
- **Checkpoint Exporter — Convert G files to inference weights in the WebUI.
- **Same SR model blending — Voice blender can now merge models that don't share the same sampling rate.
 
## Credits

- [Applio](https://github.com/IAHispano/Applio)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
