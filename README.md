# MIAMI: A Mixed Reality Interface for AI-based Music Improvisation

The repository for the MR interface is [here](https://github.com/ryorod/MIAMI_interface). ([Spectator View version](https://github.com/ryorod/MIAMI_SpectatorView))

---

Tested on system requirements below.

- macOS Catalina (10.15.4), Big Sur (11.6.1) or Monterey (12.3.1)
- [requirements.txt](/server/requirements.txt)
- Python 3.7.8
- Ableton Live 11 Suite (11.1.1)
- Max 8.1.11

---

Download pre-trained models  
- [cat-drums_2bar_small.lokl](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-drums_2bar_small.lokl.tar)  
- [cat-mel_2bar_big](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar)  
- [hierdec-trio_16bar](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/hierdec-trio_16bar.tar)  
- [3dim drums, melody, trio MidiMe models](https://github.com/ryorod/MIAMI/releases/download/drums_melody_trio_models_v1.0/drums_melody_trio_models.zip) (unzip it)

and put them under `server/model_file`.

Create `config.yml` (refer to `config.yml.sample`).

`cd server`  
`pip install -r requirements.txt`  
`python main.py --separate_mode --verbose`

---

references:  
https://github.com/magenta/magenta/tree/main/magenta/models/music_vae  
https://github.com/Elvenson/midiMe
