# MIAMI: A Mixed Reality Interface for AI-based Music Improvisation

The repository for the MR interface is [here](https://github.com/ryorod/MIAMI_interface). ([Spectator View version](https://github.com/ryorod/MIAMI_SpectatorView))

---

Download pre-trained models  
- [cat-drums_2bar_small.lokl](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-drums_2bar_small.lokl.tar)  
- [cat-mel_2bar_big](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/cat-mel_2bar_big.tar)  
- [hierdec-trio_16bar](https://storage.googleapis.com/magentadata/models/music_vae/checkpoints/hierdec-trio_16bar.tar)

`cd server`  
`pip install -r requirements.txt`  
`python main.py --separate_mode --verbose`

---

references:  
https://github.com/magenta/magenta/tree/main/magenta/models/music_vae  
https://github.com/Elvenson/midiMe
