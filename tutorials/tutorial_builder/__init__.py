"""
Builder of the narrated tutorial videos of visual-language-model-spark: it reads the storyboards of tutorials/storyboards,
the configurations and code of the repository they quote, and the figures made from the real runs by tutorials/figures,
synthesizes the narration offline, draws and captures every frame, encodes the videos with their subtitles into the
output folder, checks them, and writes the index page that lists them. build_tutorials.py is its entry point.
"""
