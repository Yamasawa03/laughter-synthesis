import os
from glob import glob
from os.path import basename, expanduser, join


class LJSpeech:
    SPEAKER = "LJSpeech"

    def __init__(self, wav_dir, textgrid_dir):
        self.wav_dir = expanduser(wav_dir)
        self.textgrid_dir = expanduser(textgrid_dir)
        self.filelist = sorted(glob(join(self.wav_dir, "*.wav")))

    @staticmethod
    def get_fid(path):
        return basename(path)[:-4]

    @staticmethod
    def get_speaker(fid):
        return LJSpeech.SPEAKER

    def collect_speakers(self):
        return [self.SPEAKER]

    def get_textgrid_path(self, fid):
        return join(self.textgrid_dir, fid + ".TextGrid")
