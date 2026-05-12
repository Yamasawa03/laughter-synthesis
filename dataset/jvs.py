import librosa
from glob import glob
from os.path import basename, dirname, expanduser, join, splitext
from collections import defaultdict


class JVS:
    def __init__(self, root):
        self.root = expanduser(root)
        self.fid2path = {}
        self.filelist = self.collect_files()

    def collect_files(self):
        files = []
        path = join(self.root, '*', 'parallel100', 'wav24kHz16bit', '*.wav')
        for f in glob(path):
            dur = librosa.get_duration(path=f)
            if dur != 0:
                files.append(f)
                self.fid2path[self.get_fid(f)] = f
        return files

    def get_speaker2fids(self):
        spkr2wavs = defaultdict(list)
        for wav in self.filelist:
            fid = self.get_fid(wav)
            spkr = self.get_speaker(fid)
            spkr2wavs[spkr].append(fid)
        return spkr2wavs

    @staticmethod
    def get_fid(path):
        speaker = basename(dirname(dirname(dirname(path))))
        filename_stem = splitext(basename(path))[0]
        return f'{speaker}_{filename_stem}'

    @staticmethod
    def get_speaker(fid):
        return fid.split('_', 1)[0]

    def collect_speakers(self):
        speakers = set()
        for wav in self.filelist:
            fid = self.get_fid(wav)
            speakers.add(self.get_speaker(fid))
        speakers = list(speakers)
        speakers.sort()
        return speakers
