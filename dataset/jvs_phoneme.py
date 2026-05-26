import os
from collections import defaultdict
from glob import glob
from os.path import basename, dirname, expanduser, join

import librosa


class JVSPhoneme:
    def __init__(self, root):
        self.root = expanduser(root)
        self.filelist = self.collect_files()
        self._transcripts = {}

    def collect_files(self):
        files = []
        pattern = join(self.root, "jvs*", "parallel100", "wav24kHz16bit", "*.wav")
        for path in sorted(glob(pattern)):
            if librosa.get_duration(path=path) != 0:
                files.append(path)
        return files

    @staticmethod
    def get_fid(path):
        speaker = basename(dirname(dirname(dirname(path))))
        wav_id = basename(path)[:-4]
        return f"{speaker}_{wav_id}"

    @staticmethod
    def get_speaker(fid):
        return fid.split("_", 1)[0]

    def collect_speakers(self):
        speakers = sorted({self.get_speaker(self.get_fid(path)) for path in self.filelist})
        return speakers

    def get_speaker2fids(self):
        speaker2fids = defaultdict(list)
        for path in self.filelist:
            fid = self.get_fid(path)
            speaker2fids[self.get_speaker(fid)].append(fid)
        return speaker2fids

    def get_transcript(self, fid):
        speaker, utt_id = fid.split("_", 1)
        if speaker not in self._transcripts:
            self._transcripts[speaker] = self._load_transcripts(speaker)
        return self._transcripts[speaker][utt_id]

    def _load_transcripts(self, speaker):
        path = join(self.root, speaker, "parallel100", "transcripts_utf8.txt")
        transcripts = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                utt_id, text = line.split(":", 1)
                transcripts[utt_id] = text
        return transcripts


if __name__ == "__main__":
    root = "~/data/jvs/jvs_ver1"
    ds = JVSPhoneme(root)
    print(len(ds.filelist))
    print(len(ds.collect_speakers()))
