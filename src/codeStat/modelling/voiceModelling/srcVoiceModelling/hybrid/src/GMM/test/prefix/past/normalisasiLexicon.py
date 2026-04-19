from codeStat.modelimports.modelFunction import *


PETA_FONEM: Dict[str, List[str]] = {
    "ng": ["ng"], "ny": ["ny"], "th": ["th"], "dh": ["dh"],
    "a": ["a"],  "b": ["b"],  "c": ["ch"], "d": ["d"],
    "e": ["e"],  "f": ["f"],  "g": ["g"],  "h": ["h"],
    "i": ["i"],  "j": ["j"],  "k": ["k"],  "l": ["l"],
    "m": ["m"],  "n": ["n"],  "o": ["o"],  "p": ["p"],
    "r": ["r"],  "s": ["s"],  "t": ["t"],  "u": ["u"],
    "v": ["v"],  "w": ["w"],  "y": ["y"],  "z": ["z"],
}


def normalisasi(teks: str) -> str:
    """Lowercase, hapus non-huruf, normalisasi spasi."""
    teks = teks.lower().strip()
    teks = re.sub(r"[^a-z\s]", "", teks)
    return re.sub(r"\s+", " ", teks).strip()


def kata_ke_fonem(kata: str) -> List[str]:
    """Kata → urutan fonem. Digraf (ng, ny, th, dh) diprioritaskan."""
    fonem, i = [], 0
    while i < len(kata):
        if i + 1 < len(kata) and kata[i: i + 2] in PETA_FONEM:
            fonem.extend(PETA_FONEM[kata[i: i + 2]])
            i += 2
        elif kata[i] in PETA_FONEM:
            fonem.extend(PETA_FONEM[kata[i]])
            i += 1
        else:
            i += 1
    return fonem if fonem else [kata]


def buat_lexicon(transkrip_list: List[str]) -> Dict[str, List[str]]:
    """Buat kamus {kata: [fonem]} dari semua transkrip."""
    lexicon: Dict[str, List[str]] = {}
    for kalimat in transkrip_list:
        for kata in normalisasi(kalimat).split():
            if kata not in lexicon:
                lexicon[kata] = kata_ke_fonem(kata)
    return lexicon