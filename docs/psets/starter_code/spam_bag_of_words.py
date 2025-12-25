# ========= config =========
DATASET = "sms"   
DATA_DIR = "data"

# ========= imports =========
import os, re, tarfile, zipfile, urllib.request, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# ========= tiny utils =========
def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"↓ downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest

def _is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    return os.path.commonprefix([abs_directory, abs_target]) == abs_directory

def safe_extract_tar(tar: tarfile.TarFile, path: str):
    for member in tar.getmembers():
        target_path = os.path.join(path, member.name)
        if not _is_within_directory(path, target_path):
            raise Exception("Blocked path traversal in tar file")
    tar.extractall(path)

def extract(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(dest)
    else:
        # auto-detect compression: r:* handles .tar.gz and .tar.bz2
        with tarfile.open(archive, "r:*") as t:
            safe_extract_tar(t, str(dest))
    return dest

TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")
def strip_headers(text: str):
    parts = text.split("\n\n", 1)
    return parts[1] if len(parts) == 2 else text

def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\b", " EMAIL ", text)
    text = re.sub(r"\d+(?:\.\d+)?", " NUM ", text)
    return TOKEN_RE.findall(text)

# ========= dataset loaders from URL =========

def load_sms_from_url(root="data/sms"):
    """
    UCI SMS Spam Collection (zip -> 'SMSSpamCollection' TSV-like file).
    """
    root = Path(root)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    arc = download(url, root / "smsspam.zip")
    ex = root / "extracted"
    if not ex.exists():
        extract(arc, ex)
    data_file = next(ex.rglob("SMSSpamCollection"), None)
    docs, y = [], []
    with open(data_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "\t" not in line:
                continue
            lab, text = line.strip().split("\t", 1)
            docs.append(text)
            y.append(1 if lab.lower()=="spam" else 0)  # spam=1, ham=0
    return docs, np.array(y)

# ========= vectorization =========
def build_vocab(docs, min_df=3, max_vocab=30000):
    df = {}
    for txt in docs:
        for tok in set(tokenize(txt)):
            df[tok] = df.get(tok, 0) + 1
    items = [(tok, c) for tok, c in df.items() if c >= min_df]
    items.sort(key=lambda x: x[1], reverse=True)
    return {tok:i for i,(tok,_) in enumerate(items[:max_vocab])}

def vectorize_df(docs, vocab, index=None):
    """
    docs: list[str] of email texts
    vocab: dict[token -> column_index]  (e.g., built earlier)
    index: optional iterable of row labels (len == len(docs));
           if None, uses RangeIndex and names it 'email'
    returns: pandas.DataFrame of shape (n_docs, |vocab|), values in {0,1}
    """
    V = len(vocab)
    X = np.zeros((len(docs), V), dtype=np.uint8)

    for i, txt in enumerate(docs):
        for tok in set(tokenize(txt)):  # presence, not counts
            j = vocab.get(tok)
            if j is not None:
                X[i, j] = 1

    # recover column order by vocab indices
    cols = [None] * V
    for tok, j in vocab.items():
        cols[j] = tok

    if index is None:
        index = pd.Index(range(len(docs)), name="email")
    else:
        if len(index) != len(docs):
            raise ValueError("index length must match number of docs")
        index = pd.Index(index, name="email")

    return pd.DataFrame(X, index=index, columns=cols)
    
# ========= run it =========
docs, y = load_sms_from_url()
print(f"Loaded {len(docs)} documents: spam={int(y.sum())}, ham={len(y)-int(y.sum())}")

vocab = build_vocab(docs, min_df=3, max_vocab=30000)
bow = vectorize_df(docs, vocab)
print(bow)
