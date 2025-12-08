import subprocess
from typing import List, Dict
import math
from functools import cmp_to_key

_WEIGHTS = {"id": 0.6, "cov": 0.5, "bits": 0.5, "evalue": 0.4, "length": 0.3}
_DEFAULT_TIE_TOL = 0.005  #
def split_string(input_string, chunk_size=2000, step_size=00):

    chunks = []
    length = len(input_string)


    if length < chunk_size:
        return [input_string]

    for i in range(0, length - chunk_size + 1, step_size):
        chunks.append(input_string[i:i + chunk_size])

    if (length - chunk_size) % step_size != 0:
        chunks.append(input_string[-chunk_size:])

    return chunks

def reverse_complement(dna):
    
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', "N": "N"}
    complemented = ''.join(complement[base] if base in complement else base for base in dna)
    return complemented[::-1]




def run_blast(
    file_path: str,
    blast_path: str,
    db_path: str,
    outfmt: str = "6 qseqid sseqid sstart send pident length mismatch gapopen bitscore evalue qcovs",
    blast_exe: str = "blastn",
) -> bool:
    result = subprocess.run(
        [blast_exe, "-query", file_path, "-db", db_path, "-out", blast_path, "-outfmt", outfmt],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0




def find_blast(gene,id, blast_path,length = 4000, bitscore = None, evalue = None):
    l = len(gene)
    with open(blast_path,"r") as f:
        for line in f:
            columns = line.strip().split("\t")
            targetid = columns[0]
            if targetid != id:
                continue
            bs = columns[4]
            evalue = columns[5]
            sseqid = columns[1].split(".")[0]  # 目标序列的ID (sseqid)
            length2 = length - len(gene)
           
            file_name = "./data/sequence/"+sseqid +".fasta"
            seq = getse(file_name=file_name)
            length = len(seq)
            send = int(columns[3])
            seq = seq[send:]
            gene = gene+seq
            break
    return gene
            
            


def find_blast_augument(gene,id, blast_path,length = 4000, bitscore = None, evalue = None):
    l = len(gene)
    with open(blast_path,"r") as f:
        flag = 1
        rows = []
        for line in f:
            columns = line.strip().split("\t")
            targetid = columns[0]
            if targetid != id and flag == 1:
                continue
            elif targetid != id and flag == 2:
                break
            rows.append(line)
    sorted_lines = rank_blast_lines(rows)
    if len(sorted_lines) == 0:
        return find_blast(gene,id, blast_path,length)
    columns = sorted_lines[0].strip().split("\t")
    length2 = length - len(gene)
    sseqid = columns[0].split(".")[0] 
    send = int(columns[2])
    
    file_name = "./data/sequence/"+sseqid +".fasta"
    try:
       seq = getse(file_name=file_name)
    except Exception:
        return find_blast(gene,id, blast_path,length)
    seq = seq[send:send + length2]
    gene = gene+seq
    return gene

def getse(file_name):
  with open(file_name, "r") as file:
    data = []
    result = ""
    lines = file.readlines()
    for  line in lines:
        if line.startswith(">"):
            continue
        data.append(line.replace("\n", ""))
    result = result.join(data)
    # print(result)
    return result

WEIGHTS = {"id": 0.6, "cov": 0.5, "bits": 0.5, "evalue": 0.4, "length": 0.3}
_DEFAULT_TIE_TOL = 0.005  # 分数近似并列的相对阈值（0.5%）


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _safe_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _parse_line(line: str) -> Dict[str, str]:
    parts = line.strip().split("\t")
    if len(parts) < 11:
        raise ValueError("BLAST line has fewer than 11 fields")
    return {
        "qseqid": parts[0],
        "sseqid": parts[1],
        "sstart": parts[2],
        "send": parts[3],
        "pident": parts[4],
        "length": parts[5],
        "mismatch": parts[6],
        "gapopen": parts[7],
        "bitscore": parts[8],
        "evalue": parts[9],
        "qcovs": parts[10],
    }


def _compute_batch_stats(rows: List[Dict[str, str]]) -> Dict[str, float]:
    max_bits = 1.0
    max_len = 1.0
    max_ev_strength = 1.0
    for r in rows:
        bits = _safe_float(r["bitscore"])
        length = _safe_int(r["length"])
        evalue = _safe_float(r["evalue"])
        ev_strength = max(0.0, min(200.0, -math.log10(evalue + 1e-200)))
        if bits > max_bits:
            max_bits = bits
        if float(length) > max_len:
            max_len = float(length)
        if ev_strength > max_ev_strength:
            max_ev_strength = ev_strength
    return {"max_bitscore": max_bits, "max_length": max_len, "max_ev_strength": max_ev_strength}


def _synergy_score(row: Dict[str, str], stats: Dict[str, float]) -> float:
    pident = _safe_float(row["pident"])
    qcovs = _safe_float(row["qcovs"])
    bits = _safe_float(row["bitscore"])
    evalue = _safe_float(row["evalue"])
    length = float(_safe_int(row["length"]))
    mismatch = float(_safe_int(row["mismatch"]))
    gapopen = float(_safe_int(row["gapopen"]))

    id_norm = max(0.0, min(1.0, pident / 100.0))
    cov_norm = max(0.0, min(1.0, qcovs / 100.0))
    bits_norm = bits / max(stats["max_bitscore"], 1e-9)
    ev_strength = max(0.0, min(200.0, -math.log10(evalue + 1e-200)))
    ev_norm = ev_strength / max(stats["max_ev_strength"], 1e-9)
    len_norm = length / max(stats["max_length"], 1e-9)

    core = (
        (id_norm ** _WEIGHTS["id"])
        * (cov_norm ** _WEIGHTS["cov"])
        * (bits_norm ** _WEIGHTS["bits"])
        * (ev_norm ** _WEIGHTS["evalue"])
        * (len_norm ** _WEIGHTS["length"])
    )

    mismatch_rate = mismatch / max(length, 1.0)
    penalty_gap = 1.0 / (1.0 + gapopen)
    penalty_mismatch = math.exp(-2.0 * mismatch_rate)
    return core * penalty_gap * penalty_mismatch


def _tie_break_compare(a: Dict[str, str], b: Dict[str, str]) -> int:
    pa = _safe_float(a["pident"]); pb = _safe_float(b["pident"])
    if pa != pb: return -1 if pa > pb else 1

    qa = _safe_float(a["qcovs"]); qb = _safe_float(b["qcovs"])
    if qa != qb: return -1 if qa > qb else 1

    ga = _safe_int(a["gapopen"]); gb = _safe_int(b["gapopen"])
    if ga != gb: return -1 if ga < gb else 1

    ma = _safe_int(a["mismatch"]); mb = _safe_int(b["mismatch"])
    if ma != mb: return -1 if ma < mb else 1

    la = _safe_int(a["length"]); lb = _safe_int(b["length"])
    if la != lb: return -1 if la > lb else 1

    sa = a["sseqid"]; sb = b["sseqid"]
    if sa < sb: return -1
    if sa > sb: return 1
    return 0


def _compare_with_tie_tol(a: Dict[str, str], b: Dict[str, str], tie_tol: float) -> int:
    sa = a["_score"]; sb = b["_score"]
    ref = max(sa, sb, 1e-9)
    if abs(sa - sb) > tie_tol * ref:
        return -1 if sa > sb else 1
    return _tie_break_compare(a, b)


def rank_blast_lines(lines: List[str], tie_tol: float = _DEFAULT_TIE_TOL) -> List[str]:
    """
    输入：每一行 BLAST 结果（TSV 11 列）的字符串列表
    输出：按协同同源评分排序后的 'sseqid\\tsstart\\tsend' 列表
    """
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(_parse_line(line))
        except ValueError:
            # 跳过非法行
            continue

    if not rows:
        return []

    stats = _compute_batch_stats(rows)
    for r in rows:
        r["_score"] = _synergy_score(r, stats)

    rows_sorted = sorted(rows, key=cmp_to_key(lambda a, b: _compare_with_tie_tol(a, b, tie_tol)))

    out = [f"{r['sseqid']}\t{r['sstart']}\t{r['send']}" for r in rows_sorted]
    return out
