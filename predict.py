import sys
import argparse
import torch
import numpy as np
import copy
from distutils.util import strtobool
from Bio import SeqIO
from tqdm import tqdm

from model.model_vitax import Encoder
from model.tokenizer_hyena import CharacterTokenizer
from lca.tree import *
from utils import *


def parse_bool(x):
    return bool(strtobool(str(x)))


def reverse_complement(dna):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    complemented = ''.join(complement.get(base, base) for base in dna)
    return complemented[::-1]


def main():
    parser = argparse.ArgumentParser(
        description="ViTax: DSDNA virus genus-level classification"
    )
    parser.add_argument('--contigs',  default="test_contigs.fasta",help='FASTA file of contigs')
    parser.add_argument('--model',  default="./model_save/model_weight.pth", help='Model weights (.pth)')
    parser.add_argument('--kmeans',   default="./model_save/kmeans.pickle",help='KMeans pickle')
    parser.add_argument('--tree',   default="./model_save/tbt.pickle", help='Taxonomy belief tree pickle')
    parser.add_argument('--index',  default="./model_save/index.pickle", help='Tree index pickle')
    parser.add_argument('--out',  default="output.txt", help='Output prediction file')
    parser.add_argument('--confidence', type=float, default=0.6, help='Confidence threshold')
    parser.add_argument('--window_size', type=int, default=400, help='Sliding step size')
    parser.add_argument('--chunk_size', type=int, default=2000, help='Chunk size for splitting')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--rc', type=parse_bool, default=True, help='Use reverse complement')
    parser.add_argument('--augment', type=parse_bool, default=True, help='Use BLAST augmentation')
    parser.add_argument('--augment_len', type=int, default=4000, help='Augmentation length')
    parser.add_argument('--blast_db', type = str, default='./data/blast/blastdb', help='Blast db')
    parser.add_argument('--blast_tmp', type=str, default='blast_temp_out', help='BLAST temp output')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device selection')


    args = parser.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    node = load_node(args.tree)
    kmean = load_node(args.kmeans)
    index = load_node(args.index)
    node = convert_node_to_treenode(node)

    model = Encoder().to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state, strict=False)

    dnatokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],
        model_max_length=32770,
        add_special_tokens=False,
        padding_side='left',
    )

    results = []

    if args.augment:
        run_blast(args.contigs, args.blast_tmp, args.blast_db)

    with torch.no_grad():
        for record in tqdm(SeqIO.parse(args.contigs, "fasta")):
            tbt = copy.deepcopy(node)
            sequence_id = record.id
            sequence = str(record.seq)

            if args.augment:
                sequence = find_blast_augument(sequence, sequence_id, args.blast_tmp, args.augment_len)

            dnas = split_string(sequence, chunk_size=args.chunk_size, step_size=args.window_size)
            lendna = len(dnas) * (2 if args.rc else 1)

            inds = []
            for i in range(0, len(dnas), args.batch_size):
                batch_dnas = dnas[i:i + args.batch_size] if i + args.batch_size <= len(dnas) else dnas[i:]
                ids = dnatokenizer(batch_dnas, padding=True, return_tensors='pt')['input_ids'].to(device)
                embedding = model(ids)
                ind = kmean.predict(embedding.cpu().numpy().tolist())
                inds.extend(ind)

            if args.rc:
                rc_dnas = split_string(reverse_complement(sequence), chunk_size=args.chunk_size, step_size=args.window_size)
                for i in range(0, len(rc_dnas), args.batch_size):
                    batch_dnas = rc_dnas[i:i + args.batch_size] if i + args.batch_size <= len(rc_dnas) else rc_dnas[i:]
                    ids = dnatokenizer(batch_dnas, padding=True, return_tensors='pt')['input_ids'].to(device)
                    embedding = model(ids)
                    ind = kmean.predict(embedding.cpu().numpy().tolist())
                    inds.extend(ind)

            add_values_node(tbt, index, inds)
            max_sum, leaf = max_leaf_sum2(tbt["root"], confidence=args.confidence, length=lendna)
            belief = (max_sum / lendna) if lendna > 0 else 0.0
            pname = "unclassified" if getattr(leaf, "level", "root") == "root" else f"{leaf.name}_{leaf.level}"
            results.append(f"{sequence_id} {pname} {belief:.2f}")

    with open(args.out, "w", encoding="utf-8") as f:
        for s in results:
            f.write(s + "\n")

    print("Done, please check the output file:", args.out)


if __name__ == "__main__":
    main()