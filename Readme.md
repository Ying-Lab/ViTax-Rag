# ViTax-RAG: Retrieval-Augmented Language Model for Viral Taxonomy Classification

## Introduction
ViTax-RAG is a hybrid viral taxonomy classification framework that integrates BLAST-based retrieval with a Bi-Hyena genomic language model to achieve adaptive, accurate, and robust taxonomic assignment from metagenomic sequences.
## Installation 

Install dependencies:
```powershell
git clone https://github.com/Ying-Lab/ViTax-Rag.git
cd ViTax-Rag
pip install -r requirements.txt
```
Install NCBI BLAST+:
```powershell
# 1) Download the latest BLAST+ tarball (adjust version if needed)
mkdir -p ~/downloads && cd ~/downloads
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.12.0+-x64-linux.tar.gz

# 2) Extract and move to a permanent location
tar -zxvf ncbi-blast-2.12.0+-x64-linux.tar.gz
mv ncbi-blast-2.12.0+ ~/blast+

# 3) Persist PATH (so BLAST commands are available in every shell)
echo 'export PATH="$HOME/blast+/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 4) Verify installation
which blastn
blastn -version
```

## Quick Start

Basic run (auto device selection):
```powershell
python predict.py --contigs test_contigs.fasta --out output.txt --device auto
```
Adjust chunking and batch size for long contigs:
```powershell
python predict.py --contigs test_contigs.fasta --chunk_size 2000 --window_size 400 
```


## Command-Line Options

- `--contigs` FASTA input path, default `test_contigs.fasta`
- `--out` output predictions file, default `output.txt`
- `--confidence` confidence threshold, default `0.6`
- `--window_size` sliding step size, default `400`
- `--chunk_size` chunk length, default `2000`
- `--batch_size` batch size, default `64`
- `--rc` Use bidirectional prediction, default `true`
- `--augment` use BLAST augmentation, default `true`
- `--augment_len` target augmented length, default `4000`
- `--device` `auto`/`cpu`/`cuda`, default `auto`


## Output

- One line per input sequence; fields are space-separated: `sequence_id label confidence`
- `sequence_id`: taken from the FASTA header (text after `>` up to the first space)
- `label`: either `unclassified` or `TaxonName_TaxonLevel`
- `TaxonLevel`: one of `Genus`, `Family`, `Order`, `Class`; if genus confidence is below `--confidence`, the classifier backs off to higher levels
- `confidence`: normalized score in `[0, 1]`, formatted to two decimals; higher means stronger support for the predicted taxon

- Ordering: lines follow the order of sequences in the input FASTA
- Device and augmentation: enabling `--rc` (reverse complement) and `--augment` (BLAST) can change predictions and confidence
- Output file: written to the path specified by `--out` and printed as `Done, please check the output file: <path>`
