"""
Module: dataset.py

Description:
    Implements the TaxonomyDataset class for loading and preprocessing 16S rRNA sequence data in the
    DeepTaxa framework. Handles parsing of plain or gzipped FASTA files, optional plain or gzipped
    taxonomy TSV files, and tokenization using a pretrained DNA-specific tokenizer. Prepares batched,
    tensorized data with aligned numeric labels for hierarchical taxonomy classification, optimized
    for PyTorch's data pipeline.
"""

import gzip
from Bio import SeqIO
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
import logging
from deeptaxa.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

NUCLEOTIDE_MAP = {'A': 0, 'a': 0, 'C': 1, 'c': 1, 'G': 2, 'g': 2, 'T': 3, 't': 3, 'U': 3, 'u': 3}

class TaxonomyDataset(Dataset):
    def __init__(self, fasta_file, taxonomy_file=None, tokenizer_name=DEFAULT_CONFIG["tokenizer_name"], max_length=512, use_raw_labels_for_true=False, encoding="dnabert"):
        """
        Initialize the TaxonomyDataset for sequence classification tasks.

        Workflow:
            1. Load a pretrained tokenizer to encode DNA sequences into token IDs.
            2. Parse FASTA file to extract sequences and their identifiers.
            3. If provided, load taxonomy data from TSV, align it with sequences, and build label mappings.
            4. Assign numeric labels, defaulting to 'Unclassified' for missing entries.

        Args:
            fasta_file (str): Path to FASTA file (plain or gzipped) containing 16S rRNA sequences.
            taxonomy_file (str, optional): Path to TSV file (plain or gzipped) with taxonomy labels.
            tokenizer_name (str): Identifier for pretrained tokenizer (e.g., DNABERT).
            max_length (int): Maximum sequence length for tokenization and padding.
            use_raw_labels_for_true (bool): Include raw string labels alongside numeric IDs for evaluation.
            encoding (str): Sequence encoding method ('dnabert' or 'onehot').

        Raises:
            ValueError: If FASTA file is empty or malformed, or if taxonomy file lacks 'sequence_id'.
        """
        self.encoding = encoding
        self.max_length = max_length
        self.use_raw_labels_for_true = use_raw_labels_for_true

        # Initialize tokenizer only for dnabert encoding
        if encoding == "dnabert":
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        else:
            self.tokenizer = None
            logger.info("Using one-hot nucleotide encoding (no tokenizer loaded)")
        
        # Load sequences from FASTA
        self.sequences, self.seq_ids = self._load_fasta(fasta_file)
        logger.info("Loaded %d sequences from %s", len(self.sequences), fasta_file)
        
        # Initialize taxonomy-related attributes
        self.taxonomy_data = None
        self.taxonomic_ranks = None
        self.labels = None
        self.raw_labels = None
        self.level_label2id = None
        
        # Process taxonomy data if provided
        if taxonomy_file:
            self.taxonomy_data = self._load_taxonomy(taxonomy_file)
            if self.taxonomy_data.empty:
                logger.error("Taxonomy file %s is empty", taxonomy_file)
                raise ValueError("Taxonomy file is empty")
            self.taxonomic_ranks = list(self.taxonomy_data.columns[1:])  # Exclude 'sequence_id'
            # Reference taxonomies often leave the deeper ranks blank when a sequence
            # is only classified partway down the lineage. Fill those blanks with
            # 'Unclassified' now so the label vocabulary and the per-sequence lookup
            # agree; otherwise an empty cell reads back as the string 'nan' and raises
            # a KeyError during alignment.
            for rank in self.taxonomic_ranks:
                values = self.taxonomy_data[rank]
                filled = values.where(values.notna(), 'Unclassified').astype(str).str.strip()
                self.taxonomy_data[rank] = filled.mask(filled.eq(''), 'Unclassified')
            self._build_label_mappings()
            self._align_taxonomy_with_sequences()
            logger.info("Loaded taxonomy data with ranks: %s", self.taxonomic_ranks)

    def _load_fasta(self, fasta_file):
        """
        Parse sequences and IDs from a FASTA file (plain or gzipped).

        Workflow:
            1. Open FASTA file in text mode, using gzip if compressed or standard file handling if plain.
            2. Use BioPython's SeqIO to extract sequence records.
            3. Store sequences as strings and IDs for alignment.

        Args:
            fasta_file (str): Path to FASTA file (plain or gzipped).

        Returns:
            tuple: (list of sequences, list of sequence IDs).

        Raises:
            ValueError: If no sequences are found.
            Exception: If parsing fails due to file issues.
        """
        sequences = []
        seq_ids = []
        try:
            # Check file extension to determine if it's gzipped or plain
            if fasta_file.endswith('.gz'):
                # Handle gzipped FASTA file
                with gzip.open(fasta_file, 'rt') as f:
                    for record in SeqIO.parse(f, 'fasta'):
                        seq_ids.append(record.id)
                        sequences.append(str(record.seq))
            else:
                # Handle plain FASTA file
                with open(fasta_file, 'r') as f:
                    for record in SeqIO.parse(f, 'fasta'):
                        seq_ids.append(record.id)
                        sequences.append(str(record.seq))
        except Exception as e:
            logger.error("Failed to parse FASTA file %s: %s", fasta_file, str(e))
            raise
        if not seq_ids:
            logger.error("No sequences found in FASTA file %s", fasta_file)
            raise ValueError("FASTA file is empty or malformed")
        return sequences, seq_ids

    def _load_taxonomy(self, taxonomy_file):
        """
        Load taxonomy labels from a TSV file (plain or gzipped).

        Workflow:
            1. Open TSV file in text mode, using gzip if compressed or standard file handling if plain.
            2. Read into a pandas DataFrame, expecting 'sequence_id' as the first column.
            3. Validate presence of 'sequence_id' for alignment with FASTA data.

        Args:
            taxonomy_file (str): Path to TSV file (plain or gzipped).

        Returns:
            pandas.DataFrame: Taxonomy data with sequence IDs and rank labels.

        Raises:
            ValueError: If 'sequence_id' column is missing.
            Exception: If file loading fails.
        """
        try:
            # Check file extension to determine if it's gzipped or plain
            if taxonomy_file.endswith('.gz'):
                # Handle gzipped TSV file
                with gzip.open(taxonomy_file, 'rt') as f:
                    df = pd.read_csv(f, sep='\t')
            else:
                # Handle plain TSV file
                with open(taxonomy_file, 'r') as f:
                    df = pd.read_csv(f, sep='\t')
            if 'sequence_id' not in df.columns:
                logger.error("Taxonomy file %s must contain a 'sequence_id' column", taxonomy_file)
                raise ValueError("Missing 'sequence_id' column in taxonomy file")
            return df
        except Exception as e:
            logger.error("Failed to load taxonomy file %s: %s", taxonomy_file, str(e))
            raise

    def _build_label_mappings(self):
        """
        Create mappings from taxonomy labels to numeric IDs for classification.

        Workflow:
            1. For each taxonomic rank, extract unique labels from the taxonomy data.
            2. Assign sequential integer IDs to each unique label.
            3. Store mappings for encoding labels and decoding predictions.

        Notes:
            Numeric IDs are essential for neural network outputs, which predict class indices.
            This step handles the multi-class nature of taxonomy prediction per rank.
        """
        self.level_label2id = {}
        for level, rank in enumerate(self.taxonomic_ranks):
            unique_labels = sorted(set(self.taxonomy_data[rank].dropna().astype(str)))
            self.level_label2id[level] = {label: idx for idx, label in enumerate(unique_labels)}
            logger.debug("Rank %s: %d unique labels", rank, len(unique_labels))

    def _align_taxonomy_with_sequences(self):
        """
        Align taxonomy labels with FASTA sequences.

        Workflow:
            1. Create a dictionary mapping sequence IDs to taxonomy rows.
            2. For each sequence ID, retrieve or assign 'Unclassified' labels.
            3. Store numeric labels and optionally raw string labels.

        Notes:
            Handles missing taxonomy data by defaulting to 'Unclassified', ensuring all sequences
            have labels. This is critical for consistent batching and training stability.
        """
        taxonomy_dict = {row['sequence_id']: row for _, row in self.taxonomy_data.iterrows()}
        
        self.labels = []
        self.raw_labels = []
        for seq_id in self.seq_ids:
            if seq_id in taxonomy_dict:
                taxonomy_row = taxonomy_dict[seq_id]
                # Store raw labels for debugging or evaluation
                raw_label_dict = {rank: str(taxonomy_row[rank]) for rank in self.taxonomic_ranks}
                self.raw_labels.append(raw_label_dict)
                # Convert to numeric IDs
                label_list = [self.level_label2id[level][str(taxonomy_row[rank])] 
                              for level, rank in enumerate(self.taxonomic_ranks)]
                self.labels.append(label_list)
            else:
                # Default to 'Unclassified' for missing entries
                logger.warning("Sequence ID %s not found in taxonomy file; using 'Unclassified'", seq_id)
                unclassified_dict = {rank: 'Unclassified' for rank in self.taxonomic_ranks}
                self.raw_labels.append(unclassified_dict)
                label_list = []
                for level, rank in enumerate(self.taxonomic_ranks):
                    if 'Unclassified' not in self.level_label2id[level]:
                        self.level_label2id[level]['Unclassified'] = len(self.level_label2id[level])
                    label_list.append(self.level_label2id[level]['Unclassified'])
                self.labels.append(label_list)
        
        # Validate alignment
        if len(self.labels) != len(self.sequences):
            logger.error("Mismatch between labels (%d) and sequences (%d)", len(self.labels), len(self.sequences))
            raise ValueError("Labels and sequences length mismatch")

    def __len__(self):
        """
        Return the total number of sequences in the dataset.

        Returns:
            int: Dataset size, used by DataLoader for iteration.
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Retrieve a preprocessed data item for a given index.

        Workflow:
            1. Tokenize the sequence with truncation and padding to max_length.
            2. Package tokenized inputs (input_ids, attention_mask) with sequence ID.
            3. Include numeric labels and optionally raw labels if taxonomy data exists.

        Args:
            idx (int): Index of the sequence to retrieve.

        Returns:
            dict: Preprocessed item with tensors for model input.
        """
        sequence = self.sequences[idx]
        seq_id = self.seq_ids[idx]

        if self.encoding == "onehot":
            # Build one-hot encoded tensor directly from nucleotide sequence
            onehot = torch.zeros(self.max_length, 4, dtype=torch.float32)
            seq_len = min(len(sequence), self.max_length)
            for pos in range(seq_len):
                nt_idx = NUCLEOTIDE_MAP.get(sequence[pos])
                if nt_idx is not None:
                    onehot[pos, nt_idx] = 1.0
            attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            attention_mask[:seq_len] = 1
            item = {
                'input_ids': onehot.transpose(0, 1),  # [4, max_length] for Conv1d
                'attention_mask': attention_mask,
                'seq_ids': seq_id
            }
        else:
            # Tokenize sequence into input_ids and attention_mask
            encoding = self.tokenizer(
                sequence,
                max_length=self.max_length,
                truncation=True,  # Truncate sequences exceeding max_length
                padding='max_length',  # Pad shorter sequences with zeros
                return_tensors='pt'  # Return PyTorch tensors
            )

            # Remove extra batch dimension from tokenizer output
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'seq_ids': seq_id
            }

        # Carry the raw sequence length so downstream code can report it without
        # having to map a batch position back to the dataset.
        item['seq_length'] = len(sequence)

        # Add labels if taxonomy data is available
        if self.taxonomy_data is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
            if self.use_raw_labels_for_true:
                item['raw_labels'] = self.raw_labels[idx]

        return item

def custom_collate_fn(batch):
    """
    Collate a list of data items into a batch for DataLoader.

    Workflow:
        1. Stack input_ids and attention_mask tensors across the batch.
        2. Collect sequence IDs as a list.
        3. Stack labels if present, and include raw labels as a list if requested.

    Args:
        batch (list): List of dicts from __getitem__.

    Returns:
        dict: Batched data with stacked tensors, ready for model input.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    seq_ids = [item['seq_ids'] for item in batch]
    seq_lengths = [item['seq_length'] for item in batch]

    collated = {'input_ids': input_ids, 'attention_mask': attention_mask, 'seq_ids': seq_ids, 'seq_lengths': seq_lengths}
    
    if 'labels' in batch[0]:
        collated['labels'] = torch.stack([item['labels'] for item in batch])
    if 'raw_labels' in batch[0]:
        collated['raw_labels'] = [item['raw_labels'] for item in batch]

    return collated

if __name__ == "__main__":
    pass

