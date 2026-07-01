#!/usr/bin/env Rscript
# Step 4. Classify the ASVs with SILVA v138 using the DADA2 naive Bayes classifier,
# as an independent cross-check of the DeepTaxa genus calls. Seeded for reproducibility.
# Inputs:  $DATA/asv_seqs.fasta and the SILVA training set at $SILVA_REF
# Output:  $DATA/asv_taxonomy_silva.tsv
suppressMessages(library(dada2))
set.seed(100)

DATA <- Sys.getenv("DATA", "../data")
REF  <- Sys.getenv("SILVA_REF", "silva_nr99_v138_train_set.fa.gz")

seqs <- getSequences(file.path(DATA, "asv_seqs.fasta"))
tt <- assignTaxonomy(unname(seqs), REF, multithread = TRUE, tryRC = TRUE, minBoot = 50)
out <- data.frame(ASV = names(seqs), tt, check.names = FALSE)
write.table(out, file.path(DATA, "asv_taxonomy_silva.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

cat("DONE:", nrow(out), "ASVs classified with SILVA v138\n")
