#!/usr/bin/env Rscript
# Step 2. Infer V4 ASVs from the raw paired-end reads with DADA2.
# DADA2 performs its own quality filtering, so fastp is not used on this route.
# NOTE: the input reads must already have the 515F/806R primers removed. The deposited
#       PRJNA566436 runs are primer-trimmed (R1 begins at the conserved base internal to 515F,
#       R2 at the 806R-adjacent region, and the resulting ASVs are 253 bp, the primer-free V4
#       insert length), so filterAndTrim below does no primer removal (no trimLeft). If you
#       substitute reads that still carry the primers, strip them first (for example with
#       cutadapt, or a trimLeft in filterAndTrim); otherwise the ASVs will be about 292 bp and
#       will misclassify.
# Inputs:  raw FASTQ in $RAW (default "raw"), sample list in $OUT/samples.tsv
# Outputs: asv_seqs.fasta, asv_counts.tsv, track_reads.tsv in $OUT (default "../data")
suppressMessages(library(dada2))
set.seed(42)

RAW  <- Sys.getenv("RAW", "raw")
OUT  <- Sys.getenv("OUT", "../data")
FILT <- "filtered"
dir.create(FILT, showWarnings = FALSE)

meta    <- read.delim(file.path(OUT, "samples.tsv"))
samples <- meta$sample
fnF <- file.path(RAW, paste0(samples, "_1.fastq.gz"))
fnR <- file.path(RAW, paste0(samples, "_2.fastq.gz"))
filtF <- file.path(FILT, paste0(samples, "_F.fq.gz"))
filtR <- file.path(FILT, paste0(samples, "_R.fq.gz"))
names(filtF) <- samples; names(filtR) <- samples

# No primer removal here: the input reads are already primer-trimmed (see the header note).
ft <- filterAndTrim(fnF, filtF, fnR, filtR,
                    truncLen = 0, maxN = 0, maxEE = c(2, 2), truncQ = 2,
                    minLen = 50, rm.phix = TRUE, compress = TRUE, multithread = TRUE)

errF <- learnErrors(filtF, multithread = TRUE)
errR <- learnErrors(filtR, multithread = TRUE)
ddF  <- dada(filtF, err = errF, multithread = TRUE, pool = "pseudo")
ddR  <- dada(filtR, err = errR, multithread = TRUE, pool = "pseudo")
mg   <- mergePairs(ddF, filtF, ddR, filtR, verbose = TRUE)
seqtab <- makeSequenceTable(mg)
st <- removeBimeraDenovo(seqtab, method = "consensus", multithread = TRUE, verbose = TRUE)

# Name ASVs and write the feature table
seqs <- getSequences(st)
ids  <- paste0("ASV", seq_along(seqs))
writeLines(as.vector(rbind(paste0(">", ids), seqs)), file.path(OUT, "asv_seqs.fasta"))
ct <- t(st); rownames(ct) <- ids
write.table(data.frame(ASV = ids, ct, check.names = FALSE),
            file.path(OUT, "asv_counts.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# Read-tracking table
getN  <- function(x) sum(getUniques(x))
track <- cbind(ft,
               denoisedF = sapply(ddF, getN), denoisedR = sapply(ddR, getN),
               merged = sapply(mg, getN), nonchim = rowSums(st))
colnames(track)[1:2] <- c("input", "filtered")
write.table(data.frame(sample = samples, track),
            file.path(OUT, "track_reads.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

cat("DONE:", length(seqs), "ASVs;",
    sum(st), "non-chimeric reads of", sum(ft[, 1]), "input\n")
