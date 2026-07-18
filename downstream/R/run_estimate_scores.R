args <- commandArgs(trailingOnly = TRUE)

parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key))
    }
    if (i == length(args)) {
      stop(sprintf("Missing value for argument: %s", key))
    }
    out[[substring(key, 3)]] <- args[[i + 1]]
    i <- i + 2
  }
  out
}

opts <- parse_args(args)
required <- c("input_csv", "output_csv")
missing_keys <- required[!required %in% names(opts)]
if (length(missing_keys) > 0) {
  stop(sprintf("Missing required args: %s", paste(missing_keys, collapse = ", ")))
}

suppressPackageStartupMessages({
  library(data.table)
  library(estimate)
})

expr <- fread(opts$input_csv, data.table = FALSE, check.names = FALSE)
if (!"gene_symbol" %in% colnames(expr)) {
  stop("Input CSV must contain gene_symbol column")
}

gene_symbols <- expr$gene_symbol
expr$gene_symbol <- NULL
mat <- as.matrix(expr)
mode(mat) <- "numeric"
rownames(mat) <- gene_symbols

original_sample_ids <- colnames(mat)
safe_sample_ids <- sprintf("sample_%05d", seq_along(original_sample_ids))
colnames(mat) <- safe_sample_ids

in_f <- tempfile(fileext = ".gct")
common_f <- tempfile(fileext = ".gct")
out_f <- tempfile(fileext = ".gct")

estimate::outputGCT(as.data.frame(mat, check.names = FALSE), in_f)
estimate::filterCommonGenes(input.f = in_f, output.f = common_f, id = "GeneSymbol")
estimate::estimateScore(input.ds = common_f, output.ds = out_f, platform = "affymetrix")

est <- read.delim(out_f, skip = 2, check.names = FALSE)
score_names <- est$NAME
est$NAME <- NULL
if ("Description" %in% colnames(est)) {
  est$Description <- NULL
}
if (ncol(est) == length(original_sample_ids) + 1) {
  est <- est[, -1, drop = FALSE]
}
if (ncol(est) != length(original_sample_ids)) {
  stop(sprintf("ESTIMATE output sample count mismatch: expected %d, got %d", length(original_sample_ids), ncol(est)))
}

colnames(est) <- original_sample_ids
score_mat <- as.matrix(est)
mode(score_mat) <- "numeric"
rownames(score_mat) <- score_names
sample_df <- data.frame(sample_id = colnames(score_mat), t(score_mat), check.names = FALSE)

wanted <- c("ImmuneScore", "StromalScore", "ESTIMATEScore", "TumorPurity")
missing_scores <- wanted[!wanted %in% colnames(sample_df)]
if (length(missing_scores) > 0) {
  stop(sprintf("ESTIMATE output missing expected scores: %s", paste(missing_scores, collapse = ", ")))
}

sample_df <- sample_df[, c("sample_id", wanted), drop = FALSE]
dir.create(dirname(opts$output_csv), showWarnings = FALSE, recursive = TRUE)
write.csv(sample_df, opts$output_csv, row.names = FALSE)
