# natverse/malecns 설치 (ENVIRONMENT.md 2절의 우회 절차 그대로).
# natmanager::install()은 pak 바이너리 비호환으로 실패하므로 remotes로 직접 설치한다.
options(repos = c(CRAN = "https://cloud.r-project.org"))
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
remotes::install_github("natverse/malecns", dependencies = TRUE, upgrade = "never")

library(malecns)
if (nchar(Sys.getenv("NEUPRINT_TOKEN")) == 0) {
  message("NEUPRINT_TOKEN 이 비어 있음 — ~/.Renviron 에 NEUPRINT_TOKEN=... 을 추가하고 R을 재시작할 것")
} else {
  print(head(mcns_neuprint_meta("/DNg100.*")))
}
