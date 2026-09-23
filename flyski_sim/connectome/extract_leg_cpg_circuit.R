# 업무4: 다리 CPG 회로 신경군 추출 (male-cns:v1.0)
#
# 04_connectome_interface.md 1절("신경군 추출") 구현. 전체 절차:
#   1) bioRxiv 10.1101/2025.09.12.675944 (Pugliese et al.)가 언급한 하행뉴런
#      DNg100/DNb08를 male-cns:v1.0에서 타입명으로 직접 조회 (확인 완료, 2026-09-23).
#   2) DNg100의 시냅스 하류를 superclass=='vnc_intrinsic'(다리 신경분절 소속)으로
#      필터링해 CPG 인터뉴런 후보를 탐색 — IN09A002/IN12B003(억제성, gaba),
#      IN01A015/IN17A001/dPR1(흥분성, acetylcholine)을 발견.
#   3) 이 문서에서: 후보 인터뉴런들 사이의 실제 연결과, 그 하류의 진짜 다리
#      운동뉴런(vnc_motor)까지 전부 추출해 하나의 회로 그래프로 저장.
#
# 주의: 이 회로가 원 논문이 말한 "정확히 그 3-뉴런 회로"인지는 논문 원문과
# 직접 대조하지 못했다 — RESEARCH_NOTES.md 13번 참고. 그래도 아래 증거는 강력하다:
#   - IN17A001(흥분성) -> IN09A002(억제성) 연결이 T1/T2/T3 전부에서 압도적으로
#     강함(196/261/431), 후방으로 갈수록 강해짐.
#   - DNg100이 5개 인터뉴런 후보 전부에 강하게 직접 연결됨(146~456).
#   - IN09A002가 실제 이름 붙은 다리 운동뉴런(MNhl29, Fe reductor MN,
#     Sternal anterior rotator MN 등)에 직접 연결됨 — "회로 -> 실제 근육"까지
#     끊기지 않고 이어짐을 확인.

library(malecns)
library(neuprintr)

OUT_DIR <- "flyski_sim/connectome"
mcns_conn <- mcns_neuprint()

# ---- 1) 후보 신경군 메타데이터 ----
candidate_types <- c("DNg100", "DNb08", "IN09A002", "IN12B003", "IN01A015",
                     "IN17A001", "dPR1")
candidates <- do.call(rbind, lapply(candidate_types, function(t) {
  mcns_neuprint_meta(sprintf("/%s.*", t))
}))
candidates$role <- ifelse(candidates$type %in% c("DNg100", "DNb08"), "afferent_command",
                    ifelse(candidates$predictedNt == "gaba", "intrinsic_inhibitory",
                                                              "intrinsic_excitatory"))
write.csv(candidates, file.path(OUT_DIR, "01_candidate_neurons.csv"), row.names = FALSE)
cat("저장:", file.path(OUT_DIR, "01_candidate_neurons.csv"), "- n =", nrow(candidates), "\n")

# ---- 2) 후보군 내부 연결 (DN -> 인터뉴런, 인터뉴런 -> 인터뉴런) ----
all_ids <- candidates$bodyid
conn_internal <- neuprint_connection_table(all_ids, prepost = "POST", by.roi = FALSE,
                                           conn = mcns_conn)
conn_internal <- conn_internal[conn_internal$partner %in% all_ids, ]
write.csv(conn_internal, file.path(OUT_DIR, "02_internal_connectivity.csv"),
         row.names = FALSE)
cat("저장:", file.path(OUT_DIR, "02_internal_connectivity.csv"),
   "- n_edges =", nrow(conn_internal), "\n")

# ---- 3) 인터뉴런 -> 실제 다리 운동뉴런(efferent) 하류 추출 ----
interneuron_ids <- candidates$bodyid[candidates$role != "afferent_command"]
conn_to_motor <- neuprint_connection_table(interneuron_ids, prepost = "POST",
                                           by.roi = FALSE, conn = mcns_conn)
motor_meta <- mcns_neuprint_meta(unique(conn_to_motor$partner))
motor_meta <- motor_meta[motor_meta$superclass == "vnc_motor" &
                         !is.na(motor_meta$superclass), ]
conn_to_motor <- conn_to_motor[conn_to_motor$partner %in% motor_meta$bodyid, ]

write.csv(motor_meta[, c("bodyid", "type", "somaNeuromere", "predictedNt")],
         file.path(OUT_DIR, "03_leg_motor_neurons.csv"), row.names = FALSE)
write.csv(conn_to_motor, file.path(OUT_DIR, "04_interneuron_to_motor_connectivity.csv"),
         row.names = FALSE)
cat("저장:", file.path(OUT_DIR, "03_leg_motor_neurons.csv"),
   "- n_motor_neurons =", nrow(motor_meta), "\n")
cat("저장:", file.path(OUT_DIR, "04_interneuron_to_motor_connectivity.csv"),
   "- n_edges =", nrow(conn_to_motor), "\n")

cat("\n데이터셋 버전: male-cns:v1.0 (neuprint.janelia.org), 추출일: 2026-09-23\n")
