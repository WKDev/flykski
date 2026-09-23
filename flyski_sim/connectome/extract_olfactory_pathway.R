# 업무4: 후각 스티어링 경로 신경군 추출 (male-cns:v1.0)
#
# 04_connectome_interface.md 1절("후각 입력군", "후각-조향 연결군") 구현.
# DA1 글로머룰루스(수컷 페로몬 cVA 관련 — male CNS라 이 채널을 대표 후각
# 입력으로 선택)의 투사뉴런(PN)에서 출발해 하행뉴런(descending)까지, 그리고
# 그 하행뉴런이 우리가 이미 찾은 다리 CPG 명령 뉴런(DNg100/DNb08,
# extract_leg_cpg_circuit.R)에 직접 연결되는지까지 추적한다.
#
# **핵심 발견**: DA1 PN -> DNpe052/DNp29/DNp32(하행뉴런) -> DNg100/DNb08 로
# 단 2홉 만에 다리 CPG 명령 뉴런까지 직접 이어진다. 즉 "냄새 -> 조향" 경로가
# "냄새 -> 다리 걷기 명령"과 뉴런 수준에서 거의 같은 병목(DNg100/DNb08)을
# 공유한다 — 이게 실제로 좌우 비대칭 조향에 쓸 수 있는지는 좌우(side) 정보가
# 없어서 아직 확인 못 함(RESEARCH_NOTES.md 20번과 같은 한계).

library(malecns)
library(neuprintr)

OUT_DIR <- "flyski_sim/connectome"
mcns_conn <- mcns_neuprint()

# ---- 1) DA1 PN (대표 후각 채널) ----
da1_pn <- mcns_neuprint_meta("/DA1_.+PN")
da1_pn$role <- "olfactory_input"
write.csv(da1_pn[, c("bodyid", "type", "predictedNt", "role")],
         file.path(OUT_DIR, "05_olfactory_pn.csv"), row.names = FALSE)
cat("저장: 05_olfactory_pn.csv - n =", nrow(da1_pn), "\n")

# ---- 2) DA1 PN 하류 전체 + 하행뉴런(descending) 필터 ----
conn_pn <- neuprint_connection_table(da1_pn$bodyid, prepost = "POST", by.roi = FALSE,
                                     conn = mcns_conn)
partner_meta <- mcns_neuprint_meta(unique(conn_pn$partner))
dns <- partner_meta[partner_meta$superclass == "descending_neuron" &
                    !is.na(partner_meta$superclass), ]
dns$role <- "olfactory_descending"
write.csv(dns[, c("bodyid", "type", "predictedNt", "somaNeuromere", "role")],
         file.path(OUT_DIR, "06_olfactory_descending_neurons.csv"), row.names = FALSE)

conn_pn_to_dn <- conn_pn[conn_pn$partner %in% dns$bodyid, ]
write.csv(conn_pn_to_dn, file.path(OUT_DIR, "07_pn_to_descending_connectivity.csv"),
         row.names = FALSE)
cat("저장: 06_olfactory_descending_neurons.csv - n =", nrow(dns), "\n")
cat("저장: 07_pn_to_descending_connectivity.csv - n_edges =", nrow(conn_pn_to_dn), "\n")

# ---- 3) 이 하행뉴런 -> 다리 CPG 후보군(DNg100/DNb08 등) 직접 연결 ----
leg_cpg <- read.csv(file.path(OUT_DIR, "01_candidate_neurons.csv"))
conn_dn <- neuprint_connection_table(dns$bodyid, prepost = "POST", by.roi = FALSE,
                                     conn = mcns_conn)
conn_dn_to_cpg <- conn_dn[conn_dn$partner %in% leg_cpg$bodyid, ]
write.csv(conn_dn_to_cpg,
         file.path(OUT_DIR, "08_olfactory_descending_to_leg_cpg_connectivity.csv"),
         row.names = FALSE)
cat("저장: 08_olfactory_descending_to_leg_cpg_connectivity.csv - n_edges =",
   nrow(conn_dn_to_cpg), "\n")

cat("\n=== 요약: 후각(DA1 PN) -> 하행뉴런 -> 다리 CPG(DNg100/DNb08) 직접 연결 ===\n")
summary_tbl <- merge(conn_dn_to_cpg, dns[, c("bodyid", "type")], by = "bodyid")
summary_tbl <- merge(summary_tbl, leg_cpg[, c("bodyid", "type")],
                     by.x = "partner", by.y = "bodyid", suffixes = c("_dn", "_cpg"))
print(summary_tbl[order(-summary_tbl$weight),
                  c("type_dn", "type_cpg", "weight")])

cat("\n데이터셋 버전: male-cns:v1.0 (neuprint.janelia.org), 추출일: 2026-09-23\n")
