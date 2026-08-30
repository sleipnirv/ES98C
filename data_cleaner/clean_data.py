"""最后的数据清洗：合并主库数据 → at%→wt% 换算 → 参比电极归一化 → cleaned_data.csv。

这一步是幂等的变换层：以后新增数据源，把 CSV 丢进 data_sources/ 即可，重跑本文件。

做两件事（与用户确认的「最后 2 项」）：
1. at% → wt%：把 manual 里约 23 行以「原子百分比」记录在 Notes 的成分（如
   `at%: Al72Cr15Ni13`、`at% orig: Fe20 Cr20 Ni20 Mo20 Mn20`）按原子量换算成 wt%，
   填进 24 元素列，并从 Notes 里清掉这段。
2. 参比电极归一化：把 Ref_Electrode 为 Ag/AgCl 的行的 E_corr_mV 统一归到 SCE
   （E(SCE) = E(Ag/AgCl) − 45 mV）。ba/bc 是斜率，与参比无关，不动。

输出：Codes/cleaned_data.csv（39 列，utf-8），供 modeling/model.py 读取。
"""
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CODES = Path(__file__).resolve().parent.parent  # Codes/
sys.path.insert(0, str(CODES))

ELEMENTS = ["B", "C", "N", "Mg", "Al", "Si", "P", "S", "Ti", "V", "Cr", "Mn",
            "Fe", "Co", "Ni", "Cu", "Y", "Nb", "Mo", "Ce", "Gd", "Ta", "W", "Re"]

ATOMIC_WT = {
    "B": 10.81, "C": 12.011, "N": 14.007, "Mg": 24.305, "Al": 26.982,
    "Si": 28.085, "P": 30.974, "S": 32.06, "Ti": 47.867, "V": 50.942,
    "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
    "Cu": 63.546, "Y": 88.906, "Nb": 92.906, "Mo": 95.95, "Ce": 140.116,
    "Gd": 157.25, "Ta": 180.948, "W": 183.84, "Re": 186.207,
}

# 主库数据源：data_sources/ 下的每个 CSV 都是一个源（新库直接丢进去，无需改代码）。
SOURCES_DIR = CODES / "data_sources"
OUTPUT = CODES / "cleaned_data.csv"

# "at%: Al72Cr15Ni13" / "at% orig: Fe20 Cr20 Ni20 Mo20 Mn20" → 整段（到逗号/分号为止）
AT_SEG_RE = re.compile(r"at%\s*(?:orig\s*)?[:\s]*[^,;]+", re.IGNORECASE)
ELEM_NUM_RE = re.compile(r"([A-Z][a-z]?)\s*(\d+(?:\.\d+)?)")


def extract_at_pairs(notes: str):
    """从 Notes 中解析 at% 片段，返回 {元素: at%数值} 或 None。"""
    m = AT_SEG_RE.search(notes)
    if not m:
        return None
    seg = m.group(0)
    seg = re.sub(r"orig", "", seg, flags=re.IGNORECASE)
    seg = re.sub(r"[^A-Za-z0-9.]", " ", seg)  # 只留字母数字和点，其余转空格
    pairs = {}
    for em in ELEM_NUM_RE.finditer(seg):
        el, num = em.group(1), em.group(2)
        if el in ATOMIC_WT:
            pairs[el] = float(num)
    return pairs or None


def at_to_wt(pairs: dict) -> dict:
    """at% → wt%：wt_i = at_i·A_i / Σ(at_j·A_j) × 100。"""
    total = sum(at * ATOMIC_WT[el] for el, at in pairs.items())
    if total <= 0:
        return {}
    return {el: round(at * ATOMIC_WT[el] / total * 100, 2) for el, at in pairs.items()}


def main() -> None:
    srcs = sorted(SOURCES_DIR.glob("*.csv"))
    if not srcs:
        raise SystemExit(f"{SOURCES_DIR} 下没有 CSV 数据源")
    dfs = [pd.read_csv(p, encoding="utf-8") for p in srcs]
    df = pd.concat(dfs, ignore_index=True)

    # ---- 1. at% → wt% ----
    n_at = 0
    for idx, row in df.iterrows():
        notes = "" if pd.isna(row.get("Notes")) else str(row["Notes"])
        pairs = extract_at_pairs(notes)
        if not pairs:
            continue
        n_at += 1
        for el, v in at_to_wt(pairs).items():
            if el in ELEMENTS and pd.isna(df.at[idx, el]):
                df.at[idx, el] = v
        cleaned = AT_SEG_RE.sub("", notes)
        cleaned = re.sub(r"^\s*[,;]\s*", "", cleaned).strip()
        df.at[idx, "Notes"] = cleaned or None

    # ---- 2. 参比电极 Ag/AgCl → SCE ----
    ag = df["Ref_Electrode"].astype(str).str.contains("Ag", case=False, na=False)
    n_ag = int(ag.sum())
    df.loc[ag, "E_corr_mV"] = df.loc[ag, "E_corr_mV"] - 45.0
    df.loc[ag, "Ref_Electrode"] = "SCE"

    # ---- 3. 删除无 i_corr 的行（无目标，无法用于建模） ----
    n_before = len(df)
    df = df[df["I_corr_uA_per_cm2"].notna()].reset_index(drop=True)
    print(f"删除无 i_corr 的行：{n_before} -> {len(df)} 行")

    df.to_csv(OUTPUT, index=False, encoding="utf-8")
    print(f"合并 {len(dfs)} 个数据源（{', '.join(p.name for p in srcs)}）→ {len(df)} 行")
    print(f"at% → wt% 换算：{n_at} 行")
    print(f"参比电极 Ag/AgCl → SCE：{n_ag} 行")
    print(f"写入 {OUTPUT}")


if __name__ == "__main__":
    main()
