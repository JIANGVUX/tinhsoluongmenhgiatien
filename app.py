import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Tính mệnh giá phát lương từ cột 'Còn được nhận' (bỏ dòng TỔNG)"

# Mệnh giá bạn yêu cầu (VND)
DENOMS_VND_DEFAULT = [500_000, 100_000, 50_000, 30_000, 20_000, 10_000, 5_000, 2_000, 1_000]


# =========================
# Utils: parse money
# =========================
def _clean_money_str(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\d\.,\-]", "", s)
    s = s.strip(".,")
    return s


def parse_money_to_int_vnd(x) -> Tuple[int, str]:
    """
    Parse money values:
    - int/float
    - '5,498,000' or '5.498.000' or '5 498 000đ'
    - returns (int_vnd, note)
    """
    if pd.isna(x):
        return 0, "NaN -> 0"

    if isinstance(x, (int, np.integer)):
        return int(x), ""
    if isinstance(x, (float, np.floating)):
        if abs(x - round(x)) < 1e-6:
            return int(round(x)), ""
        return int(round(x)), "float -> rounded"

    s0 = str(x)
    s = _clean_money_str(s0)
    if s in ("", "-", ".", ","):
        return 0, f"invalid '{s0}' -> 0"

    try:
        if "." in s and "," in s:
            last_dot = s.rfind(".")
            last_com = s.rfind(",")
            if last_com > last_dot:
                tmp = s.replace(".", "").replace(",", ".")
                val = float(tmp)
            else:
                tmp = s.replace(",", "")
                val = float(tmp)
            return int(round(val)), "mixed sep -> parsed"

        if "," in s:
            parts = s.split(",")
            if all(len(p) == 3 for p in parts[1:]):
                return int("".join(parts)), ""
            val = float(s.replace(",", "."))
            return int(round(val)), "comma decimal -> rounded"

        if "." in s:
            parts = s.split(".")
            if all(len(p) == 3 for p in parts[1:]):
                return int("".join(parts)), ""
            val = float(s)
            return int(round(val)), "dot decimal -> rounded"

        return int(s), ""
    except Exception:
        return 0, f"parse error '{s0}' -> 0"


def apply_rounding_1000(amount_vnd: int, mode: str) -> Tuple[int, int]:
    """
    Returns (amount_used_vnd, remainder_vnd<1000)
    """
    rem = amount_vnd % 1000
    if rem == 0:
        return amount_vnd, 0

    if mode == "Giữ nguyên (báo dư)":
        return amount_vnd - rem, rem
    if mode == "Làm tròn xuống 1.000":
        return amount_vnd - rem, rem
    if mode == "Làm tròn lên 1.000":
        return amount_vnd + (1000 - rem), rem
    # nearest
    if rem >= 500:
        return amount_vnd + (1000 - rem), rem
    return amount_vnd - rem, rem


# =========================
# Read payroll template: auto-detect header by 'Còn được nhận'
# =========================
def read_payroll_like_template(uploaded_file, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(uploaded_file, sheet_name=sheet_name, header=None, dtype=object)

    target = "còn được nhận"
    header_row = None
    for i in range(len(raw)):
        row_lower = raw.iloc[i].astype(str).str.strip().str.lower()
        if (row_lower == target).any():
            header_row = i
            break
    if header_row is None:
        raise ValueError("Không tìm thấy dòng tiêu đề chứa 'Còn được nhận'.")

    # subheader row is next row (thường chứa 'Số giờ', 'Thành tiền', 'Số suất'...)
    sub_row = header_row + 1 if header_row + 1 < len(raw) else None

    main = raw.iloc[header_row].ffill()
    sub = raw.iloc[sub_row].fillna("") if sub_row is not None else pd.Series([""] * raw.shape[1])

    cols = []
    for m, s in zip(main, sub):
        m = str(m).strip() if pd.notna(m) else ""
        s = str(s).strip() if pd.notna(s) else ""
        if s and s.lower() != "nan" and m:
            cols.append(f"{m} ({s})")
        else:
            cols.append(m)

    data_start = header_row + 2 if sub_row is not None else header_row + 1
    data = raw.iloc[data_start:].copy()
    data.columns = cols
    data = data.dropna(how="all")

    return data


# =========================
# Optimal coin change (DP)
# =========================
@st.cache_data(show_spinner=False)
def build_dp(denoms_k: List[int], max_amount_k: int) -> np.ndarray:
    """
    DP for minimal number of notes, tie-break prefer larger coin.
    Returns prev_coin array to reconstruct.
    """
    denoms_k = sorted(set(int(d) for d in denoms_k if d > 0))
    INF = 10**9
    dp = np.full(max_amount_k + 1, INF, dtype=np.int32)
    prev = np.zeros(max_amount_k + 1, dtype=np.int32)
    dp[0] = 0

    for a in range(1, max_amount_k + 1):
        best = INF
        best_coin = 0
        for c in denoms_k:
            if c > a:
                break
            cand = dp[a - c] + 1
            if cand < best or (cand == best and c > best_coin):
                best = cand
                best_coin = c
        dp[a] = best
        prev[a] = best_coin

    return prev


def reconstruct_counts(amount_k: int, prev: np.ndarray, denoms_k_set: set) -> Dict[int, int]:
    counts = {d: 0 for d in denoms_k_set}
    a = int(amount_k)
    while a > 0:
        c = int(prev[a])
        if c <= 0:
            break
        counts[c] += 1
        a -= c
    return counts


def to_excel_bytes(df_detail: pd.DataFrame, df_summary: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df_detail.to_excel(writer, index=False, sheet_name="Chi_tiet")
        df_summary.to_excel(writer, index=False, sheet_name="Tong_hop")

        # Auto width
        for sheet_name, df in [("Chi_tiet", df_detail), ("Tong_hop", df_summary)]:
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                width = max(10, min(45, int(df[col].astype(str).str.len().max() if len(df) else len(col)) + 2))
                ws.set_column(i, i, width)

    return bio.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

uploaded = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Bạn upload file Excel lên để bắt đầu.")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet = st.selectbox("Chọn sheet", options=xls.sheet_names, index=0)

with st.sidebar:
    st.header("Thiết lập mệnh giá")
    use_30k = st.checkbox("Dùng mệnh giá 30k", value=True)

    denoms = [500_000, 100_000, 50_000, 20_000, 10_000, 5_000, 2_000, 1_000]
    if use_30k:
        denoms.insert(3, 30_000)  # sau 50k
    denoms = sorted(denoms, reverse=True)

    st.caption("Mệnh giá đang dùng:")
    st.code(", ".join([f"{d//1000}k" for d in denoms]))

    rounding_mode = st.selectbox(
        "Nếu tiền không chia hết 1.000 thì xử lý:",
        ["Giữ nguyên (báo dư)", "Làm tròn xuống 1.000", "Làm tròn lên 1.000", "Làm tròn gần nhất 1.000"],
        index=0,
    )

try:
    df = read_payroll_like_template(uploaded, sheet)
except Exception as e:
    st.error(f"Lỗi đọc template: {e}")
    st.stop()

# BẮT BUỘC: cột này để tính
money_col = "Còn được nhận"
required_cols = ["Stt", "Mã NV", "Họ và tên", money_col]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Thiếu cột cần thiết: {missing}. Vui lòng kiểm tra file.")
    st.stop()

# Loại bỏ dòng TỔNG & ghi chú: lọc STT numeric
df["_stt_num"] = pd.to_numeric(df["Stt"], errors="coerce")
df = df[df["_stt_num"].notna()].copy()

# Parse money
money_vals = []
money_notes = []
for v in df[money_col].tolist():
    val, note = parse_money_to_int_vnd(v)
    money_vals.append(val)
    money_notes.append(note)

df["_money_raw"] = money_vals
df["_parse_note"] = money_notes

# Rounding to 1000
used = []
rem = []
for amt in df["_money_raw"].astype(int).tolist():
    u, r = apply_rounding_1000(int(amt), rounding_mode)
    used.append(u)
    rem.append(r)
df["_money_used"] = used
df["_remainder"] = rem

# Filters
st.subheader("1) Lọc danh sách lao động")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    q = st.text_input("Tìm theo tên / mã NV", "")
with c2:
    min_v = st.number_input("Tiền >= (VND)", min_value=0, value=0, step=1000)
with c3:
    max_v = st.number_input("Tiền <= (VND) (0 = không giới hạn)", min_value=0, value=0, step=1000)

mask = df["_money_raw"] >= int(min_v)
if int(max_v) > 0:
    mask &= df["_money_raw"] <= int(max_v)
if q.strip():
    q2 = q.strip().lower()
    mask &= (df["Họ và tên"].astype(str).str.lower().str.contains(q2, na=False) |
             df["Mã NV"].astype(str).str.lower().str.contains(q2, na=False))

df_f = df.loc[mask].copy()
st.caption(f"Số lao động sau lọc: **{len(df_f)}** (đã bỏ dòng TỔNG chuẩn).")

if df_f.empty:
    st.warning("Không có dữ liệu sau khi lọc.")
    st.stop()

# Build DP once
denoms_k_desc = [d // 1000 for d in denoms]             # desc
denoms_k_sorted = sorted(set(denoms_k_desc))            # asc for DP
max_k = int(df_f["_money_used"].max()) // 1000
prev = build_dp(denoms_k_sorted, max_k)
denoms_k_set = set(denoms_k_sorted)

# Compute counts
labels = [f"{d//1000}k" for d in denoms]
counts_rows = []
total_notes = []
check_sum = []

for amt in df_f["_money_used"].astype(int).tolist():
    ak = amt // 1000
    counts = reconstruct_counts(ak, prev, denoms_k_set)

    row = []
    s = 0
    n = 0
    for dk in denoms_k_desc:
        c = int(counts.get(dk, 0))
        row.append(c)
        s += c * dk * 1000
        n += c
    counts_rows.append(row)
    total_notes.append(n)
    check_sum.append(s)

df_counts = pd.DataFrame(counts_rows, columns=labels)
out = pd.concat(
    [df_f[["Stt", "Mã NV", "Họ và tên", money_col, "_money_raw", "_money_used", "_remainder", "_parse_note"]].reset_index(drop=True),
     df_counts],
    axis=1
)
out["Tổng_số_tờ"] = total_notes
out["Kiểm_tra_tổng(VND)"] = check_sum
out["Sai_lệch(VND)"] = out["Kiểm_tra_tổng(VND)"].astype(int) - out["_money_used"].astype(int)

st.subheader("2) Kết quả mệnh giá theo từng lao động")
st.caption("Ghi chú: nếu có số lẻ < 1.000 thì `_remainder` sẽ hiện phần dư (và `_money_used` là số tiền đem đi chia mệnh giá).")
st.dataframe(out, use_container_width=True, height=560)

# Summary (tổng số tờ cần chuẩn bị) - chỉ tính theo lao động sau lọc
st.subheader("3) Tổng hợp số tờ cần chuẩn bị (không tính dòng TỔNG trong file)")
sum_notes = {lab: int(out[lab].sum()) for lab in labels}
summary_rows = []
for d, lab in zip(denoms, labels):
    summary_rows.append({"Mệnh_giá": lab, "Số_tờ": sum_notes[lab], "Thành_tiền(VND)": sum_notes[lab] * d})
df_summary = pd.DataFrame(summary_rows)
df_summary.loc[len(df_summary)] = {
    "Mệnh_giá": "TỔNG",
    "Số_tờ": int(df_summary["Số_tờ"].sum()),
    "Thành_tiền(VND)": int(df_summary["Thành_tiền(VND)"].sum())
}

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Tổng người (sau lọc)", len(out))
with m2:
    st.metric("Tổng số tờ", int(df_summary.loc[df_summary["Mệnh_giá"] == "TỔNG", "Số_tờ"].iloc[0]))
with m3:
    st.metric("Tổng tiền (VND)", int(df_summary.loc[df_summary["Mệnh_giá"] == "TỔNG", "Thành_tiền(VND)"].iloc[0]))

st.dataframe(df_summary, use_container_width=True, height=360)

# Download
st.subheader("4) Tải file kết quả")
excel_bytes = to_excel_bytes(out, df_summary)
st.download_button(
    "Tải Excel (chi tiết + tổng hợp)",
    data=excel_bytes,
    file_name="ket_qua_menh_gia_phat_luong.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
