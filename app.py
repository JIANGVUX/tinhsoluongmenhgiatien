import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Tính mệnh giá phát lương (gọn: Cần lấy + Tổng hợp)"

# Mệnh giá theo yêu cầu (VND)
DENOMS_VND = [500_000, 100_000, 50_000, 30_000, 20_000, 10_000, 5_000, 2_000, 1_000]


# =========================
# Parse tiền (an toàn nhiều format)
# =========================
def _clean_money_str(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\d\.,\-]", "", s)
    return s.strip("., ")


def parse_money_to_int_vnd(x) -> Tuple[int, str]:
    if pd.isna(x):
        return 0, "NaN->0"

    if isinstance(x, (int, np.integer)):
        return int(x), ""
    if isinstance(x, (float, np.floating)):
        if abs(x - round(x)) < 1e-6:
            return int(round(x)), ""
        return int(round(x)), "float->rounded"

    s0 = str(x)
    s = _clean_money_str(s0)
    if s in ("", "-", ".", ","):
        return 0, f"invalid '{s0}'"

    try:
        if "." in s and "," in s:
            last_dot, last_com = s.rfind("."), s.rfind(",")
            if last_com > last_dot:
                tmp = s.replace(".", "").replace(",", ".")
                return int(round(float(tmp))), "mixed->parsed"
            tmp = s.replace(",", "")
            return int(round(float(tmp))), "mixed->parsed"

        if "," in s:
            parts = s.split(",")
            if all(len(p) == 3 for p in parts[1:]):
                return int("".join(parts)), ""
            return int(round(float(s.replace(",", ".")))), "comma-decimal"

        if "." in s:
            parts = s.split(".")
            if all(len(p) == 3 for p in parts[1:]):
                return int("".join(parts)), ""
            return int(round(float(s))), "dot-decimal"

        return int(s), ""
    except Exception:
        return 0, f"parse-error '{s0}'"


def round_to_1000(amount_vnd: int, mode: str) -> Tuple[int, int]:
    rem = amount_vnd % 1000
    if rem == 0:
        return amount_vnd, 0

    if mode == "Giữ nguyên (báo dư)":
        return amount_vnd - rem, rem
    if mode == "Làm tròn xuống 1.000":
        return amount_vnd - rem, rem
    if mode == "Làm tròn lên 1.000":
        return amount_vnd + (1000 - rem), rem

    # gần nhất
    if rem >= 500:
        return amount_vnd + (1000 - rem), rem
    return amount_vnd - rem, rem


# =========================
# Đọc template: tự tìm dòng tiêu đề theo "Còn được nhận"
# =========================
def read_payroll_template(file, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(file, sheet_name=sheet_name, header=None, dtype=object)
    target = "còn được nhận"

    header_row = None
    for i in range(len(raw)):
        row_lower = raw.iloc[i].astype(str).str.strip().str.lower()
        if (row_lower == target).any():
            header_row = i
            break
    if header_row is None:
        raise ValueError("Không tìm thấy dòng tiêu đề chứa 'Còn được nhận'.")

    sub_row = header_row + 1 if header_row + 1 < len(raw) else None
    main = raw.iloc[header_row].ffill()
    sub = raw.iloc[sub_row].fillna("") if sub_row is not None else pd.Series([""] * raw.shape[1])

    cols = []
    for m, s in zip(main, sub):
        m = str(m).strip() if pd.notna(m) else ""
        s = str(s).strip() if pd.notna(s) else ""
        cols.append(f"{m} ({s})" if (m and s and s.lower() != "nan") else m)

    data_start = header_row + 2 if sub_row is not None else header_row + 1
    df = raw.iloc[data_start:].copy()
    df.columns = cols
    df = df.dropna(how="all")
    return df


# =========================
# Coin Change tối ưu số tờ (DP)
# =========================
@st.cache_data(show_spinner=False)
def build_prev_coin(denoms_k: List[int], max_amount_k: int) -> np.ndarray:
    denoms_k = sorted(set(int(d) for d in denoms_k if d > 0))
    INF = 10**9
    dp = np.full(max_amount_k + 1, INF, dtype=np.int32)
    prev = np.zeros(max_amount_k + 1, dtype=np.int32)
    dp[0] = 0

    # tie-break: ưu tiên mệnh giá lớn khi số tờ bằng nhau
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


# =========================
# Export Excel (2 sheet)
# =========================
def to_excel_bytes(df_people: pd.DataFrame, df_summary: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df_people.to_excel(writer, index=False, sheet_name="Danh_sach_can_lay")
        df_summary.to_excel(writer, index=False, sheet_name="Tong_hop_so_to")

        for sheet_name, df in [("Danh_sach_can_lay", df_people), ("Tong_hop_so_to", df_summary)]:
            ws = writer.sheets[sheet_name]
            for i, col in enumerate(df.columns):
                width = max(10, min(55, int(df[col].astype(str).str.len().max() if len(df) else len(col)) + 2))
                ws.set_column(i, i, width)

    return bio.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

uploaded = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
if not uploaded:
    st.info("Upload file Excel để bắt đầu.")
    st.stop()

xls = pd.ExcelFile(uploaded)
sheet = st.selectbox("Chọn sheet", options=xls.sheet_names, index=0)

with st.sidebar:
    st.header("Thiết lập")
    rounding_mode = st.selectbox(
        "Nếu tiền không chia hết 1.000:",
        ["Giữ nguyên (báo dư)", "Làm tròn xuống 1.000", "Làm tròn lên 1.000", "Làm tròn gần nhất 1.000"],
        index=0,
    )

    st.caption("Mệnh giá dùng để tính:")
    st.code(", ".join([f"{d//1000}k" for d in DENOMS_VND]))

# đọc data
try:
    df = read_payroll_template(uploaded, sheet)
except Exception as e:
    st.error(f"Lỗi đọc file: {e}")
    st.stop()

money_col = "Còn được nhận"
need = ["Stt", "Mã NV", "Họ và tên", money_col]
missing = [c for c in need if c not in df.columns]
if missing:
    st.error(f"Thiếu cột bắt buộc: {missing}")
    st.stop()

# bỏ dòng TỔNG: chỉ giữ STT dạng số
df["_stt_num"] = pd.to_numeric(df["Stt"], errors="coerce")
df = df[df["_stt_num"].notna()].copy()

# parse money
vals, notes = [], []
for v in df[money_col].tolist():
    val, note = parse_money_to_int_vnd(v)
    vals.append(val)
    notes.append(note)
df["_money_raw"] = vals
df["_parse_note"] = notes

# rounding
used, rem = [], []
for amt in df["_money_raw"].astype(int).tolist():
    u, r = round_to_1000(int(amt), rounding_mode)
    used.append(u)
    rem.append(r)
df["_money_used"] = used
df["_remainder"] = rem

# filter
st.subheader("1) Lọc danh sách")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    q = st.text_input("Tìm theo Tên hoặc Mã NV", "")
with c2:
    min_v = st.number_input("Tiền >= (VND)", min_value=0, value=0, step=1000)
with c3:
    max_v = st.number_input("Tiền <= (VND) (0 = không giới hạn)", min_value=0, value=0, step=1000)

mask = df["_money_raw"] >= int(min_v)
if int(max_v) > 0:
    mask &= df["_money_raw"] <= int(max_v)
if q.strip():
    q2 = q.strip().lower()
    mask &= (
        df["Họ và tên"].astype(str).str.lower().str.contains(q2, na=False) |
        df["Mã NV"].astype(str).str.lower().str.contains(q2, na=False)
    )

df_f = df.loc[mask].copy()
st.caption(f"Số lao động sau lọc: **{len(df_f)}** (đã tự bỏ dòng TỔNG).")
if df_f.empty:
    st.warning("Không có dữ liệu sau khi lọc.")
    st.stop()

# tính mệnh giá
denoms_k_desc = [d // 1000 for d in DENOMS_VND]        # desc
denoms_k_sorted = sorted(set(denoms_k_desc))          # asc for DP
max_k = int(df_f["_money_used"].max()) // 1000
prev = build_prev_coin(denoms_k_sorted, max_k)
denoms_k_set = set(denoms_k_sorted)

labels = [f"{d//1000}k" for d in DENOMS_VND]           # đúng thứ tự desc theo DENOMS_VND
rows_counts, totals_notes = [], []
for amt in df_f["_money_used"].astype(int).tolist():
    ak = amt // 1000
    counts = reconstruct_counts(ak, prev, denoms_k_set)
    row, n = [], 0
    for dk in denoms_k_desc:
        c = int(counts.get(dk, 0))
        row.append(c)
        n += c
    rows_counts.append(row)
    totals_notes.append(n)

df_counts = pd.DataFrame(rows_counts, columns=labels)
out = pd.concat(
    [df_f[["Stt", "Mã NV", "Họ và tên", money_col, "_money_raw", "_money_used", "_remainder", "_parse_note"]].reset_index(drop=True),
     df_counts],
    axis=1
)
out["Tổng_số_tờ"] = totals_notes

# tạo cột "Cần lấy" (1 dòng dễ nhìn)
def build_pick_text(row):
    parts = []
    for lab in labels:
        v = int(row[lab])
        if v > 0:
            parts.append(f"{lab}×{v}")
    return ", ".join(parts)

out["Can_lay"] = out.apply(build_pick_text, axis=1)

# =========================
# 2) Bảng gọn: chỉ hiển thị thông tin cần dùng
# =========================
st.subheader("2) Danh sách phát lương (gọn – nhìn là biết cần lấy)")
ui = out[["Stt", "Mã NV", "Họ và tên", "Còn được nhận", "Can_lay", "Tổng_số_tờ"]].copy()
ui = ui.rename(columns={
    "Stt": "STT",
    "Can_lay": "Cần lấy",
    "Tổng_số_tờ": "Tổng số tờ"
})

def bold_text(val):
    if pd.isna(val) or str(val).strip() == "":
        return ""
    return "font-weight: 800;"

styled = ui.style.applymap(bold_text, subset=["Cần lấy", "Tổng số tờ"])
st.dataframe(styled, use_container_width=True, height=560)

# =========================
# 3) Tổng hợp số tờ cần chuẩn bị
# =========================
st.subheader("3) Tổng hợp số tờ cần chuẩn bị (theo danh sách đang lọc)")
sum_notes = {lab: int(out[lab].sum()) for lab in labels}

summary_rows = []
for d, lab in zip(DENOMS_VND, labels):
    summary_rows.append({
        "Mệnh giá": lab,
        "Số tờ": sum_notes[lab],
        "Thành tiền (VND)": sum_notes[lab] * d
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.loc[len(df_summary)] = {
    "Mệnh giá": "TỔNG",
    "Số tờ": int(df_summary["Số tờ"].sum()),
    "Thành tiền (VND)": int(df_summary["Thành tiền (VND)"].sum())
}

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Tổng người", len(ui))
with c2:
    st.metric("Tổng số tờ", int(df_summary.loc[df_summary["Mệnh giá"] == "TỔNG", "Số tờ"].iloc[0]))
with c3:
    st.metric("Tổng tiền (VND)", int(df_summary.loc[df_summary["Mệnh giá"] == "TỔNG", "Thành tiền (VND)"].iloc[0]))

st.dataframe(df_summary, use_container_width=True, height=380)

# =========================
# 4) Download Excel (2 sheet)
# =========================
st.subheader("4) Tải file kết quả")
excel_bytes = to_excel_bytes(ui, df_summary)
st.download_button(
    "Tải Excel (Danh sách gọn + Tổng hợp)",
    data=excel_bytes,
    file_name="ket_qua_phat_luong_can_lay.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
