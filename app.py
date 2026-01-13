import re
import unicodedata
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "Chốt ứng lương + tách lương tay/chuyển khoản + tính mệnh giá (chuẩn 100%)"

# Mệnh giá (VND) - BỎ 30K
DENOMS_VND = [500_000, 200_000, 100_000, 50_000, 20_000, 10_000, 5_000, 2_000, 1_000]


# =========================
# Utils: normalize + parse tiền
# =========================
def normalize_text(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # remove accents
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s


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
# Read payroll template: tự tìm header theo "Còn được nhận"
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
        raise ValueError("Không tìm thấy dòng tiêu đề chứa 'Còn được nhận' trong sheet lương.")

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
# Export Excel: 2 sheet (CHUYEN_KHOAN, TRA_LUONG_TAY + tổng mệnh giá nằm trong sheet)
# =========================
def export_excel_2_sheets(df_ck: pd.DataFrame, df_cash: pd.DataFrame, df_cash_summary: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        # Sheet chuyển khoản
        df_ck.to_excel(writer, index=False, sheet_name="CHUYEN_KHOAN")

        # Sheet lương tay: danh sách + tổng hợp mệnh giá nằm phía dưới
        sheet_name = "TRA_LUONG_TAY"
        df_cash.to_excel(writer, index=False, sheet_name=sheet_name, startrow=0)

        start = len(df_cash) + 3
        df_cash_summary.to_excel(writer, index=False, sheet_name=sheet_name, startrow=start)

        # Format cơ bản
        for sn, df in [("CHUYEN_KHOAN", df_ck), (sheet_name, df_cash)]:
            ws = writer.sheets[sn]
            ws.freeze_panes(1, 0)
            for i, col in enumerate(df.columns):
                width = max(10, min(55, int(df[col].astype(str).str.len().max() if len(df) else len(col)) + 2))
                ws.set_column(i, i, width)

        # set width cho bảng tổng hợp trong sheet lương tay
        ws2 = writer.sheets[sheet_name]
        for i, col in enumerate(df_cash_summary.columns):
            width = max(10, min(45, int(df_cash_summary[col].astype(str).str.len().max() if len(df_cash_summary) else len(col)) + 2))
            ws2.set_column(i, i, width)

    return bio.getvalue()


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

uploaded = st.file_uploader("Upload 1 file Excel (.xlsx) chứa sheet HỢP CHÍ + sheet ung_luong", type=["xlsx"])
if not uploaded:
    st.info("Upload file để bắt đầu.")
    st.stop()

xls = pd.ExcelFile(uploaded)
default_payroll = "HỢP CHÍ" if "HỢP CHÍ" in xls.sheet_names else xls.sheet_names[0]
default_adv = "ung_luong" if "ung_luong" in xls.sheet_names else (xls.sheet_names[1] if len(xls.sheet_names) > 1 else xls.sheet_names[0])

c1, c2 = st.columns(2)
with c1:
    payroll_sheet = st.selectbox("Sheet bảng lương", options=xls.sheet_names, index=xls.sheet_names.index(default_payroll))
with c2:
    adv_sheet = st.selectbox("Sheet ứng lương", options=xls.sheet_names, index=xls.sheet_names.index(default_adv))

with st.sidebar:
    st.header("Thiết lập chuẩn (vì tiền)")
    rounding_mode = st.selectbox(
        "Nếu tiền không chia hết 1.000:",
        ["Giữ nguyên (báo dư)", "Làm tròn xuống 1.000", "Làm tròn lên 1.000", "Làm tròn gần nhất 1.000"],
        index=0,
    )
    st.caption("Mệnh giá tính tiền mặt:")
    st.code(", ".join([f"{d//1000}k" for d in DENOMS_VND]))


# =========================
# 1) Đọc bảng lương
# =========================
try:
    df_pay = read_payroll_template(uploaded, payroll_sheet)
except Exception as e:
    st.error(f"Lỗi đọc sheet bảng lương: {e}")
    st.stop()

# chuẩn hoá tên cột (tránh lệch do dấu cách)
col_map = {c: str(c).strip() for c in df_pay.columns}
df_pay = df_pay.rename(columns=col_map)

# Chuẩn hoá tên cột (bỏ khoảng trắng thừa)
df_pay.columns = [str(c).strip() for c in df_pay.columns]

def find_col(df: pd.DataFrame, keywords: list[str]) -> str:
    for c in df.columns:
        nc = normalize_text(c)
        if any(k in nc for k in keywords):
            return c
    return ""

# Bắt buộc phải có các cột chính
must_cols = {
    "Stt": find_col(df_pay, ["stt"]),
    "Mã NV": find_col(df_pay, ["ma nv", "manv"]),
    "Họ và tên": find_col(df_pay, ["ho va ten", "hoten", "ten"]),
    "Tổng lương": find_col(df_pay, ["tong luong"]),
    "Ứng lương": find_col(df_pay, ["ung luong"]),
    "Còn được nhận": find_col(df_pay, ["con duoc nhan"]),
}
missing = [k for k, v in must_cols.items() if not v]
if missing:
    st.error(f"Thiếu cột bắt buộc trong sheet bảng lương: {missing}")
    st.stop()

# Cột Lương tay (1 = trả tay)
cash_col = find_col(df_pay, ["luong tay", "luongtay"])
if not cash_col:
    st.error("Không tìm thấy cột 'Lương tay' trong sheet HỢP CHÍ. Bạn hãy thêm cột này (đúng hàng tiêu đề) và upload lại file.")
    st.stop()

# Rename về chuẩn để code phía dưới chạy đồng nhất
df_pay = df_pay.rename(columns={v: k for k, v in must_cols.items()})
df_pay = df_pay.rename(columns={cash_col: "Lương tay"})

# Debug cực nhanh để bạn nhìn là biết app đã nhận đúng chưa
st.caption(f"✅ Đã nhận cột lương tay: **{cash_col}**")

# Lương tay = 1
df_pay["_is_cash"] = df_pay["Lương tay"].apply(is_cash_flag)
st.caption(f"✅ Số người Lương tay = 1: **{int(df_pay['_is_cash'].sum())}**")


# =========================
# 4) Tách danh sách: CHUYỂN KHOẢN vs TRẢ TIỀN MẶT
# =========================
df_cash = df_pay[df_pay["_is_cash"]].copy()
df_ck = df_pay[~df_pay["_is_cash"]].copy()

# Sheet chuyển khoản (gọn đủ tiền)
df_ck_out = df_ck[["Stt", "Mã NV", "Họ và tên", "Tổng lương", "Ứng lương", "Còn được nhận (tính)"]].copy()
df_ck_out = df_ck_out.rename(columns={
    "Stt": "STT",
    "Còn được nhận (tính)": "Số tiền chuyển khoản"
})

# =========================
# 5) Tính mệnh giá cho danh sách lương tay
# =========================
labels = [f"{d//1000}k" for d in DENOMS_VND]
denoms_k_desc = [d // 1000 for d in DENOMS_VND]
denoms_k_sorted = sorted(set(denoms_k_desc))

if len(df_cash) > 0:
    max_k = int(df_cash["_money_used"].max()) // 1000
    prev = build_prev_coin(denoms_k_sorted, max_k)
    denoms_k_set = set(denoms_k_sorted)

    rows_counts, total_notes = [], []
    for amt in df_cash["_money_used"].astype(int).tolist():
        ak = amt // 1000
        counts = reconstruct_counts(ak, prev, denoms_k_set)
        row, n = [], 0
        for dk in denoms_k_desc:
            c = int(counts.get(dk, 0))
            row.append(c)
            n += c
        rows_counts.append(row)
        total_notes.append(n)

    df_counts = pd.DataFrame(rows_counts, columns=labels)
    df_cash2 = pd.concat(
        [df_cash[["Stt", "Mã NV", "Họ và tên", "Tổng lương", "Ứng lương", "Còn được nhận (tính)", "_remainder"]].reset_index(drop=True),
         df_counts],
        axis=1
    )
    df_cash2["Tổng số tờ"] = total_notes

    # tạo "Cần lấy"
    def build_pick_text(row):
        parts = []
        for lab in labels:
            v = int(row[lab])
            if v > 0:
                parts.append(f"{lab}×{v}")
        return ", ".join(parts)

    df_cash2["Cần lấy"] = df_cash2.apply(build_pick_text, axis=1)

    # UI/Excel: 0 -> trống cho dễ nhìn ở cột mệnh giá
    df_cash_out = df_cash2.copy()
    for c in labels:
        df_cash_out[c] = df_cash_out[c].replace(0, "")

    df_cash_out = df_cash_out.rename(columns={
        "Stt": "STT",
        "Còn được nhận (tính)": "Số tiền trả tay",
        "_remainder": "Phần lẻ (<1.000)"
    })

    # Tổng hợp số lượng từng mệnh giá
    sum_notes = {lab: int(df_cash2[lab].sum()) for lab in labels}
    summary_rows = []
    for d, lab in zip(DENOMS_VND, labels):
        summary_rows.append({"Mệnh giá": lab, "Số tờ": sum_notes[lab], "Thành tiền (VND)": sum_notes[lab] * d})
    df_cash_summary = pd.DataFrame(summary_rows)
    df_cash_summary.loc[len(df_cash_summary)] = {
        "Mệnh giá": "TỔNG",
        "Số tờ": int(df_cash_summary["Số tờ"].sum()),
        "Thành tiền (VND)": int(df_cash_summary["Thành tiền (VND)"].sum())
    }
else:
    df_cash_out = pd.DataFrame(columns=["STT", "Mã NV", "Họ và tên", "Số tiền trả tay", "Cần lấy"])
    df_cash_summary = pd.DataFrame(columns=["Mệnh giá", "Số tờ", "Thành tiền (VND)"])

# =========================
# 6) Hiển thị kết quả + Download
# =========================
st.subheader("Kết quả 1) Danh sách CHUYỂN KHOẢN")
st.dataframe(df_ck_out, use_container_width=True, height=360)

st.subheader("Kết quả 2) Danh sách TRẢ LƯƠNG TAY + chi tiết mệnh giá")
st.dataframe(df_cash_out, use_container_width=True, height=460)

st.subheader("Tổng số lượng từng mệnh giá (chỉ tính danh sách LƯƠNG TAY)")
st.dataframe(df_cash_summary, use_container_width=True, height=320)

st.subheader("Tải file kết quả (2 sheet)")
excel_bytes = export_excel_2_sheets(df_ck_out, df_cash_out, df_cash_summary)
st.download_button(
    "Tải Excel: CHUYEN_KHOAN + TRA_LUONG_TAY",
    data=excel_bytes,
    file_name="ket_qua_chot_luong_ung_luong_tach_luong_tay.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
