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

need_pay = ["Stt", "Mã NV", "Họ và tên", "Tổng lương", "Ứng lương", "Còn được nhận", "Lương Tay"]
missing = [c for c in need_pay if c not in df_pay.columns]
if missing:
    st.error(f"Thiếu cột bắt buộc trong sheet bảng lương: {missing}\n"
             f"Bạn cần có tối thiểu các cột: {need_pay}")
    st.stop()

# bỏ dòng tổng: STT phải là số
df_pay["_stt_num"] = pd.to_numeric(df_pay["Stt"], errors="coerce")
df_pay = df_pay[df_pay["_stt_num"].notna()].copy()

# =========================
# 2) Đọc sheet ứng lương
# =========================
try:
    df_adv_raw = pd.read_excel(uploaded, sheet_name=adv_sheet, dtype=object)
except Exception as e:
    st.error(f"Lỗi đọc sheet ứng lương: {e}")
    st.stop()

# auto-detect cột tên + tổng ứng (nhưng vẫn cho chọn nếu muốn chắc)
def find_col_by_keywords(df: pd.DataFrame, keywords: List[str]) -> str:
    cols = list(df.columns)
    norm_cols = {c: normalize_text(c) for c in cols}
    for c in cols:
        n = norm_cols[c]
        if any(k in n for k in keywords):
            return c
    return ""

name_guess = find_col_by_keywords(df_adv_raw, ["ho va ten", "hoten", "ten", "nhan vien"])
amt_guess = find_col_by_keywords(df_adv_raw, ["tong ung", "ung luong", "so tien ung", "ung"])

st.subheader("0) Chọn đúng cột trong sheet ung_luong (để khớp 100%)")
cc1, cc2 = st.columns(2)
with cc1:
    adv_name_col = st.selectbox("Cột TÊN trong ung_luong", options=list(df_adv_raw.columns), index=(list(df_adv_raw.columns).index(name_guess) if name_guess in df_adv_raw.columns else 0))
with cc2:
    adv_amt_col = st.selectbox("Cột TỔNG ỨNG trong ung_luong", options=list(df_adv_raw.columns), index=(list(df_adv_raw.columns).index(amt_guess) if amt_guess in df_adv_raw.columns else 0))

df_adv = df_adv_raw[[adv_name_col, adv_amt_col]].copy()
df_adv = df_adv.rename(columns={adv_name_col: "Họ và tên", adv_amt_col: "Tổng ứng"})

# parse tiền ứng
adv_amounts, adv_notes = [], []
for v in df_adv["Tổng ứng"].tolist():
    val, note = parse_money_to_int_vnd(v)
    adv_amounts.append(val)
    adv_notes.append(note)
df_adv["_adv_vnd"] = adv_amounts
df_adv["_adv_note"] = adv_notes

# =========================
# 3) ĐỐI CHIẾU KHỚP 100% (theo Họ và tên đã normalize)
# =========================
df_pay["_key"] = df_pay["Họ và tên"].apply(normalize_text)
df_adv["_key"] = df_adv["Họ và tên"].apply(normalize_text)

# check duplicate keys
dup_pay = df_pay[df_pay["_key"].duplicated(keep=False)]["_key"].unique().tolist()
dup_adv = df_adv[df_adv["_key"].duplicated(keep=False)]["_key"].unique().tolist()
if dup_pay or dup_adv:
    st.error("❌ LỖI TRÙNG NHÂN VIÊN (không thể khớp 100% vì có trùng khóa tên).")
    if dup_pay:
        st.write("Trùng trong HỢP CHÍ:", dup_pay)
    if dup_adv:
        st.write("Trùng trong ung_luong:", dup_adv)
    st.stop()

pay_keys = set(df_pay["_key"].tolist())
adv_keys = set(df_adv["_key"].tolist())
not_found = sorted(list(adv_keys - pay_keys))

if not_found:
    st.error("❌ LỖI KHÔNG KHỚP 100%: Có nhân viên trong ung_luong KHÔNG TỒN TẠI trong HỢP CHÍ. DỪNG XỬ LÝ.")
    show = df_adv[df_adv["_key"].isin(not_found)][["Họ và tên", "_adv_vnd", "_adv_note"]].copy()
    st.dataframe(show, use_container_width=True, height=300)
    st.stop()

# map tiền ứng vào bảng lương
adv_map = dict(zip(df_adv["_key"], df_adv["_adv_vnd"]))
df_pay["_ung_goc_parse"], _ = zip(*[parse_money_to_int_vnd(x) for x in df_pay["Ứng lương"].tolist()])
df_pay["Ứng lương"] = df_pay["_key"].map(adv_map).fillna(0).astype(int)

# parse Tổng lương
pay_total_vals, pay_total_notes = [], []
for v in df_pay["Tổng lương"].tolist():
    val, note = parse_money_to_int_vnd(v)
    pay_total_vals.append(val)
    pay_total_notes.append(note)
df_pay["_tong_luong_vnd"] = pay_total_vals
df_pay["_tong_note"] = pay_total_notes

# tính Còn được nhận = Tổng lương - Ứng lương (CHUẨN vì pandas không chạy công thức excel)
df_pay["Còn được nhận (tính)"] = (df_pay["_tong_luong_vnd"].astype(int) - df_pay["Ứng lương"].astype(int)).astype(int)

# validate không âm
neg = df_pay[df_pay["Còn được nhận (tính)"] < 0][["Mã NV", "Họ và tên", "_tong_luong_vnd", "Ứng lương", "Còn được nhận (tính)"]]
if len(neg) > 0:
    st.error("❌ LỖI: Có người có Còn được nhận (tính) < 0 (Tổng lương - Ứng lương âm). DỪNG XỬ LÝ.")
    st.dataframe(neg, use_container_width=True, height=260)
    st.stop()

# rounding
used_list, rem_list = [], []
for amt in df_pay["Còn được nhận (tính)"].astype(int).tolist():
    u, r = round_to_1000(int(amt), rounding_mode)
    used_list.append(u)
    rem_list.append(r)
df_pay["_money_used"] = used_list
df_pay["_remainder"] = rem_list

# flag lương tay
def is_cash_flag(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    try:
        return int(float(s)) == 1
    except Exception:
        return s == "1"

df_pay["_is_cash"] = df_pay["Lương Tay"].apply(is_cash_flag)

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
