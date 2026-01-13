import re
import unicodedata
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_TITLE = "CHỐT ỨNG LƯƠNG + TÁCH LƯƠNG TAY/CK + TÍNH MỆNH GIÁ (CHUẨN 100%)"

# Mệnh giá (VND) - BỎ 30K
DENOMS_VND = [500_000, 200_000, 100_000, 50_000, 20_000, 10_000, 5_000, 2_000, 1_000]


# =========================
# Text normalize (quan trọng: đổi 'đ' -> 'd' để match chuẩn)
# =========================
def normalize_text(x) -> str:
    s = "" if pd.isna(x) else str(x)
    s = s.strip().lower()
    s = s.replace("đ", "d").replace("Đ", "d")
    s = re.sub(r"\s+", " ", s)
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s


def find_col(df: pd.DataFrame, keywords: List[str]) -> str:
    for c in df.columns:
        nc = normalize_text(c)
        if any(k in nc for k in keywords):
            return c
    return ""


# =========================
# Parse tiền an toàn
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
        return int(round(x)), "" if abs(x - round(x)) < 1e-6 else "float->rounded"

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


def is_cash_flag(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    if s == "":
        return False
    try:
        return int(float(s)) == 1
    except Exception:
        return s == "1"


def round_to_1000(amount_vnd: int, mode: str) -> Tuple[int, int]:
    rem = amount_vnd % 1000
    if rem == 0:
        return amount_vnd, 0

    if mode in ("Giữ nguyên (báo dư)", "Làm tròn xuống 1.000"):
        return amount_vnd - rem, rem
    if mode == "Làm tròn lên 1.000":
        return amount_vnd + (1000 - rem), rem

    # gần nhất
    if rem >= 500:
        return amount_vnd + (1000 - rem), rem
    return amount_vnd - rem, rem


# =========================
# Đọc sheet lương kiểu template: tự tìm header theo "Còn được nhận"
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
        raise ValueError("Không tìm thấy dòng tiêu đề chứa 'Còn được nhận' trong sheet bảng lương.")

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
    df.columns = [str(c).strip() for c in cols]
    df = df.dropna(how="all")
    return df


# =========================
# Coin change tối ưu số tờ (DP)
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
# Export Excel: 2 sheet
# =========================
def export_excel_2_sheets(df_ck: pd.DataFrame, df_cash: pd.DataFrame, df_cash_summary: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df_ck.to_excel(writer, index=False, sheet_name="CHUYEN_KHOAN")

        sheet_cash = "TRA_LUONG_TAY"
        df_cash.to_excel(writer, index=False, sheet_name=sheet_cash, startrow=0)

        start = len(df_cash) + 3
        df_cash_summary.to_excel(writer, index=False, sheet_name=sheet_cash, startrow=start)

        wb = writer.book
        fmt_money = wb.add_format({"num_format": "#,##0"})
        fmt_int = wb.add_format({"num_format": "0"})
        fmt_header = wb.add_format({"bold": True})

        # format CK
        ws_ck = writer.sheets["CHUYEN_KHOAN"]
        ws_ck.freeze_panes(1, 0)
        for i, col in enumerate(df_ck.columns):
            width = max(10, min(55, int(df_ck[col].astype(str).str.len().max() if len(df_ck) else len(col)) + 2))
            ws_ck.set_column(i, i, width)
        for col_name in ["Tổng lương", "Ứng lương", "Số tiền chuyển khoản"]:
            if col_name in df_ck.columns:
                j = df_ck.columns.get_loc(col_name)
                ws_ck.set_column(j, j, 18, fmt_money)

        # format CASH
        ws_cash = writer.sheets[sheet_cash]
        ws_cash.freeze_panes(1, 0)
        for i, col in enumerate(df_cash.columns):
            width = max(10, min(55, int(df_cash[col].astype(str).str.len().max() if len(df_cash) else len(col)) + 2))
            ws_cash.set_column(i, i, width)

        for col_name in ["Tổng lương", "Ứng lương", "Số tiền trả tay"]:
            if col_name in df_cash.columns:
                j = df_cash.columns.get_loc(col_name)
                ws_cash.set_column(j, j, 18, fmt_money)

        # denom cols int
        for d in DENOMS_VND:
            lab = f"{d//1000}k"
            if lab in df_cash.columns:
                j = df_cash.columns.get_loc(lab)
                ws_cash.set_column(j, j, 9, fmt_int)

        # summary format
        for i, col in enumerate(df_cash_summary.columns):
            ws_cash.write(start, i, col, fmt_header)
            width = max(10, min(45, int(df_cash_summary[col].astype(str).str.len().max() if len(df_cash_summary) else len(col)) + 2))
            ws_cash.set_column(i, i, width)

        if "Thành tiền (VND)" in df_cash_summary.columns:
            j = df_cash_summary.columns.get_loc("Thành tiền (VND)")
            ws_cash.set_column(j, j, 18, fmt_money)

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
if len(xls.sheet_names) == 0:
    st.error("File không có sheet nào.")
    st.stop()

default_pay = "HỢP CHÍ" if "HỢP CHÍ" in xls.sheet_names else xls.sheet_names[0]
default_adv = "ung_luong" if "ung_luong" in xls.sheet_names else (xls.sheet_names[1] if len(xls.sheet_names) > 1 else xls.sheet_names[0])

c1, c2 = st.columns(2)
with c1:
    payroll_sheet = st.selectbox("Sheet bảng lương", options=xls.sheet_names, index=xls.sheet_names.index(default_pay))
with c2:
    adv_sheet = st.selectbox("Sheet ứng lương", options=xls.sheet_names, index=xls.sheet_names.index(default_adv))

with st.sidebar:
    st.header("Thiết lập")
    rounding_mode = st.selectbox(
        "Nếu tiền trả tay không chia hết 1.000:",
        ["Giữ nguyên (báo dư)", "Làm tròn xuống 1.000", "Làm tròn lên 1.000", "Làm tròn gần nhất 1.000"],
        index=0,
    )
    st.caption("Mệnh giá dùng để chia tiền mặt:")
    st.code(", ".join([f"{d//1000}k" for d in DENOMS_VND]))


# =========================
# 1) Đọc bảng lương
# =========================
try:
    df_pay = read_payroll_template(uploaded, payroll_sheet)
except Exception as e:
    st.error(f"Lỗi đọc sheet bảng lương: {e}")
    st.stop()

df_pay.columns = [str(c).strip() for c in df_pay.columns]

col_stt = find_col(df_pay, ["stt"])
col_code = find_col(df_pay, ["ma nv", "manv"])
col_name = find_col(df_pay, ["ho va ten", "hoten", "ten"])
col_total = find_col(df_pay, ["tong luong"])
col_adv_in_pay = find_col(df_pay, ["ung luong"])
col_cash = find_col(df_pay, ["luong tay", "luongtay"])
col_con = find_col(df_pay, ["con duoc nhan"])  # có thể dùng để đối chiếu

need_missing = []
for k, v in [
    ("Stt", col_stt),
    ("Mã NV", col_code),
    ("Họ và tên", col_name),
    ("Tổng lương", col_total),
    ("Ứng lương", col_adv_in_pay),
    ("Lương tay", col_cash),
]:
    if not v:
        need_missing.append(k)

if need_missing:
    st.error(f"Thiếu cột bắt buộc trong sheet bảng lương: {need_missing}")
    st.stop()

df_pay = df_pay.rename(columns={
    col_stt: "Stt",
    col_code: "Mã NV",
    col_name: "Họ và tên",
    col_total: "Tổng lương",
    col_adv_in_pay: "Ứng lương (gốc)",
    col_cash: "Lương tay",
})
if col_con:
    df_pay = df_pay.rename(columns={col_con: "Còn được nhận (gốc)"})

# Bỏ dòng TỔNG: chỉ lấy STT numeric
df_pay["_stt_num"] = pd.to_numeric(df_pay["Stt"], errors="coerce")
df_pay = df_pay[df_pay["_stt_num"].notna()].copy()

# =========================
# 2) Đọc sheet ứng lương + chọn đúng cột "Tên" & "tổng ứng"
# =========================
try:
    df_adv_raw = pd.read_excel(uploaded, sheet_name=adv_sheet, dtype=object)
except Exception as e:
    st.error(f"Lỗi đọc sheet ứng lương: {e}")
    st.stop()

if df_adv_raw.empty:
    st.error("Sheet ứng lương đang trống.")
    st.stop()

adv_cols = list(df_adv_raw.columns)

# auto detect
auto_name = ""
auto_amt = ""
for c in adv_cols:
    nc = normalize_text(c)
    if not auto_name and ("ten" == nc or "ho va ten" in nc or nc.endswith("ten")):
        auto_name = c
    if not auto_amt and ("tong ung" in nc or ("tong" in nc and "ung" in nc)):
        auto_amt = c

with st.sidebar:
    st.header("Chọn cột ứng lương (đúng chuẩn)")
    adv_name_col = st.selectbox("Cột TÊN trong sheet ung_luong", options=adv_cols, index=adv_cols.index(auto_name) if auto_name in adv_cols else 0)
    adv_amt_col = st.selectbox("Cột TỔNG ỨNG trong sheet ung_luong", options=adv_cols, index=adv_cols.index(auto_amt) if auto_amt in adv_cols else 0)

df_adv = df_adv_raw[[adv_name_col, adv_amt_col]].copy()
df_adv.columns = ["Tên", "Tổng ứng"]

# drop rows rỗng tên
df_adv["Tên"] = df_adv["Tên"].astype(str).str.strip()
df_adv = df_adv[df_adv["Tên"].astype(str).str.strip() != ""].copy()

# parse tổng ứng
adv_vals = []
adv_notes = []
for v in df_adv["Tổng ứng"].tolist():
    val, note = parse_money_to_int_vnd(v)
    adv_vals.append(val)
    adv_notes.append(note)
df_adv["_adv_vnd"] = adv_vals
df_adv["_adv_note"] = adv_notes

# =========================
# 3) MATCH 100%: ung_luong phải nằm trong danh sách HỢP CHÍ
# =========================
df_pay["_key_name"] = df_pay["Họ và tên"].apply(normalize_text)
df_adv["_key_name"] = df_adv["Tên"].apply(normalize_text)

# check trùng tên trong ung_luong
dup_adv = df_adv["_key_name"][df_adv["_key_name"].duplicated()].unique().tolist()
if dup_adv:
    st.error("❌ LỖI: Sheet ung_luong có TÊN bị trùng (không đảm prove 100%). Hãy sửa trùng tên trước.")
    st.write(pd.DataFrame({"Tên_trùng(đã chuẩn hoá)": dup_adv}))
    st.stop()

# check trùng tên trong bảng lương
dup_pay = df_pay["_key_name"][df_pay["_key_name"].duplicated()].unique().tolist()
if dup_pay:
    st.error("❌ LỖI: Sheet bảng lương có NHÂN VIÊN trùng tên. Muốn 100% thì hãy match bằng MÃ NV trong ung_luong.")
    st.write(pd.DataFrame({"Tên_trùng(đã chuẩn hoá)": dup_pay}))
    st.stop()

pay_key_set = set(df_pay["_key_name"].tolist())
adv_key_set = set(df_adv["_key_name"].tolist())
not_found = sorted(list(adv_key_set - pay_key_set))
if not_found:
    st.error("❌ LỖI: Có nhân viên trong sheet ung_luong KHÔNG khớp 100% với sheet bảng lương. Dừng xử lý để tránh sai tiền.")
    st.write(pd.DataFrame({"Tên_không_khớp(đã chuẩn hoá)": not_found}))
    st.stop()

# map ứng lương
adv_map = dict(zip(df_adv["_key_name"].tolist(), df_adv["_adv_vnd"].astype(int).tolist()))
df_pay["Ứng lương"] = df_pay["_key_name"].map(adv_map).fillna(0).astype(int)

# parse tổng lương
pay_total_vals = []
pay_total_notes = []
for v in df_pay["Tổng lương"].tolist():
    val, note = parse_money_to_int_vnd(v)
    pay_total_vals.append(val)
    pay_total_notes.append(note)
df_pay["_tongluong_vnd"] = np.array(pay_total_vals, dtype=np.int64)
df_pay["_tongluong_note"] = pay_total_notes

# tính còn được nhận (tính)
df_pay["Còn được nhận (tính)"] = (df_pay["_tongluong_vnd"].astype(np.int64) - df_pay["Ứng lương"].astype(np.int64)).astype(np.int64)

# cash flag
df_pay["_is_cash"] = df_pay["Lương tay"].apply(is_cash_flag)

# =========================
# 4) Tách danh sách CK vs TRẢ TAY
# =========================
df_cash = df_pay[df_pay["_is_cash"]].copy()
df_ck = df_pay[~df_pay["_is_cash"]].copy()

st.success("✅ Match ung_luong -> bảng lương: KHỚP 100% (mới cho chạy tiếp).")
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Số người TRẢ TAY (Lương tay=1)", int(df_cash.shape[0]))
with m2:
    st.metric("Số người CHUYỂN KHOẢN", int(df_ck.shape[0]))
with m3:
    st.metric("Tổng người", int(df_pay.shape[0]))

# =========================
# 5) Danh sách chuyển khoản (gọn đủ)
# =========================
df_ck_out = df_ck[["Stt", "Mã NV", "Họ và tên", "Tổng lương", "Ứng lương", "Còn được nhận (tính)"]].copy()
df_ck_out = df_ck_out.rename(columns={
    "Stt": "STT",
    "Còn được nhận (tính)": "Số tiền chuyển khoản",
})

# =========================
# 6) Tính mệnh giá cho danh sách trả tay
# =========================
labels = [f"{d//1000}k" for d in DENOMS_VND]
denoms_k_desc = [d // 1000 for d in DENOMS_VND]
denoms_k_sorted = sorted(set(denoms_k_desc))

if df_cash.empty:
    df_cash_out = pd.DataFrame(columns=["STT", "Mã NV", "Họ và tên", "Tổng lương", "Ứng lương", "Số tiền trả tay", "Cần lấy"])
    df_cash_summary = pd.DataFrame(columns=["Mệnh giá", "Số tờ", "Thành tiền (VND)"])
else:
    # tiền đem chia mệnh giá: làm tròn theo 1.000
    used_list = []
    rem_list = []
    for amt in df_cash["Còn được nhận (tính)"].astype(np.int64).tolist():
        used, rem = round_to_1000(int(amt), rounding_mode)
        used_list.append(int(used))
        rem_list.append(int(rem))
    df_cash["_money_used"] = used_list
    df_cash["_remainder"] = rem_list

    max_k = int(df_cash["_money_used"].max()) // 1000
    prev = build_prev_coin(denoms_k_sorted, max_k)
    denoms_k_set = set(denoms_k_sorted)

    rows_counts, total_notes = [], []
    for amt in df_cash["_money_used"].astype(int).tolist():
        ak = amt // 1000
        counts = reconstruct_counts(ak, prev, denoms_k_set)
        row = []
        n = 0
        for dk in denoms_k_desc:
            c = int(counts.get(dk, 0))
            row.append(c)
            n += c
        rows_counts.append(row)
        total_notes.append(n)

    df_counts = pd.DataFrame(rows_counts, columns=labels)

    df_cash2 = pd.concat(
        [
            df_cash[["Stt", "Mã NV", "Họ và tên", "Tổng lương", "Ứng lương"]].reset_index(drop=True),
            pd.Series(df_cash["_money_used"].astype(int).tolist(), name="Số tiền trả tay"),
            pd.Series(df_cash["_remainder"].astype(int).tolist(), name="Phần lẻ (<1.000)"),
            df_counts.reset_index(drop=True),
        ],
        axis=1,
    )
    df_cash2["Tổng số tờ"] = total_notes

    def build_pick_text(row):
        parts = []
        for lab in labels:
            v = int(row[lab])
            if v > 0:
                parts.append(f"{lab}×{v}")
        return ", ".join(parts)

    df_cash2["Cần lấy"] = df_cash2.apply(build_pick_text, axis=1)

    # 0 -> trống ở cột mệnh giá cho dễ nhìn
    df_cash_out = df_cash2.copy()
    for c in labels:
        df_cash_out[c] = df_cash_out[c].replace(0, "")

    df_cash_out = df_cash_out.rename(columns={"Stt": "STT"})

    # Tổng hợp số lượng từng mệnh giá (tính theo df_cash2 vì còn số)
    sum_notes = {lab: int(df_cash2[lab].sum()) for lab in labels}
    summary_rows = []
    for d, lab in zip(DENOMS_VND, labels):
        summary_rows.append({"Mệnh giá": lab, "Số tờ": sum_notes[lab], "Thành tiền (VND)": sum_notes[lab] * d})
    df_cash_summary = pd.DataFrame(summary_rows)
    df_cash_summary.loc[len(df_cash_summary)] = {
        "Mệnh giá": "TỔNG",
        "Số tờ": int(df_cash_summary["Số tờ"].sum()),
        "Thành tiền (VND)": int(df_cash_summary["Thành tiền (VND)"].sum()),
    }

# =========================
# 7) Hiển thị + Download
# =========================
st.subheader("1) DANH SÁCH CHUYỂN KHOẢN")
st.dataframe(df_ck_out, use_container_width=True, height=420)

st.subheader("2) DANH SÁCH TRẢ LƯƠNG TAY (kèm mệnh giá + Cần lấy)")
st.caption("Cột mệnh giá: số 0 được để trống cho dễ nhìn. 'Phần lẻ' là số dư < 1.000 (nếu có).")
st.dataframe(df_cash_out, use_container_width=True, height=520)

st.subheader("3) TỔNG SỐ LƯỢNG TỪNG MỆNH GIÁ (chỉ tính danh sách TRẢ TAY)")
st.dataframe(df_cash_summary, use_container_width=True, height=320)

st.subheader("4) TẢI FILE KẾT QUẢ (2 SHEET)")
excel_bytes = export_excel_2_sheets(df_ck_out, df_cash_out, df_cash_summary)
st.download_button(
    "Tải Excel: CHUYEN_KHOAN + TRA_LUONG_TAY",
    data=excel_bytes,
    file_name="KQ_CHOT_UNG_LUONG_TACH_TRA_TAY_CK.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
