import pandas as pd
import numpy as np
import re

# ====================================================
# 共通ヘルパー関数
# ====================================================
def parse_and_clean_code(raw_str):
    if pd.isna(raw_str) or not str(raw_str).strip():
        return "", ""
    s = str(raw_str).strip()
    s = re.split(r'[\s ]', s)[0]
    parts = [p.strip() for p in s.split('/') if p.strip()]
    k1, k2 = "", ""
    if len(parts) >= 1: k1 = parts[0]
    if len(parts) >= 2: k2 = parts[1]
    return k1, k2


def normalize_code(code_str):
    if pd.isna(code_str):
        return ""
    s = str(code_str).upper().strip()
    trans = str.maketrans({
        'O': '0', 'Q': '0', 'C': '0',
        'I': '1', 'J': '1', 'L': '1',
        'B': '8', '日': '8',
        'S': '5',
        'Z': '2',
        'ー': '', '-': '', ' ': '', '_': '', '/': ''
    })
    return s.translate(trans)


def is_code_match(norm_a, norm_b):
    if not norm_a or not norm_b:
        return False
    
    # 双方とも完全一致する場合は無条件でTrue
    if norm_a == norm_b:
        return True
    
    # 部分一致（in 判定）を行う場合は「3文字以上」の型番のみ許可する
    # ※1〜2文字（S, L, 1, 5など）の短コードによる誤マッチングを防ぐ
    MIN_LENGTH = 3
    
    if len(norm_a) >= MIN_LENGTH and len(norm_b) >= MIN_LENGTH:
        return (norm_a in norm_b) or (norm_b in norm_a)
        
    return False


def normalize_dept_name(dept_str):
    if pd.isna(dept_str):
        return ""
    s = str(dept_str).strip()
    s = re.sub(r'\s+', '', s)
    
    if 'A4' in s:
        if 'ICU' in s or 'icu' in s:
            return 'A4病棟(ICU)'
        return 'A4病棟'

    if 'A7' in s:
        if 'ベビー' in s:
            return 'A7病棟(ベビー)'
        return 'A7病棟'

    if '放射線' in s:
        if '血管' in s:
            return '放射線科血管'
        return '放射線科'

    if '滅菌' in s: return '中央滅菌室'
    if '健診' in s or 'ドック' in s: return '健診センター'
    if '内視鏡' in s: return '内視鏡室'
    if '初療' in s: return '救命初療室'
    if '調度' in s or '倉庫' in s: return '調度課倉庫'
    
    s_no_paren = re.sub(r'[\(（].*?[\)）]', '', s)
    match = re.match(r'^([A-Z][0-9](?:北|南)?)', s_no_paren)
    if match:
        core = match.group(1)
        return f"{core}病棟"
        
    return s_no_paren


# ====================================================
# 【前処理】データクレンジング・正規化後SUM集約
# ====================================================
def preprocess_data(meisai_path, nouhin_path):
    print("=== 【前処理】データクレンジング・正規化後集約（SUM）の開始 ===")
    
    meisai = pd.read_csv(meisai_path)
    nouhin = pd.read_csv(nouhin_path)

    def clean_and_aggregate(df, is_nouhin=False):
        df = df.copy()
        
        # 1. 列ズレ補正
        if '数量' in df.columns:
            for idx, row in df.iterrows():
                if str(row['数量']).strip() in ['個', '本', '箱', '大箱', 'セット', '袋', 'ケース', 'パック']:
                    df.loc[idx, '単位'] = row['数量']
                    df.loc[idx, '数量'] = row['規格コード型番2']
                    df.loc[idx, '規格コード型番2'] = np.nan

        for col in ['部署名', '規格コード型番1', '規格コード型番2']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().replace({'nan': np.nan, 'None': np.nan, '': np.nan})
                
        # 2. 部署名の名寄せ統一
        df['部署名_統一'] = df['部署名'].apply(normalize_dept_name)
        
        # 3. 型番のパース（分解）
        for idx, row in df.iterrows():
            raw1 = str(row['規格コード型番1']) if pd.notna(row['規格コード型番1']) else ""
            raw2 = str(row['規格コード型番2']) if pd.notna(row['規格コード型番2']) else ""
            
            p1_k1, p1_k2 = parse_and_clean_code(raw1)
            p2_k1, _ = parse_and_clean_code(raw2)
            
            final_k1 = p1_k1
            final_k2 = raw2 if raw2 else (p1_k2 if p1_k2 else p2_k1)
            
            df.loc[idx, '規格コード型番1'] = final_k1 if final_k1 else np.nan
            df.loc[idx, '規格コード型番2'] = final_k2 if final_k2 else np.nan

        # 4. 型番の正規化（記号全消去・OCR補正文字列を作成）
        df['norm1'] = df['規格コード型番1'].apply(normalize_code)
        df['norm2'] = df['規格コード型番2'].apply(normalize_code)
        
        cond_dept = df['部署名_統一'].isna() | (df['部署名_統一'] == '')
        cond_qty = df['数量'].isna()
        cond_code = (df['norm1'] == '') & (df['norm2'] == '')
        invalid_mask = cond_dept | cond_qty | cond_code
        
        valid_df = df[~invalid_mask].copy()
        invalid_df = df[invalid_mask].copy()
        
        valid_df['数量'] = pd.to_numeric(valid_df['数量'], errors='coerce')
        
        # ★【ご指摘通りの重要修正】正規化（norm1, norm2）が終わった「あと」で集約（SUM）を実行！
        # これにより表記揺れ（FVM-50JL35 と FVM50JL35）も同じ型番として正しく足し合わされます
        group_cols = ['部署名_統一', 'norm1', 'norm2']
        
        # 表示用テキストの保持用
        agg_rules = {
            '数量': 'sum',
            '規格コード型番1': 'first',
            '規格コード型番2': 'first',
            '単位': 'first'
        }
        if is_nouhin and '伝票No' in valid_df.columns:
            agg_rules['伝票No'] = lambda x: ', '.join(sorted(list(set(str(v) for v in x if pd.notna(v)))))
            
        aggregated_df = valid_df.groupby(group_cols, as_index=False, dropna=False).agg(agg_rules)
        
        return aggregated_df, invalid_df

    m_valid, m_invalid = clean_and_aggregate(meisai, is_nouhin=False)
    n_valid, n_invalid = clean_and_aggregate(nouhin, is_nouhin=True)

    print(f" 明細書 : 正規化・事前集約後 {len(m_valid)} 件 / 不備データ {len(m_invalid)} 件")
    print(f" 納品書 : 正規化・事前集約後 {len(n_valid)} 件 / 不備データ {len(n_invalid)} 件")
    return m_valid, n_valid, m_invalid, n_invalid


# ====================================================
# 【メインエンジン】事前集約済みデータの照合
# ====================================================
def run_loop_matching(m_valid, n_valid):
    print("\n=== 【照合処理】突合エンジンの実行 ===")
    
    m_pool = m_valid.copy()
    n_pool = n_valid.copy()
    
    all_matched_records = []
    unm_m_idx = []
    
    for m_idx, m_row in m_pool.iterrows():
        m_dept = m_row['部署名_統一']
        m_qty = m_row['数量']
        m_n1, m_n2 = m_row['norm1'], m_row['norm2']
        
        cands = n_pool[(n_pool['部署名_統一'] == m_dept) & (n_pool['数量'] == m_qty)]
        matched_n_idx = None
        match_pattern = None
        
        for n_idx, n_row in cands.iterrows():
            n_n1, n_n2 = n_row['norm1'], n_row['norm2']
            
            if is_code_match(m_n1, n_n1):
                matched_n_idx, match_pattern = n_idx, '集約後一致(型番1≒納品1)'
                break
            elif is_code_match(m_n1, n_n2):
                matched_n_idx, match_pattern = n_idx, '集約後一致(型番1≒納品2)'
                break
            elif is_code_match(m_n2, n_n1):
                matched_n_idx, match_pattern = n_idx, '集約後一致(型番2≒納品1)'
                break
            elif is_code_match(m_n2, n_n2):
                matched_n_idx, match_pattern = n_idx, '集約後一致(型番2≒納品2)'
                break
                
        if matched_n_idx is not None:
            n_row = n_pool.loc[matched_n_idx]
            all_matched_records.append({
                '部署名': m_dept, '数量': m_qty,
                '明細_型番1': m_row['規格コード型番1'], '明細_型番2': m_row['規格コード型番2'],
                '納品_型番1': n_row['規格コード型番1'], '納品_型番2': n_row['規格コード型番2'],
                '一致パターン': match_pattern, 'ステータス': '一致',
                '伝票No': n_row.get('伝票No', '')
            })
            n_pool = n_pool.drop(matched_n_idx)
        else:
            unm_m_idx.append(m_idx)
            
    m_final_unm = m_pool.loc[unm_m_idx].drop(columns=['norm1', 'norm2'])
    n_final_unm = n_pool.drop(columns=['norm1', 'norm2'])
    matched_df = pd.DataFrame(all_matched_records)
    
    print(f" ［結果］ 一致: {len(matched_df)} 件 / 未一致 明細: {len(m_final_unm)} 件, 納品: {len(n_final_unm)} 件")
    return matched_df, m_final_unm, n_final_unm


# ====================================================
# 【レポート出力】Excel書き出し
# ====================================================
def export_excel_report(final_unm_meisai, final_unm_nouhin, meisai_invalid, nouhin_invalid, all_matched_df, output_path='突合結果_最終レポート.xlsx'):
    print("\n=== 【レポート出力】Excelファイルの書き出し ===")
    
    full_unm_meisai = pd.concat([final_unm_meisai, meisai_invalid], ignore_index=True)
    full_unm_nouhin = pd.concat([final_unm_nouhin, nouhin_invalid], ignore_index=True)
    
    full_unm_meisai['規格コード型番1'] = full_unm_meisai['規格コード型番1'].fillna('（コード未読取）')
    full_unm_nouhin['規格コード型番1'] = full_unm_nouhin['規格コード型番1'].fillna('（コード未読取）')
    
    depts = sorted(list(set(full_unm_meisai['部署名_統一'].dropna().tolist() + full_unm_nouhin['部署名_統一'].dropna().tolist())))
    report_rows = []
    
    for dept in depts:
        m_dept_df = full_unm_meisai[full_unm_meisai['部署名_統一'] == dept].reset_index(drop=True)
        n_dept_df = full_unm_nouhin[full_unm_nouhin['部署名_統一'] == dept].reset_index(drop=True)
        
        max_len = max(len(m_dept_df), len(n_dept_df))
        
        for i in range(max_len):
            row = {'対象部署': dept}
            
            if i < len(m_dept_df):
                m_row = m_dept_df.iloc[i]
                row['【明細】規格コード型番1'] = m_row['規格コード型番1']
                row['【明細】規格コード型番2'] = m_row.get('規格コード型番2', '') if pd.notna(m_row.get('規格コード型番2', '')) else ''
                row['【明細】数量'] = m_row['数量'] if pd.notna(m_row['数量']) else ''
                row['【明細】単位'] = m_row.get('単位', '') if pd.notna(m_row.get('単位', '')) else ''
            else:
                row['【明細】規格コード型番1'] = ''
                row['【明細】規格コード型番2'] = ''
                row['【明細】数量'] = ''
                row['【明細】単位'] = ''
                
            if i < len(n_dept_df):
                n_row = n_dept_df.iloc[i]
                row['【納品】規格コード型番1'] = n_row['規格コード型番1']
                row['【納品】規格コード型番2'] = n_row.get('規格コード型番2', '') if pd.notna(n_row.get('規格コード型番2', '')) else ''
                row['【納品】数量'] = n_row['数量'] if pd.notna(n_row['数量']) else ''
                row['【納品】単位'] = n_row.get('単位', '') if pd.notna(n_row.get('単位', '')) else ''
                row['【納品】伝票No'] = n_row.get('伝票No', '') if pd.notna(n_row.get('伝票No', '')) else ''
            else:
                row['【納品】規格コード型番1'] = ''
                row['【納品】規格コード型番2'] = ''
                row['【納品】数量'] = ''
                row['【納品】単位'] = ''
                row['【納品】伝票No'] = ''
                
            report_rows.append(row)
            
    final_report_df = pd.DataFrame(report_rows)
    
    meisai_invalid = meisai_invalid.copy()
    nouhin_invalid = nouhin_invalid.copy()
    meisai_invalid['区分'] = '明細書(データ不備)'
    nouhin_invalid['区分'] = '納品書(データ不備)'
    invalid_all = pd.concat([meisai_invalid, nouhin_invalid], ignore_index=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        final_report_df.to_excel(writer, sheet_name='部署別不一致リスト', index=False)
        invalid_all.to_excel(writer, sheet_name='データ不備リスト', index=False)
        all_matched_df.to_excel(writer, sheet_name='全一致データ一覧', index=False)
        
    print(f" Excel出力成功: {output_path} に保存されました！")