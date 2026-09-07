import os
import io
import re
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# 既存の突合処理モジュール
from steps import preprocess_data, run_loop_matching, export_excel_report

st.set_page_config(page_title="PDF自動突合アプリ", layout="wide")
st.title("📄 PDF自動突合システム")

# サイドバー設定
st.sidebar.header("設定")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.subheader("1. PDFファイルのアップロード")
col1, col2 = st.columns(2)

with col1:
    meisai_pdf = st.file_uploader("明細PDFをドロップ", type=["pdf"], key="meisai")
with col2:
    nouhin_pdf = st.file_uploader("納品PDFをドロップ", type=["pdf"], key="nouhin")


# Geminiからの返答から純粋なCSV部分だけを切り出す関数
def clean_csv_response(text: str) -> str:
    match = re.search(r'```(?:csv)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    return text.strip()


# 明細書専用の抽出関数
def extract_meisai_csv(pdf_file, client):
    prompt = """
あなたは添付された「明細書」の画像から文字列を読み取り、CSVデータを作成するOCR専門システムです。

【抽出項目と出力順序】
部署名,規格コード型番1,規格コード型番2,数量,単位

1行目のヘッダー行は必ず固定出力:
部署名,規格コード型番1,規格コード型番2,数量,単位

【1. 規格コード型番の絶対ルール】
「品名/規格/梱包内訳」のボックス枠内から、英数字のコードを読み取ってください。

・規格コード型番1:
枠内にある英数字コードを【左から順にすべて】読み取り、スラッシュ「/」で繋いで出力してください。
(例)：「JF-3YSL35Q (型番 110500240)/1箱」とある場合 → 「JF-3YSL35Q/110500240」と出力
(例)：「AGS-6285-3/1箱」とある場合 → 「AGS-6285-3」と出力

・規格コード型番2:
常に空欄（カンマのみ）にしてください。

・除外テキスト：
日本語の品名、「型番」という日本語単語、カッコ「()」、「/1箱=20セット」などの梱包数はすべて除外して、純粋な英数字コードのみを抽出してください。

【2. 部署名・数量・単位】
・部署名：「部門」列の文字列を見えたまま抽出（例：c4南病棟）。
・数量：「数量」列の半角数字のみ抽出。
・単位：「単位」列の文字列（箱、本 等）を抽出。

【出力ルール】
・コードブロック内の純粋なcsv形式のみを出力してください。
"""
    pdf_bytes = pdf_file.read()
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
            prompt
        ]
    )
    
    cleaned_csv = clean_csv_response(response.text)
    return pd.read_csv(io.StringIO(cleaned_csv))


# 納品書専用の抽出関数
def extract_nouhin_csv(pdf_file, client):
    prompt = """
あなたは添付された「納品書」の画像から、型番・部署名・数量・伝票番号を極めて正確に読み取り、CSVデータとして書き出す文字認識 (OCR) 専門システムです。

【抽出項目と出力順序】
以下の6つの列を順に出力してください (ヘッダー行固定):
部署名,規格コード型番1,規格コード型番2,数量,単位,伝票No

【1. 規格コード型番の絶対抽出・分離ルール】 ※最重要
「品名 / 規格」枠内から、型番・品番コード (英数字) を以下の【横方向 (同一行) 限定ルール】に従って正確に切り出してください。

・規格コード型番1:
「品名 / 規格」枠内の中段 (1行目の品名の下) にある【左端の英数字コード】(例: 000MD0553, 588-002-01, 076162, 311002 等) を「規格コード型番1」として抽出してください。

・規格コード型番2 (【絶対条件】同じ行の右隣からの抽出):
「規格コード型番1」と【同一行 (同じ高さの横並び)】に印字されているコード (例: 「000MD0553 515-A」→ 515-A、「588-002-01 N10039」→ N10039) のみを「規格コード型番2」として抽出してください。

・【絶対禁止命令】下段 (改行先) の6桁数字の除外:
改行された下の行 (最下段) に印字されている数字のみの6桁コード (例: 030312, 009787, 006032, 009568 等) は、社内の管理枝番です。
これを「規格コード型番2」として抽出することは絶対厳禁・禁止とします。同一行の右隣に2つ目の型番がない場合、規格コード型番2は必ず「空欄 (カンマのみ)」にしてください。

・その他除外条件:
「M 35-40CM」などのサイズ・単位表記や、「20入」「3,000入」などの梱包数は型番に入れないでください。

【2. 部署名・数量・伝票Noの抽出ルール】
1. 部署名: 用紙左上「京都第二赤十字病院」の直下にある部署名 (例: 形成外科, 調度課倉庫 等) をそのまま抽出してください。
2. 数量: 「数量」列の数値 (例: 1.0 -> 1) のみを抽出してください。
3. 単位: 「単位」列の文字列 (箱, ケース 等) をそのまま抽出してください。
4. 伝票No: 用紙右上の「No.」直後にある6〜7桁の半角数字 (例: 1031466) を出力してください。

【出力ルール】
・純粋なcsv形式 (コードブロック内) のみを出力してください。前置きや挨拶は一切禁止です。
"""
    pdf_bytes = pdf_file.read()
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type='application/pdf'),
            prompt
        ]
    )
    
    cleaned_csv = clean_csv_response(response.text)
    return pd.read_csv(io.StringIO(cleaned_csv))


# 実行ボタン処理
if st.button("🚀 突合処理を開始", type="primary"):
    if not api_key:
        st.error("サイドバーに Gemini API キーを入力してください。")
    elif not meisai_pdf or not nouhin_pdf:
        st.warning("明細PDFと納品PDFの両方をアップロードしてください。")
    else:
        with st.spinner("Geminiが各PDFを解析中..."):
            try:
                client = genai.Client(api_key=api_key)
                
                df_meisai = extract_meisai_csv(meisai_pdf, client)
                df_nouhin = extract_nouhin_csv(nouhin_pdf, client)
                
                # preprocess_dataに渡す一時ファイルとして保存
                df_meisai.to_csv("meisai_temp.csv", index=False, encoding="utf-8-sig")
                df_nouhin.to_csv("nouhin_temp.csv", index=False, encoding="utf-8-sig")
                
            except Exception as e:
                st.error(f"PDFの解析中にエラーが発生しました: {e}")
                st.stop()

        with st.spinner("突合エンジンを実行中..."):
            try:
                m_valid, n_valid, m_invalid, n_invalid = preprocess_data("meisai_temp.csv", "nouhin_temp.csv")
                matched_df, final_unm_m, final_unm_n = run_loop_matching(m_valid, n_valid)
                
                st.success("✨ 突合処理が完了しました！")
                st.subheader("2. 突合結果")
                
                # 無効データ（コード未読取等）も未一致リストに統合
                all_unm_m = pd.concat([final_unm_m, m_invalid], ignore_index=True) if not m_invalid.empty else final_unm_m
                all_unm_n = pd.concat([final_unm_n, n_invalid], ignore_index=True) if not n_invalid.empty else final_unm_n

                # 部署順ソート ＆ 1からの連番インデックス付与を行う関数
                def prepare_display_df(df):
                    if df.empty:
                        return df
                    sort_col = "部署名_統一" if "部署名_統一" in df.columns else ("部署名" if "部署名" in df.columns else None)
                    if sort_col:
                        df = df.sort_values(by=sort_col)
                    # 1からの連番インデックスを設定
                    df = df.reset_index(drop=True)
                    df.index = df.index + 1
                    return df

                # 表示用データの作成
                sorted_unm_m = prepare_display_df(all_unm_m)
                sorted_unm_n = prepare_display_df(all_unm_n)
                sorted_matched = prepare_display_df(matched_df)

                # 2タブ構成（不一致リスト優先）
                tab1, tab2 = st.tabs(["⚠️ 不一致リスト", "✅ 一致リスト"])
                
                with tab1:
                    st.markdown("### 【明細書側】未一致データ（部署順）")
                    st.dataframe(sorted_unm_m, use_container_width=True)
                    
                    st.divider()
                    
                    st.markdown("### 【納品書側】未一致データ（部署順）")
                    st.dataframe(sorted_unm_n, use_container_width=True)

                with tab2:
                    st.markdown("### 完全一致データ（部署順）")
                    st.dataframe(sorted_matched, use_container_width=True)

                # 報告用Excelの出力 & ダウンロードボタン
                output_excel_path = "突合結果_最終レポート.xlsx"
                export_excel_report(
                    final_unm_meisai=final_unm_m,
                    final_unm_nouhin=final_unm_n,
                    meisai_invalid=m_invalid,
                    nouhin_invalid=n_invalid,
                    all_matched_df=matched_df,
                    output_path=output_excel_path
                )
                
                st.divider()
                with open(output_excel_path, "rb") as f:
                    st.download_button(
                        label="📥 突合結果Excelをダウンロード",
                        data=f,
                        file_name=output_excel_path,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                st.error(f"突合処理中にエラーが発生しました: {e}")