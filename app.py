import streamlit as st
import pandas as pd
import io
import re
import time
import random
import pypdf
from google import genai
from google.genai import types

# ---------------------------------------------------------
# ページ設定・タイトルの表示
# ---------------------------------------------------------
st.set_page_config(page_title="PDF自動突合システム", layout="wide")
st.title("📄 PDF自動突合システム")

# サイドバーでAPIキーを受け取る
st.sidebar.header("設定")
api_key = st.sidebar.text_input("Gemini API Key", type="password")

st.subheader("1. PDFファイルのアップロード")
col1, col2 = st.columns(2)

with col1:
    meisai_pdf = st.file_uploader("明細PDFをドロップ", type=["pdf"], key="meisai")
with col2:
    nouhin_pdf = st.file_uploader("納品PDFをドロップ", type=["pdf"], key="nouhin")


# ---------------------------------------------------------
# 補助関数
# ---------------------------------------------------------
def clean_csv_response(text: str) -> str:
    match = re.search(r'```(?:csv)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    return text.strip()


# ---------------------------------------------------------
# PDFページ分割処理（フリーズ防止 ＆ 高速化 ＆ 1本バー連携）
# ---------------------------------------------------------
def process_pdf_with_backoff(pdf_file, client, prompt, progress_bar, start_page_offset, total_all_pages):
    pdf_reader = pypdf.PdfReader(pdf_file)
    all_dfs = []

    for i, page in enumerate(pdf_reader.pages):
        # 全体通算ページ数から全体の進捗率（%）を更新
        current_page = start_page_offset + i + 1
        percent = int((current_page / total_all_pages) * 100)
        progress_bar.progress(percent)

        writer = pypdf.PdfWriter()
        writer.add_page(page)
        page_bytes_io = io.BytesIO()
        writer.write(page_bytes_io)
        page_bytes = page_bytes_io.getvalue()

        response = None
        last_error = None

        # 最大3回までのリトライに制限（長時間フリーズを防止）
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model='gemini-3.8-flash',
                    contents=[
                        types.Part.from_bytes(data=page_bytes, mime_type='application/pdf'),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.0
                    )
                )
                last_error = None
                break
            except Exception as e:
                last_error = e
                err_msg = str(e)
                # 一時的な混雑エラー（503/429等）は1秒〜1.5秒だけ待って再試行
                if any(code in err_msg for code in ["503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"]):
                    time.sleep(1.0 + random.uniform(0.1, 0.5))
                else:
                    # 認証エラーなどの致命的エラーは即中断
                    break
        
        # 3回失敗した場合は画面にエラーを出して安全停止
        if last_error is not None:
            st.error(f"⚠️ {current_page}ページ目でエラーが発生しました: {last_error}")
            st.stop()

        # ページ間の通常インターバル（0.5秒）
        time.sleep(0.5)

        if response and response.text:
            cleaned_csv = clean_csv_response(response.text)
            if cleaned_csv:
                try:
                    df_page = pd.read_csv(io.StringIO(cleaned_csv))
                    all_dfs.append(df_page)
                except Exception:
                    pass

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


# ---------------------------------------------------------
# 抽出関数（明細書・納品書）
# ---------------------------------------------------------
def extract_meisai_csv(pdf_file, client, progress_bar, start_page_offset, total_all_pages):
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
(例) : 「JF-3YSL35Q (型番 110500240)/1箱」とある場合 → 「JF-3YSL35Q/110500240」と出力

・規格コード型番2:
「(例) : (123456789)」のように括弧内の数値コードがある場合のみ「123456789」を出力してください。無い場合は空欄。

【2. 数量と単位の絶対ルール】
・「数量」は数値のみ出力。
・「単位」は「箱」「個」「枚」「本」「セット」等の文字を出力。

コードブロック内の純粋なcsv形式のみを出力してください。
"""
    return process_pdf_with_backoff(pdf_file, client, prompt, progress_bar, start_page_offset, total_all_pages)


def extract_nouhin_csv(pdf_file, client, progress_bar, start_page_offset, total_all_pages):
    prompt = """
あなたは添付された「納品書」の画像から文字列を読み取り、CSVデータを作成するOCR専門システムです。

【抽出項目と出力順序】
対象部署,規格コード型番1,規格コード型番2,数量,単位,伝票No

1行目のヘッダー行は必ず固定出力:
対象部署,規格コード型番1,規格コード型番2,数量,単位,伝票No

【抽出ルール】
1. 対象部署: 納品先や部署名を抽出。
2. 規格コード型番1: 「品名・規格」欄のボックス枠内の英数字コードを抽出。
3. 規格コード型番2: 補足コードがあれば抽出。無ければ空欄。
4. 数量: 数値のみ抽出。
5. 単位: 単位テキストを抽出。
6. 伝票No: 伝票番号を抽出。

コードブロック内の純粋なcsv形式のみを出力してください。
"""
    return process_pdf_with_backoff(pdf_file, client, prompt, progress_bar, start_page_offset, total_all_pages)


# ---------------------------------------------------------
# 実行ボタン処理（1本のバーで全体進捗）
# ---------------------------------------------------------
if st.button("🚀 突合処理を開始", type="primary"):
    if not api_key:
        st.error("サイドバーに Gemini API キーを入力してください。")
    elif not meisai_pdf or not nouhin_pdf:
        st.warning("明細PDFと納品PDFの両方をアップロードしてください。")
    else:
        try:
            # APIキーの前後スペースを除去してクライアント初期化
            clean_key = api_key.strip()
            client = genai.Client(api_key=clean_key)

            # 1. 両方のPDFの合計ページ数を計算
            meisai_reader = pypdf.PdfReader(meisai_pdf)
            nouhin_reader = pypdf.PdfReader(nouhin_pdf)
            total_meisai_pages = len(meisai_reader.pages)
            total_nouhin_pages = len(nouhin_reader.pages)
            total_all_pages = total_meisai_pages + total_nouhin_pages

            # 2. テキストなしの1本プログレスバーを生成
            progress_bar = st.progress(0)

            # 3. 明細PDFの解析（0% -> 明細分の%）
            df_meisai = extract_meisai_csv(meisai_pdf, client, progress_bar, 0, total_all_pages)

            # 4. 納品PDFの解析（明細分の% -> 100%）
            df_nouhin = extract_nouhin_csv(nouhin_pdf, client, progress_bar, total_meisai_pages, total_all_pages)

            # バー消去
            progress_bar.empty()

            # 読み取りチェックガード
            if df_meisai.empty:
                st.error("明細PDFからデータを読み取れませんでした。")
                st.stop()
            if df_nouhin.empty:
                st.error("納品PDFからデータを読み取れませんでした。プロンプトまたはファイル内容を確認してください。")
                st.stop()

            # 一時ファイル保存
            df_meisai.to_csv("meisai_temp.csv", index=False, encoding="utf-8-sig")
            df_nouhin.to_csv("nouhin_temp.csv", index=False, encoding="utf-8-sig")

            # 突合処理の実行（※前処理・突合関数が別ファイル等で用意されている前提の呼び出しです）
            m_df = load_and_preprocess_meisai("meisai_temp.csv")
            n_df = load_and_preprocess_nouhin("nouhin_temp.csv")
            
            m_valid, m_invalid = validate_meisai(m_df)
            n_valid, n_invalid = validate_nouhin(n_df)
            
            matched_df, sorted_unm_m, sorted_unm_n, sorted_matched, final_unm_m, final_unm_n = match_data(m_valid, n_valid)

            output_excel_path = "突合結果_最終レポート.xlsx"
            export_excel_report(
                final_unm_meisai=final_unm_m,
                final_unm_nouhin=final_unm_n,
                meisai_invalid=m_invalid,
                nouhin_invalid=n_invalid,
                all_matched_df=matched_df,
                output_path=output_excel_path
            )

            # 画面結果表示
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

            st.divider()
            with open(output_excel_path, "rb") as f:
                st.download_button(
                    label="📊 突合結果Excelをダウンロード",
                    data=f,
                    file_name=output_excel_path,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"突合処理中にエラーが発生しました: {e}")
            st.stop()
