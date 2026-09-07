import pandas as pd
import streamlit as st

# 元の main.py と同じ関数をそのまま呼び出し
from steps import preprocess_data, run_loop_matching, export_excel_report

st.set_page_config(page_title="CSV直接突合アプリ", layout="wide")
st.title("📊 CSV直接突合システム")

st.subheader("1. CSVファイルのアップロード")
col1, col2 = st.columns(2)

with col1:
    meisai_csv = st.file_uploader("明細CSVをドロップ", type=["csv"], key="meisai")
with col2:
    nouhin_csv = st.file_uploader("納品CSVをドロップ", type=["csv"], key="nouhin")

# 実行ボタン処理
if st.button("🚀 突合処理を開始", type="primary"):
    if not meisai_csv or not nouhin_csv:
        st.warning("明細CSVと納品CSVの両方をアップロードしてください。")
    else:
        with st.spinner("突合エンジンを実行中..."):
            try:
                # 1. 受け取ったCSVデータをそのまま一切加工せずに保存
                with open("meisai_temp.csv", "wb") as f:
                    f.write(meisai_csv.getbuffer())
                with open("nouhin_temp.csv", "wb") as f:
                    f.write(nouhin_csv.getbuffer())

                # 2. main.py と完全に同じ処理を直接実行
                m_valid, n_valid, m_invalid, n_invalid = preprocess_data("meisai_temp.csv", "nouhin_temp.csv")
                matched_df, final_unm_m, final_unm_n = run_loop_matching(m_valid, n_valid)

                st.success("✨ 突合処理が完了しました！")
                st.subheader("2. 突合結果")

                # 無効データ（コード未読取等）も未一致リストに統合
                all_unm_m = pd.concat([final_unm_m, m_invalid], ignore_index=True) if not m_invalid.empty else final_unm_m
                all_unm_n = pd.concat([final_unm_n, n_invalid], ignore_index=True) if not n_invalid.empty else final_unm_n

                # 部署順ソート ＆ 1からの連番インデックス付与
                def prepare_display_df(df):
                    if df.empty:
                        return df
                    sort_col = "部署名_統一" if "部署名_統一" in df.columns else ("部署名" if "部署名" in df.columns else None)
                    if sort_col:
                        df = df.sort_values(by=sort_col)
                    df = df.reset_index(drop=True)
                    df.index = df.index + 1
                    return df

                # 表示用データの作成
                sorted_unm_m = prepare_display_df(all_unm_m)
                sorted_unm_n = prepare_display_df(all_unm_n)
                sorted_matched = prepare_display_df(matched_df)

                # 画面表示（2タブ構成）
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

                # Excel出力
                output_excel_path = "突合結果_最終レポート.xlsx"
                export_excel_report(
                    final_unm_meisai=final_unm_m,
                    final_unm_nouhin=final_unm_n,
                    meisai_invalid=m_invalid,
                    nouhin_invalid=n_invalid,
                    all_matched_df=matched_df,
                    output_path=output_excel_path,
                )

                st.divider()
                with open(output_excel_path, "rb") as f:
                    st.download_button(
                        label="📥 突合結果Excelをダウンロード",
                        data=f,
                        file_name=output_excel_path,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

            except Exception as e:
                st.error(f"突合処理中にエラーが発生しました: {e}")