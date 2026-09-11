import pandas as pd
from steps import preprocess_data, run_loop_matching, export_excel_report

if __name__ == '__main__':
    print("==========================================")
    print("   自律ループ式 突合パイプライン実行開始   ")
    print("==========================================")
    
    # 1. 前処理（クレンジング ＋ 型番正規化）
    m_valid, n_valid, m_invalid, n_invalid = preprocess_data('meisai_gemini.csv', 'nouhin_gemini.csv')
    
    # 2. 多周回ループ照合エンジン実行
    matched_df, final_unm_m, final_unm_n = run_loop_matching(m_valid, n_valid)
    
    # 3. レポートExcel出力
    export_excel_report(
        final_unm_meisai=final_unm_m,
        final_unm_nouhin=final_unm_n,
        meisai_invalid=m_invalid,
        nouhin_invalid=n_invalid,
        all_matched_df=matched_df,
        output_path='突合結果_最終レポート.xlsx'
    )
    
    print("\n==========================================")
    print("              全処理が正常完了            ")
    print("==========================================")

    