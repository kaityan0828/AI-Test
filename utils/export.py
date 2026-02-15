import pandas as pd
import io

def export_to_excel(ticker_info, hist_data, indicators_df, forecast_df, sentiment_label):
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # タイムゾーン情報を削除 (Excelはタイムゾーン付きの日時を非サポートのため)
        hist_data_clean = hist_data.copy()
        if hist_data_clean.index.tz is not None:
            hist_data_clean.index = hist_data_clean.index.tz_localize(None)
            
        indicators_df_clean = indicators_df.copy()
        if indicators_df_clean.index.tz is not None:
            indicators_df_clean.index = indicators_df_clean.index.tz_localize(None)
            
        forecast_df_clean = forecast_df.copy()
        if not forecast_df_clean.empty and forecast_df_clean.index.tz is not None:
            forecast_df_clean.index = forecast_df_clean.index.tz_localize(None)

        # 1. サマリー情報
        summary_data = {
            '項目': ['銘柄', '現在値', '会社名', '業種', 'AI感情分析'],
            '内容': [
                ticker_info.get('symbol', 'N/A'),
                ticker_info.get('currentPrice', 'N/A'),
                ticker_info.get('longName', 'N/A'),
                ticker_info.get('industry', 'N/A'),
                sentiment_label
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. ヒストリカルデータ
        hist_data_clean.to_excel(writer, sheet_name='Historical Data')
        
        # 3. テクニカル指標付きデータ
        indicators_df_clean.to_excel(writer, sheet_name='Analysis')
        
        # 4. 未来予測
        forecast_df_clean.to_excel(writer, sheet_name='Forecast')
        
    return output.getvalue()
