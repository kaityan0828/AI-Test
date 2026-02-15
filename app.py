import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from transformers import pipeline
import time

from utils.analysis import calculate_indicators, analyze_sentiment, predict_stock_price
from utils.export import export_to_excel
from deep_translator import GoogleTranslator

# ページ設定
st.set_page_config(
    page_title="AI株価分析ダッシュボード",
    page_icon="📈",
    layout="wide"
)

# セッション状態の初期化
if "ticker_symbol" not in st.session_state:
    st.session_state.ticker_symbol = "7267.T"
if "messages" not in st.session_state:
    st.session_state.messages = []

# カスタムCSSでプロの雰囲気に
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .sentiment-box {
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 AI株価分析・予測ダッシュボード")
st.markdown("API連携、AI分析、未来予測を備えた次世代投資支援ツール")

# サイドバー設定
with st.sidebar:
    st.header("🔍 銘柄検索")
    ticker_symbol = st.text_input("ティッカーシンボルを入力", st.session_state.ticker_symbol).upper()
    if ticker_symbol != st.session_state.ticker_symbol:
        st.session_state.ticker_symbol = ticker_symbol
        st.rerun()
        
    st.caption("※日本株はコードの後に '.T' を付けてください (例: 7203.T)")
    
    period = st.selectbox(
        "分析期間を選択",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )
    
    st.divider()
    st.header("🚀 オプション")
    show_prediction = st.checkbox("未来予測を表示", value=True)
    prediction_days = st.slider("予測期間 (日)", 1, 30, 7)
    show_sentiment = st.checkbox("AIニュース感情分析を有効化", value=True)

# 感情分析モデルのキャッシュ
@st.cache_resource
def get_sentiment_pipeline():
    try:
        # メタテンソルエラーを避けるために明示的に CPU (device=-1) を指定
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
    except Exception as e:
        st.error(f"モデルのロードに失敗しました: {e}")
        return None

# 翻訳機能のキャッシュ
@st.cache_data
def translate_text(text, target_lang='ja'):
    if not text or text == "タイトルなし" or text == "不明" or text == "No Title" or text == "Unknown":
        return text
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text

# データ取得
@st.cache_data(ttl=3600)
def load_data(symbol, p):
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=p)
    info = ticker.info
    news = ticker.news
    return df, info, news

# Excel出力用データクリーニング
def clean_for_excel(df):
    if df is None or df.empty:
        return df
    df_clean = df.copy()
    
    # インデックスのタイムゾーンを削除
    try:
        if hasattr(df_clean.index, 'tz') and df_clean.index.tz is not None:
            df_clean.index = df_clean.index.tz_convert(None).tz_localize(None)
    except:
        try:
            df_clean.index = df_clean.index.tz_localize(None)
        except:
            pass

    # 全カラムをループしてタイムゾーンを削除
    for col in df_clean.columns:
        if pd.api.types.is_datetime64tz_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].dt.tz_convert(None).dt.tz_localize(None)
        elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
            try:
                df_clean[col] = df_clean[col].dt.tz_localize(None)
            except:
                pass
    return df_clean

try:
    with st.spinner(f"{ticker_symbol} のデータを取得中..."):
        df, info, news = load_data(ticker_symbol, period)
    
    # 予測データの初期化
    prediction_df = pd.DataFrame()
    
    if df.empty:
        st.error(f"銘柄 '{ticker_symbol}' のデータが見つかりませんでした。正しいシンボルを入力してください。")
    else:
        # 通貨の自動判別
        currency = info.get('currency', 'USD')
        currency_symbol = "¥" if currency == "JPY" else "$"
        
        # メインメトリクス表示
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]
        delta = current_price - prev_price
        
        with col1:
            st.metric("現在値", f"{currency_symbol}{current_price:,.2f}", f"{delta:,.2f} ({delta/prev_price:.2%})")
        with col2:
            # 銘柄名も日本語訳があれば使用、なければそのまま
            raw_name = info.get('shortName', ticker_symbol)
            st.metric("銘柄名", translate_text(raw_name) if currency == "JPY" else raw_name)
        with col3:
            st.metric("市場", info.get('exchange', 'N/A'))
        with col4:
            industry = info.get('industry', 'N/A')
            st.metric("業種", translate_text(industry) if currency == "JPY" else industry)

        # タブ構成
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 チャート分析", "🤖 AIニュース分析", "📅 未来予測", "💬 投資AIチャット", "📁 銘柄カタログ"])

        with tab1:
            st.subheader("株価チャート & テクニカル指標")
            
            # 指標計算
            df_with_inds = calculate_indicators(df.copy())
            
            fig = go.Figure()
            # ローソク足
            fig.add_trace(go.Candlestick(
                x=df_with_inds.index,
                open=df_with_inds['Open'],
                high=df_with_inds['High'],
                low=df_with_inds['Low'],
                close=df_with_inds['Close'],
                name='株価'
            ))
            
            # 移動平均
            fig.add_trace(go.Scatter(x=df_with_inds.index, y=df_with_inds['SMA_20'], name='20日移動平均', line=dict(color='orange', width=1.5)))
            fig.add_trace(go.Scatter(x=df_with_inds.index, y=df_with_inds['SMA_50'], name='50日移動平均', line=dict(color='blue', width=1.5)))
            
            fig.update_layout(
                height=600,
                xaxis_rangeslider_visible=False,
                template="plotly_white",
                margin=dict(l=20, r=20, t=20, b=20)
            )
            fig.update_xaxes(tickformat="%Y年%m月")
            st.plotly_chart(fig, use_container_width=True)
            
            # RSI
            st.write("**RSI (相対力指数)**")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df_with_inds.index, y=df_with_inds['RSI'], name='RSI', line=dict(color='purple')))
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
            fig_rsi.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20), yaxis_range=[0, 100])
            fig_rsi.update_xaxes(tickformat="%m月")
            st.plotly_chart(fig_rsi, use_container_width=True)

        with tab2:
            st.subheader("AIセンチメント分析")
            if show_sentiment:
                try:
                    with st.spinner("ニュースを分析中..."):
                        classifier = get_sentiment_pipeline()
                        if classifier:
                            # ニュースタイトルのリストを作成
                            titles = []
                            if isinstance(news, list):
                                for n in news[:8]:
                                    if isinstance(n, dict):
                                        # 新旧両方のデータ構造に対応
                                        title = n.get('title') or n.get('content', {}).get('title')
                                        if title:
                                            titles.append(title)
                            
                            if titles:
                                sentiments = classifier(titles)
                                scores = [s['score'] if s['label'] == 'POSITIVE' else 1 - s['score'] for s in sentiments]
                                avg_score = sum(scores) / len(scores)
                                
                                label = "ポジティブ 😊" if avg_score > 0.6 else "ネガティブ 😞" if avg_score < 0.4 else "中立 😐"
                                st.markdown(f"**AI判断: {label} (スコア: {avg_score:.2f})**")
                                st.progress(avg_score)
                            else:
                                st.warning("分析可能なニュースがありませんでした。")
                        else:
                            st.warning("感情分析モデルがロードされていないため、分析をスキップします。")
                except Exception as e:
                    st.error(f"分析中にエラーが発生しました: {e}")
                
                st.write("**最新ニュース一覧 (自動翻訳):**")
                if isinstance(news, list):
                    for n in news[:5]:
                        if isinstance(n, dict):
                            # 新旧両方のデータ構造に対応
                            title_en = n.get('title') or n.get('content', {}).get('title', 'No Title')
                            publisher_en = n.get('publisher') or n.get('content', {}).get('publisher', 'Unknown')
                            link = n.get('link') or n.get('content', {}).get('link', '#')
                            
                            # 日本語に翻訳
                            title_ja = translate_text(title_en)
                            publisher_ja = translate_text(publisher_en)
                            
                            with st.expander(title_ja):
                                st.write(f"原文: {title_en}")
                                st.write(f"発行元: {publisher_ja} ({publisher_en})")
                                st.write(f"リンク: {link}")
                else:
                    st.write("ニュースデータが取得できませんでした。")
            else:
                st.info("サイドバーでAI分析を有効にしてください")

        with tab3:
            st.subheader("📅 AIによる未来予測チャート")
            if show_prediction:
                with st.spinner("AIがトレンドを予測中..."):
                    prediction_df = predict_stock_price(df, days_to_predict=prediction_days)
                
                if not prediction_df.empty and 'Upper_Bound' in prediction_df.columns:
                    fig_pred = go.Figure()
                    
                    # 実績データ
                    fig_pred.add_trace(go.Scatter(
                        x=df.index, y=df['Close'], 
                        name='実績値', line=dict(color='#1E88E5', width=3)
                    ))
                    
                    # 予測範囲 (信頼区間)
                    # DatetimeIndex を一旦リストに変換してから結合
                    fig_pred.add_trace(go.Scatter(
                        x=list(prediction_df.index) + list(prediction_df.index[::-1]),
                        y=list(prediction_df['Upper_Bound']) + list(prediction_df['Lower_Bound'][::-1]),
                        fill='toself',
                        fillcolor='rgba(255, 152, 0, 0.2)',
                        line=dict(color='rgba(255, 255, 255, 0)'),
                        hoverinfo="skip",
                        showlegend=True,
                        name='予測の幅 (信頼区間)'
                    ))
                    
                    # 予測データ
                    fig_pred.add_trace(go.Scatter(
                        x=prediction_df.index, y=prediction_df['Predicted_Close'], 
                        name='AI予測値', line=dict(color='#FF9800', width=3, dash='dash')
                    ))
                    
                    currency = info.get('currency', 'USD')
                    c_sym = "¥" if currency == "JPY" else "$"
                    
                    fig_pred.update_layout(
                        height=450,
                        margin=dict(l=20, r=20, t=20, b=20),
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    fig_pred.update_yaxes(tickprefix=c_sym, tickformat=",")
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # 予測サマリー
                    last_price = df['Close'].iloc[-1]
                    pred_price = prediction_df['Predicted_Close'].iloc[-1]
                    growth = (pred_price - last_price) / last_price
                    
                    trend_msg = "上昇" if growth > 0 else "下落"
                    trend_color = "green" if growth > 0 else "red"
                    
                    st.info(f"AIによる分析の結果、今後 {prediction_days} 日間で株価は **:{trend_color}[{trend_msg}] ({growth:+.2%})** する可能性があると予測されました。")
                    st.caption("※この予測は過去のトレンドに基づいた統計的な数値であり、投資の助言ではありません。")
                else:
                    st.warning("予測に必要なデータが不足しているか、計算に失敗しました。")
            else:
                st.info("左側のオプションから未来予測を有効にしてください。")

        with tab4:
            st.subheader("💬 投資プランナーAI")
            st.write(f"{ticker_symbol}に関する質問を入力してください。現在のデータに基づいて回答します。")

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # ユーザー入力
            if prompt := st.chat_input("例: 今は買い時ですか？"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    # 回答の生成ロジック
                    last_price = df_with_inds['Close'].iloc[-1]
                    last_rsi = df_with_inds['RSI'].iloc[-1]
                    sma20 = df_with_inds['SMA_20'].iloc[-1]
                    
                    # 再度通貨を判別
                    currency = info.get('currency', 'USD')
                    c_sym = "円" if currency == "JPY" else "ドル"
                    
                    response = ""
                    if "買い" in prompt or "買う" in prompt or "買い時" in prompt:
                        # 明確な売買シグナル判定
                        signal = "様子見 😐"
                        reason = []
                        
                        # 買いシグナル
                        if last_rsi < 30:
                            signal = "買い時！ 🚀"
                            reason.append(f"RSIが{last_rsi:.1f}で「売られすぎ」水準です。反発の可能性が高いです。")
                        elif last_price > sma20 and sma20 > df_with_inds['SMA_50'].iloc[-1]:
                            signal = "買い時！ 🚀"
                            reason.append(f"株価が上昇トレンド（現在の株価 > 20日平均 > 50日平均）にあり、勢いがあります。")
                            
                        # 売りシグナル
                        elif last_rsi > 70:
                            signal = "売り時！ 📉"
                            reason.append(f"RSIが{last_rsi:.1f}で「買われすぎ」水準です。調整が入る可能性があります。")
                        elif last_price < sma20 and sma20 < df_with_inds['SMA_50'].iloc[-1]:
                            signal = "売り時！ 📉"
                            reason.append(f"株価が下降トレンド（現在の株価 < 20日平均 < 50日平均）にあり、下落のリスクがあります。")
                        
                        # 様子見
                        else:
                            reason.append(f"RSIは{last_rsi:.1f}で中立圏内です。")
                            reason.append(f"移動平均線との位置関係も明確なトレンドを示していません。")
                            
                        response = f"### 判定: **{signal}**\n\n" + "\n".join([f"- {r}" for r in reason])
                    elif "予測" in prompt or "将来" in prompt or "今後" in prompt:
                        response = f"未来予測タブのトレンドラインを確認すると、短期的な傾向がわかります。現在は{last_price:,.1f}{c_sym}ですが、統計的にはトレンドを維持する可能性が高いです。"
                    elif "分析" in prompt or "教えて" in prompt:
                        response = f"{ticker_symbol}の現在の株価は{last_price:,.2f}{c_sym}です。直近の主要な動きはチャート分析ダッシュボード、市場の反応はAIニュースタブで詳細を確認できます。"
                    else:
                        response = f"ご質問ありがとうございます。{ticker_symbol}について分析をお手伝いします。現在は株価{last_price:,.2f}{c_sym}、RSI{last_rsi:.1f}といった状況です。何か具体的な分析について知りたいことはありますか？"

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

        with tab5:
            st.subheader("📁 人気銘柄カタログ")
            st.write("気になる銘柄を選んでください。瞬時に分析が切り替わります。")
            
            stock_catalog = {
                "🇯🇵 日本の主要株": [
                    {"name": "ホンダ (7267)", "ticker": "7267.T"},
                    {"name": "トヨタ (7203)", "ticker": "7203.T"},
                    {"name": "ソニーG (6758)", "ticker": "6758.T"},
                    {"name": "任天堂 (7974)", "ticker": "7974.T"},
                    {"name": "ソフトバンクG (9984)", "ticker": "9984.T"},
                    {"name": "キーエンス (6861)", "ticker": "6861.T"}
                ],
                "🇺🇸 米国の主要株": [
                    {"name": "Apple (AAPL)", "ticker": "AAPL"},
                    {"name": "NVIDIA (NVDA)", "ticker": "NVDA"},
                    {"name": "Microsoft (MSFT)", "ticker": "MSFT"},
                    {"name": "Tesla (TSLA)", "ticker": "TSLA"},
                    {"name": "Alphabet (GOOGL)", "ticker": "GOOGL"},
                    {"name": "Amazon (AMZN)", "ticker": "AMZN"}
                ],
                "🚀 成長・注目株": [
                    {"name": "三菱UFJ (8306)", "ticker": "8306.T"},
                    {"name": "ファーストリテイ (9983)", "ticker": "9983.T"},
                    {"name": "Netflix (NFLX)", "ticker": "NFLX"},
                    {"name": "Meta (META)", "ticker": "META"},
                    {"name": "Intel (INTC)", "ticker": "INTC"}
                ]
            }
            
            for category, stocks in stock_catalog.items():
                st.write(f"### {category}")
                cols = st.columns(3)
                for i, stock in enumerate(stocks):
                    with cols[i % 3]:
                        if st.button(stock["name"], key=f"cat_{stock['ticker']}", use_container_width=True):
                            st.session_state.ticker_symbol = stock["ticker"]
                            st.rerun()
                st.write("")

        # Excelエクスポートセクション
        st.divider()
        st.subheader("📁 データの蓄積とエクスポート")
        
        try:
            # データクリーニングを確実に実施
            df_clean = clean_for_excel(df)
            df_inds_clean = clean_for_excel(df_with_inds)
            df_forecast_clean = clean_for_excel(prediction_df if show_prediction else pd.DataFrame())
            
            final_sentiment = "分析実施済" if show_sentiment else "未実施"
            
            excel_data = export_to_excel(info, df_clean, df_inds_clean, df_forecast_clean, final_sentiment)
            
            st.download_button(
                label="📊 分析レポートをExcelでダウンロード",
                data=excel_data,
                file_name=f"{ticker_symbol}_AI_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.error(f"Excel用のデータ準備中にエラーが発生しました: {e}")
        
        st.success("データの蓄積が完了しました。最新の分析結果をExcel形式で保存できます。")

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
    st.info("ティッカーシンボルが正しいか確認してください（例: AppleならAAPL、トヨタなら7203.T）")

# フッター
st.divider()
st.caption("Powered by Streamlit, yfinance & Hugging Face Transformers. 投資の最終判断は自己責任で行ってください。")
