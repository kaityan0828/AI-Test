import pandas as pd
import numpy as np
from transformers import pipeline
import yfinance as yf
from sklearn.linear_model import LinearRegression

# 株価のテクニカル指標を計算する
def calculate_indicators(df):
    # 移動平均
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (相対力指数)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

# ニュースの感情分析を行う
def analyze_sentiment(news_list):
    """
    ニュースのタイトルから感情を分析する
    """
    if not news_list or not isinstance(news_list, list):
        return "データなし", 0.5
    
    try:
        # device=-1 を追加してCPUを使用するように指定 (メタテンソルエラー回避のため)
        classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1)
        
        results = []
        # 有効なタイトルを安全に抽出
        for item in news_list[:8]:
            if isinstance(item, dict):
                content = item.get('title')
                if content:
                    sentiment = classifier(content)[0]
                    score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else 1 - sentiment['score']
                    results.append(score)
        
        avg_score = sum(results) / len(results) if results else 0.5
        
        if avg_score > 0.6:
            label = "ポジティブ 😊 (買い傾向)"
        elif avg_score < 0.4:
            label = "ネガティブ 😞 (売り傾向)"
        else:
            label = "中立 😐"
            
        return label, avg_score
    except Exception as e:
        # エラー発生時は簡易的なキーワードベースの分析をfallbackとして実行
        positive_words = ['up', 'rise', 'growth', 'profit', 'positive', 'buy', 'bullish']
        negative_words = ['down', 'fall', 'loss', 'negative', 'sell', 'bearish']
        
        titles = [n.get('title', '') for n in news_list if isinstance(n, dict)]
        score = 0.5
        for t in titles[:10]:
            t_lower = t.lower()
            if any(w in t_lower for w in positive_words): score += 0.05
            if any(w in t_lower for w in negative_words): score -= 0.05
        
        score = max(0.1, min(0.9, score))
        label = "中立 (簡易分析)"
        return label, score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 未来予測を行う (多項式回帰による強化版)
def predict_stock_price(df, days_to_predict=7):
    """
    株価の未来予測を行う。多項式回帰を使用し、信頼区間も計算する。
    """
    if len(df) < 10:
        return pd.DataFrame(columns=['Predicted_Close', 'Upper_Bound', 'Lower_Bound'])

    # インデックスの重複を排除し、コピーを作成
    df_copy = df[~df.index.duplicated(keep='last')].copy()
    
    # 日付列を確実に datetime 型にしてから序数に変換
    df_copy.index = pd.to_datetime(df_copy.index)
    df_copy['Date_Ordinal'] = df_copy.index.map(pd.Timestamp.toordinal)
    
    # 欠損値を除去
    df_copy = df_copy.dropna(subset=['Close', 'Date_Ordinal'])
    
    if len(df_copy) < 10:
        return pd.DataFrame(columns=['Predicted_Close', 'Upper_Bound', 'Lower_Bound'])

    X = df_copy[['Date_Ordinal']].values
    y = df_copy['Close'].values
    
    try:
        # 多項式回帰 (Degree 2)
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # 標準誤差の計算
        preds_train = model.predict(X_poly)
        mse = np.mean((y - preds_train)**2)
        std_error = np.sqrt(mse) if mse > 0 else np.std(y) * 0.1
        
        # 未来の日付を作成
        last_date = df_copy.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_predict + 1)]
        future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
        future_poly = poly.transform(future_ordinals)
        
        # 予測
        predictions = model.predict(future_poly).flatten()
        
        # 信頼区間の計算
        z_score = 1.96
        upper_bound = [float(p + (z_score * std_error * np.sqrt(i+1) * 0.5)) for i, p in enumerate(predictions)]
        lower_bound = [float(p - (z_score * std_error * np.sqrt(i+1) * 0.5)) for i, p in enumerate(predictions)]
        
        prediction_df = pd.DataFrame({
            'Predicted_Close': predictions,
            'Upper_Bound': upper_bound,
            'Lower_Bound': lower_bound
        }, index=future_dates)
        prediction_df.index.name = 'Date'
        
        return prediction_df
    except Exception as e:
        print(f"Prediction calculation error: {e}")
        return pd.DataFrame(columns=['Predicted_Close', 'Upper_Bound', 'Lower_Bound'])
