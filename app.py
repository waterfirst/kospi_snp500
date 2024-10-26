import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# 페이지 설정
st.set_page_config(page_title="KOSPI & S&P 500 지수 비교 및 분석", layout="wide")

# 제목
st.title("KOSPI와 S&P 500 지수 비교 및 상관관계 분석")


# 데이터 가져오기 함수
# 데이터 가져오기 함수
@st.cache_data(ttl=3600)  # 1시간 캐시
def get_data(start_date, end_date):
    try:
        # KOSPI 데이터 가져오기 (여러 티커 시도)
        kospi_tickers = ["^KS11", "^KS11.KS", "KS11.KS"]
        kospi = None
        for ticker in kospi_tickers:
            try:
                kospi = yf.download(
                    ticker, start=start_date, end=end_date, progress=False
                )
                if not kospi.empty:
                    break
            except:
                continue

        # S&P 500 데이터 가져오기 (여러 티커 시도)
        sp500_tickers = ["^GSPC", "^SPX", "SPX"]
        sp500 = None
        for ticker in sp500_tickers:
            try:
                sp500 = yf.download(
                    ticker, start=start_date, end=end_date, progress=False
                )
                if not sp500.empty:
                    break
            except:
                continue

        # 데이터 검증
        if kospi is None or kospi.empty:
            st.error("KOSPI 데이터를 가져올 수 없습니다.")
            return None, None

        if sp500 is None or sp500.empty:
            st.error("S&P 500 데이터를 가져올 수 없습니다.")
            return None, None

        # 데이터 전처리
        kospi = kospi.astype(float)
        sp500 = sp500.astype(float)

        return kospi, sp500

    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
        return None, None


# 날짜 범위 선택
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "시작 날짜",
        value=datetime(2023, 1, 1),
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now(),
    )
with col2:
    end_date = st.date_input(
        "종료 날짜",
        value=datetime.now(),
        min_value=datetime(2020, 1, 1),
        max_value=datetime.now(),
    )

if start_date <= end_date:
    # Convert date to datetime for yfinance
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.min.time())

    # 데이터 로드
    kospi, sp500 = get_data(start_datetime, end_datetime)

    if kospi is not None and sp500 is not None and not kospi.empty and not sp500.empty:
        # 이중 축 그래프 생성
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])

        fig1.add_trace(
            go.Scatter(
                x=kospi.index,
                y=kospi["Close"],
                name="KOSPI",
                line=dict(color="blue"),
            ),
            secondary_y=False,
        )

        fig1.add_trace(
            go.Scatter(
                x=sp500.index,
                y=sp500["Close"],
                name="S&P 500",
                line=dict(color="red"),
            ),
            secondary_y=True,
        )

        fig1.update_layout(
            title="KOSPI와 S&P 500 지수 비교 (이중 축)",
            xaxis_title="날짜",
            legend_title="지수",
            hovermode="x unified",
        )

        fig1.update_yaxes(title_text="KOSPI", secondary_y=False)
        fig1.update_yaxes(title_text="S&P 500", secondary_y=True)

        st.plotly_chart(fig1, use_container_width=True)

        # 상관관계 분석
        combined_data = pd.concat(
            [sp500["Close"], kospi["Close"]],
            axis=1,
            keys=["S&P 500", "KOSPI"],
        ).dropna()

        if not combined_data.empty:
            # 산점도
            fig2 = px.scatter(
                combined_data, x="S&P 500", y="KOSPI", title="S&P 500 vs KOSPI 산점도"
            )

            # 회귀선 추가
            x = combined_data["S&P 500"]
            y = combined_data["KOSPI"]
            coeffs = np.polyfit(x, y, deg=1)
            line = coeffs[0] * x + coeffs[1]
            fig2.add_trace(
                go.Scatter(x=x, y=line, mode="lines", name="Regression Line")
            )

            st.plotly_chart(fig2, use_container_width=True)

            # 상관관계 분석
            correlation = combined_data["KOSPI"].corr(combined_data["S&P 500"])
            st.subheader("상관관계 분석")
            st.write(f"KOSPI와 S&P 500의 상관계수: {correlation:.4f}")

            # 선행성 분석
            max_lag = 10
            lags = range(1, max_lag + 1)
            correlations = []

            for lag in lags:
                lag_corr = combined_data["KOSPI"].corr(
                    combined_data["S&P 500"].shift(lag)
                )
                correlations.append(lag_corr)

            max_correlation = max(correlations)
            best_lag = lags[correlations.index(max_correlation)]

            st.write(
                f"최대 상관계수: {max_correlation:.4f}, 발생 시점: {best_lag}일 전"
            )

            # 데이터 표시
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("KOSPI 데이터")
                st.dataframe(kospi)
            with col2:
                st.subheader("S&P 500 데이터")
                st.dataframe(sp500)

        else:
            st.warning("선택된 기간에 대한 데이터가 충분하지 않습니다.")
else:
    st.error("종료 날짜는 시작 날짜보다 커야 합니다.")
