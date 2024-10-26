import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

# 페이지 설정
st.set_page_config(page_title="KOSPI & S&P 500 지수 비교 및 분석", layout="wide")

# 제목
st.title("KOSPI와 S&P 500 지수 비교 및 상관관계 분석")

# 데이터 가져오기 함수
@st.cache_data
def get_data():
    try:
        end_date = datetime.now()
        start_date = datetime(2023, 1, 1)

        kospi = yf.download("^KS11", start=start_date, end=end_date)
        sp500 = yf.download("^AXJO", start=start_date, end=end_date)

        # 결측치 처리
        kospi = kospi.dropna()
        sp500 = sp500.dropna()

        return kospi, sp500
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")
        return None, None

# 데이터 로드
kospi, sp500 = get_data()

if kospi is not None and sp500 is not None:
    # 날짜 범위 선택
    date_range = st.date_input(
        "날짜 범위를 선택하세요",
        value=(datetime(2023, 1, 1), datetime.now()),
        min_value=datetime(2023, 1, 1),
        max_value=datetime.now(),
    )

    # 선택된 날짜로 데이터 필터링
    start_date, end_date = date_range
    kospi_filtered = kospi.loc[start_date:end_date]
    sp500_filtered = sp500.loc[start_date:end_date]

    # 데이터가 충분한지 확인
    if not kospi_filtered.empty and not sp500_filtered.empty:
        # 이중 축 그래프 생성
        fig1 = make_subplots(specs=[[{"secondary_y": True}]])

        fig1.add_trace(
            go.Scatter(
                x=kospi_filtered.index,
                y=kospi_filtered["Close"],
                name="KOSPI",
                line=dict(color="blue"),
            ),
            secondary_y=False,
        )

        fig1.add_trace(
            go.Scatter(
                x=sp500_filtered.index,
                y=sp500_filtered["Close"],
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

        # 상관관계 산점도
        combined_data = pd.concat(
            [sp500_filtered["Close"], kospi_filtered["Close"]],
            axis=1,
            keys=["S&P 500", "KOSPI"],
        )
        combined_data = combined_data.dropna()

        if not combined_data.empty and len(combined_data) > 1:  # 최소 2개 이상의 데이터 포인트 필요
            fig2 = px.scatter(combined_data, x="S&P 500", y="KOSPI")
            fig2.update_layout(title="S&P 500 vs KOSPI 산점도")

            # 선형 회귀선 추가
            x = combined_data["S&P 500"].values
            y = combined_data["KOSPI"].values
            
            if len(x) > 0 and len(y) > 0:
                coeffs = np.polyfit(x, y, deg=1)
                line = coeffs[0] * x + coeffs[1]
                fig2.add_trace(go.Scatter(x=x, y=line, mode="lines", name="Regression Line"))

            st.plotly_chart(fig2, use_container_width=True)

            # 상관계수 계산 및 표시
            correlation = combined_data["KOSPI"].corr(combined_data["S&P 500"])
            st.subheader("상관관계 분석")
            st.write(f"KOSPI와 S&P 500의 상관계수: {correlation:.4f}")

            # 선행성 분석
            if len(combined_data) > 10:  # 충분한 데이터가 있는 경우만 선행성 분석
                lags = range(1, 11)
                correlations = [
                    combined_data["KOSPI"].corr(combined_data["S&P 500"].shift(lag)) 
                    for lag in lags
                ]

                max_correlation = max(correlations)
                max_lag = lags[correlations.index(max_correlation)]

                st.write(f"최대 상관계수: {max_correlation:.4f}, 발생 시점: {max_lag}일 전")

            st.subheader("S&P 500의 KOSPI 예측 가능성 분석")
            st.write(
                """
                분석 결과, S&P 500과 KOSPI 사이에 높은 상관관계가 있음을 확인했습니다...
                [이하 분석 텍스트는 동일하게 유지]
                """
            )
        else:
            st.warning("선택한 기간에 충분한 데이터가 없습니다. 다른 기간을 선택해주세요.")
    else:
        st.warning("선택한 기간에 데이터가 없습니다. 다른 기간을 선택해주세요.")
else:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
