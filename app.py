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
@st.cache_data(ttl=3600, show_spinner=True)
def fetch_data(ticker, start_date, end_date, retries=3):
    for attempt in range(retries):
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                return data
        except Exception as e:
            if attempt == retries - 1:
                st.error(f"{ticker} 데이터 다운로드 실패: {str(e)}")
                return None
            continue
    return None


def get_data():
    try:
        end_date = datetime.now()
        start_date = datetime(2023, 1, 1)

        # KOSPI 데이터 시도
        kospi = fetch_data("^KS11", start_date, end_date)
        if kospi is None or kospi.empty:
            kospi = fetch_data("EWY", start_date, end_date)
            if kospi is None or kospi.empty:
                st.error("KOSPI 데이터를 불러올 수 없습니다.")
                return None, None

        # S&P 500 데이터 시도
        sp500 = fetch_data("SPY", start_date, end_date)
        if sp500 is None or sp500.empty:
            sp500 = fetch_data("^GSPC", start_date, end_date)
            if sp500 is None or sp500.empty:
                st.error("S&P 500 데이터를 불러올 수 없습니다.")
                return None, None

        # 결측치 처리
        kospi = kospi.dropna()
        sp500 = sp500.dropna()

        # Close 컬럼이 있는지 확인
        if "Close" not in kospi.columns or "Close" not in sp500.columns:
            st.error("필요한 데이터 컬럼이 없습니다.")
            return None, None

        # 기준일 대비 변화율로 정규화
        kospi_norm = kospi.copy()
        sp500_norm = sp500.copy()

        if not kospi_norm.empty and not sp500_norm.empty:
            # 첫 날 종가 확인
            kospi_start = kospi_norm["Close"].iloc[0]
            sp500_start = sp500_norm["Close"].iloc[0]

            if kospi_start > 0 and sp500_start > 0:
                kospi_norm["Close"] = (kospi_norm["Close"] / kospi_start) * 100
                sp500_norm["Close"] = (sp500_norm["Close"] / sp500_start) * 100
            else:
                st.error("유효하지 않은 기준가격이 있습니다.")
                return None, None

        return kospi_norm, sp500_norm

    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")
        return None, None


# 데이터 로드
with st.spinner("데이터를 불러오는 중입니다..."):
    kospi, sp500 = get_data()

if kospi is not None and sp500 is not None and not kospi.empty and not sp500.empty:
    try:
        # 날짜 범위 선택
        min_date = kospi.index.min().date()
        max_date = kospi.index.max().date()

        date_range = st.date_input(
            "날짜 범위를 선택하세요",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        # date_range 처리
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        # 날짜 필터링을 위한 문자열 변환
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

        # 데이터 필터링
        kospi_filtered = kospi[start_str:end_str].copy()
        sp500_filtered = sp500[start_str:end_str].copy()

        if len(kospi_filtered) > 0 and len(sp500_filtered) > 0:
            # 그래프 생성
            fig1 = go.Figure()

            fig1.add_trace(
                go.Scatter(
                    x=kospi_filtered.index,
                    y=kospi_filtered["Close"],
                    name="KOSPI",
                    line=dict(color="blue"),
                )
            )

            fig1.add_trace(
                go.Scatter(
                    x=sp500_filtered.index,
                    y=sp500_filtered["Close"],
                    name="S&P 500",
                    line=dict(color="red"),
                )
            )

            fig1.update_layout(
                title="KOSPI와 S&P 500 지수 비교 (기준일 대비 %)",
                xaxis_title="날짜",
                yaxis_title="기준일 대비 변화율 (%)",
                legend_title="지수",
                hovermode="x unified",
            )

            st.plotly_chart(fig1, use_container_width=True)

            # 상관관계 분석
            common_dates = kospi_filtered.index.intersection(sp500_filtered.index)

            if len(common_dates) > 1:
                analysis_data = pd.DataFrame(
                    {
                        "KOSPI": kospi_filtered.loc[common_dates, "Close"],
                        "S&P 500": sp500_filtered.loc[common_dates, "Close"],
                    }
                )

                analysis_data = analysis_data.dropna()

                if len(analysis_data) > 1:
                    # 산점도 생성
                    fig2 = px.scatter(analysis_data, x="S&P 500", y="KOSPI")
                    fig2.update_layout(
                        title="S&P 500 vs KOSPI 산점도 (기준일 대비 %)",
                        xaxis_title="S&P 500 변화율 (%)",
                        yaxis_title="KOSPI 변화율 (%)",
                    )

                    # 회귀선 추가
                    x = analysis_data["S&P 500"].values
                    y = analysis_data["KOSPI"].values

                    if len(x) > 1 and len(y) > 1:
                        coeffs = np.polyfit(x, y, deg=1)
                        line_x = np.array([x.min(), x.max()])
                        line_y = coeffs[0] * line_x + coeffs[1]

                        fig2.add_trace(
                            go.Scatter(
                                x=line_x,
                                y=line_y,
                                mode="lines",
                                name="Regression Line",
                                line=dict(color="red"),
                            )
                        )

                    st.plotly_chart(fig2, use_container_width=True)

                    # 상관계수 계산 및 표시
                    correlation = analysis_data["KOSPI"].corr(analysis_data["S&P 500"])
                    st.subheader("상관관계 분석")
                    st.write(f"KOSPI와 S&P 500의 상관계수: {correlation:.4f}")

                    # 선행성 분석
                    if len(analysis_data) > 10:
                        max_lag = min(10, len(analysis_data) - 1)
                        lag_correlations = []

                        for lag in range(max_lag + 1):
                            lagged_data = analysis_data.copy()
                            lagged_data["S&P 500"] = lagged_data["S&P 500"].shift(lag)
                            corr = lagged_data["KOSPI"].corr(lagged_data["S&P 500"])
                            if not pd.isna(corr):
                                lag_correlations.append((lag, corr))

                        if lag_correlations:
                            max_corr_info = max(
                                lag_correlations, key=lambda x: abs(x[1])
                            )
                            st.write(
                                f"최대 상관계수: {max_corr_info[1]:.4f}, 발생 시점: {max_corr_info[0]}일 전"
                            )

                else:
                    st.warning("상관관계 분석을 위한 충분한 데이터가 없습니다.")
            else:
                st.warning("선택한 기간에 공통된 거래일이 충분하지 않습니다.")
        else:
            st.warning("선택한 기간에 데이터가 없습니다. 다른 기간을 선택해주세요.")
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        st.write("Error details:", str(e))
else:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
