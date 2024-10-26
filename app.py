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

        # 종목 변경: KOSPI 지수로 KS11.KS 사용, S&P 500 지수로 SPY 사용
        kospi = yf.download("^KS11", start=start_date, end=end_date)
        sp500 = yf.download("SPY", start=start_date, end=end_date)

        if kospi.empty or sp500.empty:
            st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
            return None, None

        # 결측치 처리
        kospi = kospi.dropna()
        sp500 = sp500.dropna()

        return kospi, sp500
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")
        return None, None


# 데이터 로드
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

        # date_range가 단일 날짜인 경우 처리
        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date = end_date = date_range

        # 인덱스 필터링을 위한 날짜 변환
        start_dt = pd.to_datetime(start_date)
        end_dt = (
            pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        )

        # 데이터 필터링
        kospi_filtered = kospi[start_dt:end_dt].copy()
        sp500_filtered = sp500[start_dt:end_dt].copy()

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

            # 날짜 인덱스 맞추기
            kospi_daily = kospi_filtered.resample("D").last()
            sp500_daily = sp500_filtered.resample("D").last()

            common_dates = kospi_daily.index.intersection(sp500_daily.index)

            if len(common_dates) > 1:
                # 상관관계 분석을 위한 데이터 준비
                analysis_data = pd.DataFrame(
                    {
                        "KOSPI": kospi_daily.loc[common_dates, "Close"],
                        "S&P 500": sp500_daily.loc[common_dates, "Close"],
                    }
                )

                analysis_data = analysis_data.dropna()

                if len(analysis_data) > 1:
                    # 산점도 생성
                    fig2 = px.scatter(analysis_data, x="S&P 500", y="KOSPI")
                    fig2.update_layout(title="S&P 500 vs KOSPI 산점도")

                    # 회귀선 추가
                    x = analysis_data["S&P 500"].values
                    y = analysis_data["KOSPI"].values

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
                        max_lag = 10
                        lag_correlations = []

                        for lag in range(max_lag + 1):
                            lagged_data = analysis_data.copy()
                            lagged_data["S&P 500"] = lagged_data["S&P 500"].shift(lag)
                            corr = lagged_data["KOSPI"].corr(lagged_data["S&P 500"])
                            lag_correlations.append((lag, corr))

                        # NaN이 아닌 값들 중에서 최대 상관계수 찾기
                        valid_correlations = [
                            (lag, corr)
                            for lag, corr in lag_correlations
                            if not pd.isna(corr)
                        ]
                        if valid_correlations:
                            max_corr_lag, max_corr = max(
                                valid_correlations, key=lambda x: abs(x[1])
                            )
                            st.write(
                                f"최대 상관계수: {max_corr:.4f}, 발생 시점: {max_corr_lag}일 전"
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
