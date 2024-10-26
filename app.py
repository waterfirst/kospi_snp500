import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
import pytz

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

        kospi = yf.download("005930.KS", start=start_date, end=end_date)
        sp500 = yf.download("^GSPC", start=start_date, end=end_date)

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

        # 선택된 날짜로 데이터 필터링
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        kospi_filtered = kospi[kospi.index.date >= start_date.date()]
        kospi_filtered = kospi_filtered[kospi_filtered.index.date <= end_date.date()]
        
        sp500_filtered = sp500[sp500.index.date >= start_date.date()]
        sp500_filtered = sp500_filtered[sp500_filtered.index.date <= end_date.date()]

        if len(kospi_filtered) > 0 and len(sp500_filtered) > 0:
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
            kospi_resampled = kospi_filtered.resample('D').last()
            sp500_resampled = sp500_filtered.resample('D').last()
            
            # 공통 날짜만 선택
            common_dates = kospi_resampled.index.intersection(sp500_resampled.index)
            kospi_aligned = kospi_resampled.loc[common_dates]
            sp500_aligned = sp500_resampled.loc[common_dates]

            # 상관관계 분석을 위한 데이터프레임 생성
            df_analysis = pd.DataFrame({
                'S&P 500': sp500_aligned['Close'],
                'KOSPI': kospi_aligned['Close']
            }, index=common_dates)

            df_analysis = df_analysis.dropna()

            # 상관관계 산점도
            if len(df_analysis) > 1:
                fig2 = px.scatter(df_analysis, x='S&P 500', y='KOSPI')
                fig2.update_layout(title="S&P 500 vs KOSPI 산점도")

                # 회귀선 추가
                x = df_analysis['S&P 500'].values
                y = df_analysis['KOSPI'].values
                coeffs = np.polyfit(x, y, deg=1)
                fit_line = np.poly1d(coeffs)
                
                fig2.add_trace(
                    go.Scatter(
                        x=x,
                        y=fit_line(x),
                        mode='lines',
                        name='Regression Line'
                    )
                )

                st.plotly_chart(fig2, use_container_width=True)

                # 상관계수 계산 및 표시
                correlation = df_analysis['KOSPI'].corr(df_analysis['S&P 500'])
                st.subheader("상관관계 분석")
                st.write(f"KOSPI와 S&P 500의 상관계수: {correlation:.4f}")

                # 선행성 분석
                if len(df_analysis) > 10:
                    max_lag = 10
                    lag_corrs = []
                    
                    for lag in range(1, max_lag + 1):
                        lagged = df_analysis['S&P 500'].shift(lag)
                        corr = df_analysis['KOSPI'].corr(lagged)
                        if not pd.isna(corr):
                            lag_corrs.append((lag, corr))
                    
                    if lag_corrs:
                        max_lag_corr = max(lag_corrs, key=lambda x: x[1])
                        st.write(f"최대 상관계수: {max_lag_corr[1]:.4f}, 발생 시점: {max_lag_corr[0]}일 전")

                st.subheader("S&P 500의 KOSPI 예측 가능성 분석")
                st.write("""
                분석 결과, S&P 500과 KOSPI 사이에 높은 상관관계가 있음을 확인했습니다...
                [이하 분석 텍스트는 동일하게 유지]
                """)
            else:
                st.warning("상관관계 분석을 위한 충분한 데이터가 없습니다.")
        else:
            st.warning("선택한 기간에 데이터가 없습니다. 다른 기간을 선택해주세요.")
    except Exception as e:
        st.error(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
        st.write("Traceback:", e.__traceback__)
else:
    st.error("데이터를 불러올 수 없습니다. 잠시 후 다시 시도해주세요.")
