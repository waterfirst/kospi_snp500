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
    end_date = datetime.now()
    start_date = datetime(2023, 1, 1)

    kospi = yf.download("^KS11", start=start_date, end=end_date)
    sp500 = yf.download("^AXJO", start=start_date, end=end_date)

    return kospi, sp500

# 데이터 로드
kospi, sp500 = get_data()

# 날짜 범위 선택
date_range = st.date_input(
    "날짜 범위를 선택하세요",
    value=(datetime(2023, 1, 1), datetime.now()),
    min_value=datetime(2023, 1, 1),
    max_value=datetime.now()
)

# 선택된 날짜로 데이터 필터링
start_date, end_date = date_range
kospi_filtered = kospi.loc[start_date:end_date]
sp500_filtered = sp500.loc[start_date:end_date]

# 이중 축 그래프 생성
fig1 = make_subplots(specs=[[{"secondary_y": True}]])

fig1.add_trace(
    go.Scatter(x=kospi_filtered.index, y=kospi_filtered['Close'], name="KOSPI", line=dict(color="blue")),
    secondary_y=False,
)

fig1.add_trace(
    go.Scatter(x=sp500_filtered.index, y=sp500_filtered['Close'], name="S&P 500", line=dict(color="red")),
    secondary_y=True,
)

fig1.update_layout(
    title="KOSPI와 S&P 500 지수 비교 (이중 축)",
    xaxis_title="날짜",
    legend_title="지수",
    hovermode="x unified"
)

fig1.update_yaxes(title_text="KOSPI", secondary_y=False)
fig1.update_yaxes(title_text="S&P 500", secondary_y=True)

st.plotly_chart(fig1, use_container_width=True)

# 상관관계 산점도
combined_data = pd.concat([sp500_filtered['Close'], kospi_filtered['Close']], axis=1, keys=['S&P 500', 'KOSPI'])
combined_data = combined_data.dropna()

fig2 = px.scatter(combined_data, x='S&P 500', y='KOSPI')
fig2.update_layout(title="S&P 500 vs KOSPI 산점도")

# 선형 회귀선 추가 (trendline="ols" 대신 수동으로 계산)
x = combined_data['S&P 500']
y = combined_data['KOSPI']
coeffs = np.polyfit(x, y, deg=1)
line = coeffs[0] * x + coeffs[1]
fig2.add_trace(go.Scatter(x=x, y=line, mode='lines', name='Regression Line'))

st.plotly_chart(fig2, use_container_width=True)

# 상관계수 계산 및 표시
correlation = combined_data['KOSPI'].corr(combined_data['S&P 500'])
st.subheader("상관관계 분석")
st.write(f"KOSPI와 S&P 500의 상관계수: {correlation:.4f}")

# 선행성 분석
lags = range(1, 11)  # 1일부터 10일까지의 지연 확인
correlations = [combined_data['KOSPI'].corr(combined_data['S&P 500'].shift(lag)) for lag in lags]

max_correlation = max(correlations)
max_lag = lags[correlations.index(max_correlation)]

st.write(f"최대 상관계수: {max_correlation:.4f}, 발생 시점: {max_lag}일 전")

st.subheader("S&P 500의 KOSPI 예측 가능성 분석")
st.write("""
분석 결과, S&P 500과 KOSPI 사이에 높은 상관관계가 있음을 확인했습니다. 이는 두 지수가 비슷한 패턴으로 움직이는 경향이 있다는 것을 의미합니다.

S&P 500이 KOSPI를 선행하는지 여부를 판단하기 위해, 우리는 다양한 시차(lag)에 대한 상관관계를 분석했습니다. 최대 상관계수가 발생하는 시차를 확인함으로써 S&P 500이 KOSPI에 선행하는지, 그리고 얼마나 선행하는지 추정할 수 있습니다.

1. 만약 최대 상관계수가 0일 시차에서 발생한다면, 두 지수는 동시에 움직이는 경향이 있다고 볼 수 있습니다.
2. 만약 최대 상관계수가 양의 시차에서 발생한다면, S&P 500이 KOSPI를 선행한다고 볼 수 있습니다.
3. 반대로, 음의 시차에서 최대 상관계수가 발생한다면 KOSPI가 S&P 500을 선행한다고 볼 수 있습니다.

이 분석 결과를 바탕으로, S&P 500이 KOSPI를 예측하는 데 사용될 수 있는지 추론해볼 수 있습니다:

1. 선행성: S&P 500이 KOSPI를 선행하는 것으로 나타난다면, 이는 S&P 500의 움직임이 KOSPI의 미래 움직임을 예측하는 데 도움이 될 수 있음을 시사합니다.

2. 상관관계의 강도: 상관계수가 높을수록 두 지수 간의 관계가 강하다는 것을 의미하며, 이는 예측의 신뢰도를 높일 수 있습니다.

3. 시차의 일관성: 선행성이 일관되게 나타나는지 확인하는 것이 중요합니다. 시간에 따라 선행성이 변할 수 있으므로, 장기간의 데이터를 분석하여 패턴의 안정성을 확인해야 합니다.

4. 외부 요인: 두 지수 모두 글로벌 경제 상황, 정치적 사건, 산업 트렌드 등 다양한 외부 요인의 영향을 받습니다. 따라서 S&P 500만으로 KOSPI를 완벽하게 예측하는 것은 불가능합니다.

5. 예측 모델 개발: 단순한 상관관계를 넘어, 머신러닝 기법을 활용하여 더 복잡한 예측 모델을 개발할 수 있습니다. 이 경우 S&P 500 외에도 다른 관련 지표들을 함께 고려하여 예측 정확도를 높일 수 있습니다.

결론적으로, S&P 500은 KOSPI의 움직임을 예측하는 데 유용한 지표가 될 수 있습니다. 하지만 이를 실제 투자 결정에 활용하기 위해서는 더 심도 있는 분석과 다른 요인들의 고려가 필요합니다. 또한, 과거의 패턴이 미래에도 지속된다는 보장은 없으므로, 이러한 분석 결과를 참고하되 신중하게 접근해야 합니다.
""")

# 데이터 표시
col1, col2 = st.columns(2)

with col1:
    st.subheader("KOSPI 데이터")
    st.dataframe(kospi_filtered)

with col2:
    st.subheader("S&P 500 데이터")
    st.dataframe(sp500_filtered)
