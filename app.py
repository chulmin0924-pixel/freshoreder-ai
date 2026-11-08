"""AI food inventory and auto-order dashboard, ready for Streamlit Cloud."""
from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import clean_demand_data, generate_demo_data, run_forecast, simulate_inventory

st.set_page_config(
    page_title="FreshOrder AI | 자동 발주 대시보드", page_icon="🥬",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.hero {padding: 1.45rem 1.6rem; border-radius: 18px;
       background: linear-gradient(110deg,#143c38 0%,#17685b 100%); color: white;
       margin-bottom: 1.15rem;}
.hero h1 {font-size: 2.05rem; margin: 0.1rem 0 0.45rem; color: white;}
.hero p {margin: 0; color: #e4f5ed;}
.small {font-size: 0.88rem; color: #73817d;}
</style>
""", unsafe_allow_html=True)

st.markdown('''<div class="hero"><div style="font-size:.88rem;font-weight:700;letter-spacing:.1em">FRESHORDER AI · PROJECT DEMO</div>
<h1>🥬 식재료 수요 예측 & 자동 발주</h1>
<p>인공신경망(MLP) 예측 · 유효 재고 기반 주문 추천 · FIFO 시뮬레이션 · 발주 비용 비교</p></div>''', unsafe_allow_html=True)
st.caption("※ 학습용 프로토타입입니다. 기본 자료는 실제 매장 기록이 아닌 합성 데이터이며, 실제 발주 결정에는 추가 검증이 필요합니다.")

st.sidebar.header("⚙️ 분석 설정")
st.sidebar.markdown("**① 데이터 선택**")
source = st.sidebar.radio("데이터", ["예제 데이터(90일)", "내 CSV 업로드"], label_visibility="collapsed")
if source == "예제 데이터(90일)":
    data = generate_demo_data()
    st.sidebar.caption("90일 합성 소비량 데이터 (kg/일)")
else:
    uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"], help="date, demand_kg 두 열 필요")
    if uploaded is None:
        st.info("왼쪽에서 CSV를 업로드하거나 예제 데이터를 선택해 주세요.")
        st.stop()
    try:
        content = uploaded.getvalue()
        try:
            raw = pd.read_csv(BytesIO(content), encoding="utf-8-sig")
        except UnicodeDecodeError:
            raw = pd.read_csv(BytesIO(content), encoding="cp949")
        data = clean_demand_data(raw)
    except (ValueError, pd.errors.ParserError, KeyError) as error:
        st.error(f"CSV 읽기에 실패했습니다: {error}")
        st.stop()

@st.cache_data(show_spinner=False)
def cached_forecast(df: pd.DataFrame):
    return run_forecast(df)

try:
    result = cached_forecast(data)
except (ValueError, RuntimeError) as error:
    st.error(f"AI 수요 예측 실패: {error}")
    st.stop()

st.sidebar.markdown("**② 발주 · 보관 조건**")
shelf_life = st.sidebar.slider("보관 가능 기간(일)", 1, 10, 3,
                              help="입고 당일을 포함해 사용 가능한 날짜 수입니다. 실제 식품 안전기준이 아닙니다.")
fixed_qty = st.sidebar.number_input("기존 매장 고정 발주량 (kg/일)", min_value=0, max_value=300,
                                   value=result.baseline_quantity, step=1)
safety_stock = st.sidebar.slider("AI 안전재고 목표(kg)", min_value=0, max_value=30, value=0)

st.sidebar.markdown("**③ 비용 가정**")
purchase_cost = st.sidebar.number_input("식재료 구매 단가(원/kg)", 0, 100000, 4500, 100)
stockout_cost = st.sidebar.number_input("품절 손실 단가(원/kg)", 0, 100000, 6500, 100)
waste_cost = st.sidebar.number_input("폐기 단가(원/kg)", 0, 100000, 1200, 100)

with st.spinner("MLP 모델과 FIFO 발주 시뮬레이션 계산 중..."):
    fixed_daily, fixed = simulate_inventory(
        result.eval_frame, "fixed", shelf_life, fixed_qty, safety_stock,
        purchase_cost, stockout_cost, waste_cost,
    )
    ai_daily, ai = simulate_inventory(
        result.eval_frame, "ai", shelf_life, fixed_qty, safety_stock,
        purchase_cost, stockout_cost, waste_cost,
    )

cost_delta = fixed["total_cost"] - ai["total_cost"]
cost_delta_pct = cost_delta / fixed["total_cost"] * 100 if fixed["total_cost"] else 0

st.subheader("오늘의 의사결정")
st.caption(f"데이터 마지막 일자: {result.history['date'].max():%Y-%m-%d} · 추천 대상: {result.next_date:%Y-%m-%d} · 단위: kg")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("다음 날 AI 예측 수요", f"{result.next_prediction:,.1f} kg")
with c2:
    st.metric("평가 기간 예측 오차(MAE)", f"{result.mae:,.2f} kg", help="최근 20일에 대해 실제 소비량과 예측량의 절댓값 차이 평균")
with c3:
    st.metric("20일 시뮬레이션 비용 차이 (고정−AI)", f"{cost_delta:,.0f} 원",
              delta=(f"{abs(cost_delta_pct):.1f}% 비용 절감" if cost_delta >= 0
                     else f"{abs(cost_delta_pct):.1f}% 비용 증가") if fixed['total_cost'] else "비교 비용 없음",
              delta_color="normal" if cost_delta >= 0 else "inverse")

st.markdown("#### 🛒 다음 날 발주 추천")
left, right = st.columns([1.2, 1.8])
with left:
    available_stock = st.number_input("다음 날 사용 가능한 재고량(kg)", min_value=0.0,
                                      max_value=10000.0, value=12.0, step=1.0,
                                      help="소비기한 이전에 사용할 수 있는 재고만 입력합니다.")
    expiring_stock = st.number_input("다음 날 전에 만료될 재고(kg, 제외)", min_value=0.0,
                                     max_value=10000.0, value=0.0, step=1.0,
                                     help="참고용 표시 항목이며 발주 계산에 포함되지 않습니다.")
    recommended = max(0, int(np.ceil(result.next_prediction + safety_stock - available_stock)))
with right:
    st.success(f"### 추천 발주량: {recommended:,} kg")
    st.write(f"예상 수요 **{result.next_prediction:.1f}kg** + 안전재고 **{safety_stock}kg** − 유효 재고 **{available_stock:.1f}kg**")
    if expiring_stock:
        st.warning(f"다음 날 이전 만료 재고 {expiring_stock:.1f}kg는 추천 발주량 계산에서 제외했습니다.")
    st.caption("판매기록을 기반으로 계산한 제안 수량입니다. 납기·최소주문량·재료별 보관조건은 별도 검토해야 합니다.")

forecast_tab, cost_tab, details_tab = st.tabs(["📈 AI 수요 예측", "📊 발주 방식 비교", "🗂 데이터 · 모델 설명"])

with forecast_tab:
    st.subheader("최근 20일 실제 수요 vs AI 예측")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.eval_frame["date"], y=result.eval_frame["demand_kg"],
                             name="실제 소비량", mode="lines+markers", line=dict(color="#0c7a61", width=3)))
    fig.add_trace(go.Scatter(x=result.eval_frame["date"], y=result.eval_frame["predicted_kg"],
                             name="MLP 예측값", mode="lines+markers", line=dict(color="#e69d37", width=2, dash="dash")))
    fig.update_layout(height=380, margin=dict(l=5,r=5,t=15,b=5),
                      yaxis_title="식재료 소비량 (kg)", xaxis_title="평가 기간", legend=dict(orientation="h",y=1.12))
    st.plotly_chart(fig, use_container_width=True)
    st.info(f"학습: {result.training_rows}개 유효 일자 · 평가: 최근 {len(result.eval_frame)}일 · MLP 은닉층 (12, 8) · 평가 MAE {result.mae:.2f}kg")

with cost_tab:
    st.subheader("동일한 20일 수요에 대한 발주 정책 비교")
    table = pd.DataFrame({
        "항목": ["총 발주량 (kg)", "폐기량 (kg)", "품절량 (kg)", "최종 잔여재고 (kg)", "총비용 (원)"],
        "고정 발주": [fixed['ordered_kg'], fixed['waste_kg'], fixed['shortage_kg'],
                     fixed['ending_inventory_kg'], fixed['total_cost']],
        "AI 발주": [ai['ordered_kg'], ai['waste_kg'], ai['shortage_kg'],
                   ai['ending_inventory_kg'], ai['total_cost']],
    })
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"고정 발주": st.column_config.NumberColumn(format="%.1f"),
                                "AI 발주": st.column_config.NumberColumn(format="%.1f")})
    comp1, comp2 = st.columns(2)
    with comp1:
        fig_cost = go.Figure(go.Bar(x=["고정 발주", "AI 발주"],
                                    y=[fixed['total_cost'], ai['total_cost']],
                                    marker_color=["#aab3b1", "#0c7a61"],
                                    text=[f"{fixed['total_cost']:,.0f}", f"{ai['total_cost']:,.0f}"],
                                    textposition="auto"))
        fig_cost.update_layout(title="총비용 비교 (원)", height=320, margin=dict(l=5,r=5,t=45,b=5))
        st.plotly_chart(fig_cost, use_container_width=True)
    with comp2:
        f = go.Figure()
        f.add_trace(go.Bar(name="고정 발주 품절", x=fixed_daily["date"], y=fixed_daily["shortage_kg"],marker_color="#aab3b1"))
        f.add_trace(go.Bar(name="AI 발주 품절", x=ai_daily["date"], y=ai_daily["shortage_kg"],marker_color="#0c7a61"))
        f.update_layout(title="일별 품절량 (kg)", barmode="group", height=320,
                        margin=dict(l=5,r=5,t=45,b=5),legend=dict(orientation="h",y=1.17))
        st.plotly_chart(f, use_container_width=True)

    daily_comparison = ai_daily.copy()
    daily_comparison = daily_comparison.rename(columns={
        "date": "날짜", "actual_kg": "실제 수요(kg)", "predicted_kg": "예측 수요(kg)",
        "ordered_kg": "AI 발주량(kg)", "expired_kg": "AI 폐기량(kg)",
        "shortage_kg": "AI 품절량(kg)", "remaining_kg": "AI 잔여재고(kg)", "cost_krw": "AI 당일 비용(원)"})
    daily_comparison["고정 발주량(kg)"] = fixed_daily["ordered_kg"].values
    daily_comparison["고정 품절량(kg)"] = fixed_daily["shortage_kg"].values
    st.download_button("⬇️ 20일 분석 결과 CSV 다운로드", daily_comparison.to_csv(index=False).encode("utf-8-sig"),
                       file_name="ai_order_comparison.csv", mime="text/csv")
    st.caption("시뮬레이션 가정: 초기 재고 0kg · 당일 입고 · FIFO 소비 · 보관 가능 기간 초과 시 폐기. 총비용=구매량×구매 단가 + 품절량×손실 단가 + 폐기량×폐기 단가. 기말 재고는 별도 표시합니다.")

with details_tab:
    st.subheader("서비스 처리 구조")
    st.markdown("**과거 소비량 → 7일 평균·요일 등 특징 생성 → MLP 예측 → 유효 재고 확인 → 발주 추천 → 비용·품절·폐기 비교**")
    st.markdown("**AI 모델**: scikit-learn `MLPRegressor`, 은닉층 12개·8개 뉴런, 시간순 학습/평가 분리. 평가 기간 예측에는 당일까지 관측 가능한 과거 소비량만 입력합니다.")
    st.markdown("**데이터 형식**: 날짜(`date`) 및 하루 식재료 소비량(`demand_kg`) 두 열, 최소 40일 연속 관측. CSV에는 헤더 행이 필요합니다.")
    st.dataframe(result.history.tail(12), hide_index=True, use_container_width=True)
    st.download_button("⬇️ 입력 데이터 CSV 다운로드", result.history.to_csv(index=False).encode("utf-8-sig"),
                       file_name="sample_demand.csv", mime="text/csv")
    st.caption("논문과 동일한 연구 아이디어 및 주요 실험 조건에 기반한 재구현입니다. 합성 난수와 학습 조건이 달라 논문의 기존 수치가 정확히 재현된다는 의미는 아닙니다.")

st.divider()
st.caption("FreshOrder AI · 인공지능 과목 Streamlit 대시보드 과제 · 교육용 실험 프로젝트")
