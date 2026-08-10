"""
传播动力学模型 — SIR 变体模拟舆情扩散

将沙盘推演中的智能体权重映射为 SIR 模型参数，
生成量化传播预测（感染曲线、峰值、基本再生数）。

为何是 SIR 而非 Agent-Based Model:
- 参数少、可解释、结果可复现
- 足够区分"系统性预测"和"LLM 的泛泛而谈"
- 不要求精准人口模拟，只需要相对趋势对比（A文案 vs B文案）
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ------------------------------------------------------------------
# 模型参数映射
# ------------------------------------------------------------------

@dataclass
class ModelParams:
    """SIR 模型参数"""
    population: float      # 总人口（归一化为 100 万）
    beta: float            # 传播率（感染→易感的接触传染概率）
    gamma: float           # 恢复率（感染→恢复的概率）
    i0: float              # 初始感染人数
    beta_spike: float      # 催化剂注入后的传播率（Round 2+）
    spike_time: int        # 催化剂注入时间步

    def effective_beta(self, t: int) -> float:
        return self.beta_spike if t >= self.spike_time else self.beta


def map_agents_to_model_params(
    agents: list[dict],
    population: float = 1_000_000.0,
    rounds: int = 4,
    catalyst_round: int = 2,
) -> ModelParams:
    """
    将沙盘智能体列表映射为 SIR 模型参数。

    映射逻辑：
    - I₀：hostile 权重总和在总权重中的占比 × 基础种子数
    - β：hostile 影响力占比（愤怒比中立传播更快）
    - γ：supportive 占比作为"免疫因子" + 自然遗忘率
    - β_spike：催化剂注入后 β 大幅提升
    """
    if not agents:
        return ModelParams(
            population=population, beta=0.15, gamma=0.05,
            i0=100, beta_spike=0.35, spike_time=catalyst_round,
        )

    total_weight = sum(a.get("weight", 300) for a in agents)
    hostile_weight = sum(a.get("weight", 0) for a in agents if a.get("stance") == "hostile")
    supportive_weight = sum(a.get("weight", 0) for a in agents if a.get("stance") == "supportive")
    neutral_weight = total_weight - hostile_weight - supportive_weight

    h_frac = hostile_weight / max(total_weight, 1)
    s_frac = supportive_weight / max(total_weight, 1)

    # 基础传播率：核心来自 hostile 占比，neutral 少量贡献
    base_beta = 0.12 + 0.28 * h_frac + 0.06 * (neutral_weight / max(total_weight, 1))
    base_beta = min(base_beta, 0.45)

    # 催化剂注入后的增强传播率
    spike_beta = base_beta * 2.0 + 0.10
    spike_beta = min(spike_beta, 0.65)

    # 恢复率：supportive 占比提供免疫力，加上自然遗忘
    base_gamma = 0.04 + 0.14 * s_frac
    base_gamma = min(base_gamma, 0.25)

    # 初始感染人数：hostile 权重映射到人群中的种子
    i0 = max(50, h_frac * 5000 + 200)
    i0 = min(i0, population * 0.01)

    return ModelParams(
        population=population,
        beta=round(base_beta, 4),
        gamma=round(base_gamma, 4),
        i0=int(i0),
        beta_spike=round(spike_beta, 4),
        spike_time=catalyst_round,
    )


# ------------------------------------------------------------------
# SIR 模拟器
# ------------------------------------------------------------------

@dataclass
class SpreadResult:
    """传播模拟结果"""
    peak_infected: int           # 峰值感染人数
    peak_time: int               # 峰值时间步
    total_ever_infected: int     # 累计感染（触达）人数
    r0: float                    # 基本再生数（β/γ）
    s_curve: list[float]         # 易感人群曲线
    i_curve: list[float]         # 感染人群曲线
    r_curve: list[float]         # 康复人群曲线
    time_steps: list[int]        # 时间步
    sentiment_breakdown: dict    # 情感构成（基于 agent 占比估算）

    def __post_init__(self):
        if not self.sentiment_breakdown:
            self.sentiment_breakdown = {}


def simulate(
    params: ModelParams,
    rounds: int = 4,
    steps_per_round: int = 3,
) -> SpreadResult:
    """
    运行 SIR 模型模拟。

    参数:
        params: 模型参数
        rounds: 沙盘轮次数
        steps_per_round: 每轮内部细分步数（用于画平滑曲线）

    返回:
        SpreadResult 包含完整传播曲线和关键指标
    """
    total_steps = rounds * steps_per_round
    dt = 1.0 / steps_per_round

    S = np.zeros(total_steps + 1)
    I = np.zeros(total_steps + 1)
    R = np.zeros(total_steps + 1)

    N = params.population
    I[0] = params.i0
    S[0] = N - I[0]
    R[0] = 0

    for t in range(total_steps):
        current_round = t // steps_per_round
        beta_t = params.effective_beta(current_round)

        dS = -beta_t * S[t] * I[t] / N * dt
        dR = params.gamma * I[t] * dt
        dI = -dS - dR

        S[t + 1] = max(0, S[t] + dS)
        I[t + 1] = max(0, I[t] + dI)
        R[t + 1] = max(0, R[t] + dR)

    peak_idx = int(np.argmax(I))
    peak_infected = int(I[peak_idx])
    peak_time = peak_idx
    total_ever_infected = int(I[0] + sum(max(0, S[i] - S[i + 1]) for i in range(total_steps)))

    r0 = params.beta / max(params.gamma, 0.001)

    # 情感构成：基于传播率推断愤怒/讽刺/防御的比例
    h_frac = (params.beta - 0.12) / max(0.28, 0.01)
    h_frac = max(0.2, min(0.85, h_frac))
    sentiment_breakdown = {
        "anger": round(h_frac * 0.55, 2),
        "mockery": round(h_frac * 0.25, 2),
        "defense": round((1 - h_frac) * 0.6, 2),
        "neutral_residual": round(1 - h_frac * 0.8 - (1 - h_frac) * 0.6, 2),
    }

    return SpreadResult(
        peak_infected=peak_infected,
        peak_time=peak_time,
        total_ever_infected=total_ever_infected,
        r0=round(r0, 2),
        s_curve=[round(v, 1) for v in S.tolist()],
        i_curve=[round(v, 1) for v in I.tolist()],
        r_curve=[round(v, 1) for v in R.tolist()],
        time_steps=list(range(total_steps + 1)),
        sentiment_breakdown=sentiment_breakdown,
    )


# ------------------------------------------------------------------
# 可视化（Streamlit）
# ------------------------------------------------------------------

def render_spread_chart(result: SpreadResult):
    """在 Streamlit 中渲染传播曲线图 + 关键指标卡片"""
    import streamlit as st

    # 指标卡片
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("峰值感染人数", f"{result.peak_infected:,}")
    with c2:
        st.metric("峰值时间步", f"t={result.peak_time}")
    with c3:
        st.metric("累计触达", f"{result.total_ever_infected:,}")
    with c4:
        st.metric("R₀ (基本再生数)", f"{result.r0}")
        if result.r0 > 1:
            st.caption(":red[R₀ > 1: 传播将持续扩大]")
        else:
            st.caption(":green[R₀ < 1: 传播将自行消退]")

    # 情感构成
    sb = result.sentiment_breakdown
    if sb:
        st.caption(
            f"情感构成预测 — :red[愤怒 {sb.get('anger',0):.0%}] "
            f":orange[讽刺 {sb.get('mockery',0):.0%}] "
            f":green[防御 {sb.get('defense',0):.0%}] "
            f":gray[中性/残余 {sb.get('neutral_residual',0):.0%}]"
        )

    # Matplotlib 图表
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(8, 3.5))
    t = result.time_steps
    ax.plot(t, result.s_curve, color="#3b82f6", linewidth=2, label="S 易感 (尚未接触)")
    ax.plot(t, result.i_curve, color="#ef4444", linewidth=2, label="I 感染 (正在传播)")
    ax.plot(t, result.r_curve, color="#22c55e", linewidth=2, label="R 康复 (热度消退)")

    # 标注峰值
    peak_t = result.peak_time
    peak_i = result.peak_infected
    ax.annotate(
        f"峰值: {peak_i:,} 人 (t={peak_t})",
        xy=(peak_t, peak_i),
        xytext=(peak_t + 1, peak_i * 1.15),
        arrowprops=dict(arrowstyle="->", color="#ef4444"),
        fontsize=10, color="#ef4444", fontweight="bold",
    )

    # 催化剂注入标记线
    catalyst_step = 2 * 3  # Round 2 × steps_per_round
    if catalyst_step < len(t):
        ax.axvline(x=catalyst_step, color="#f59e0b", linestyle="--", alpha=0.7, linewidth=1.2)
        ax.text(catalyst_step + 0.2, max(result.i_curve) * 0.75,
                "催化剂注入", color="#f59e0b", fontsize=9, fontweight="bold")

    ax.set_xlabel("时间步 (t)", fontsize=10)
    ax.set_ylabel("人数", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("SIR 舆情传播动力学模拟", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e4:.0f}万" if x >= 1e4 else f"{x:.0f}"))

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
