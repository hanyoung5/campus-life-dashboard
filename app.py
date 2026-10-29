import os
import platform
import re
from collections import Counter

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from wordcloud import WordCloud

# =========================================================
# 0. 페이지 기본 설정 & OS별 한글 폰트 설정
# =========================================================
st.set_page_config(
    page_title="캠퍼스 라이프 설문 분석 대시보드",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def set_korean_font():
    """운영체제별 한글 폰트 감지 및 Matplotlib 전역 설정"""
    sys_name = platform.system()
    if sys_name == "Windows":
        font_path = "c:/Windows/Fonts/malgun.ttf"
        font_family = "Malgun Gothic"
    elif sys_name == "Darwin":  # macOS
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
        font_family = "AppleGothic"
    else:  # Linux (Ubuntu / Docker 등)
        font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
        font_family = "NanumGothic"

    if os.path.exists(font_path):
        plt.rc("font", family=font_family)
    plt.rcParams["axes.unicode_minus"] = False  # 마이너스 부호 깨짐 방지
    return font_path


font_file_path = set_korean_font()


# =========================================================
# 1. 데이터 로드 및 결측치/이상치 자동 정제
# =========================================================
@st.cache_data
def load_and_preprocess_data(csv_path="campus_life.csv"):
    # 1) UTF-8 BOM 인코딩 대응
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 2) 만족도 컬럼의 비정상 값('모름') -> 결측치 변환 후 수치형 변환
    df["만족도"] = pd.to_numeric(df["만족도"].replace("모름", np.nan))

    # 만족도 결측치를 해당 학과의 중앙값으로 보정
    df["만족도"] = df.groupby("학과")["만족도"].transform(
        lambda x: x.fillna(x.median())
    )

    # 3) 공부시간 결측치를 해당 학과의 평균값으로 대치
    df["공부시간"] = df.groupby("학과")["공부시간"].transform(
        lambda x: x.fillna(round(x.mean(), 1))
    )

    # 4) 컬럼명 단위 명확화
    df = df.rename(
        columns={
            "통학시간": "통학시간(분)",
            "공부시간": "공부시간(시간)",
            "스마트폰": "스마트폰(시간)",
            "수면시간": "수면시간(시간)",
        }
    )

    return df


try:
    df_cleaned = load_and_preprocess_data("campus_life.csv")
except Exception as e:
    st.error(
        f"🚨 데이터를 불러오지 못했습니다. `campus_life.csv` 파일이 동일 폴더에 있는지 확인해주세요. ({e})"
    )
    st.stop()


# =========================================================
# 2. [인터페이스 1] 사이드바 (Sidebar) - 조작 및 필터
# =========================================================
with st.sidebar:
    st.header("🎛️ 필터 및 분석 제어")
    st.markdown("조회하고자 하는 학과를 선택하면 모든 지표가 즉시 갱신됩니다.")

    # 학과 선택 기능
    dept_options = ["전체 학과"] + sorted(list(df_cleaned["학과"].unique()))
    selected_dept = st.selectbox("📌 학과 선택", dept_options)

    # 학년 선택 기능
    available_grades = sorted(list(df_cleaned["학년"].unique()))
    selected_grades = st.multiselect(
        "🎓 학년 선택 (다중 선택 가능)",
        available_grades,
        default=available_grades,
    )

    st.markdown("---")
    st.subheader("🛠️ 데이터 정제 내역 요약")
    st.info(
        """
    - **S017:** 만족도 `'모름'` ➡️ 학과 중앙값 보정
    - **S042:** 공부시간 결측(`NaN`) ➡️ 학과 평균 보정
    - **인코딩:** UTF-8 BOM 헤더 정규화
    """
    )
    st.caption("총학생회 정책기획국 데이터분석팀")

# 필터링 적용
filtered_df = df_cleaned[df_cleaned["학년"].isin(selected_grades)]
if selected_dept != "전체 학과":
    filtered_df = filtered_df[filtered_df["학과"] == selected_dept]

if filtered_df.empty:
    st.warning("선택한 조건에 부합하는 설문 데이터가 없습니다.")
    st.stop()


# =========================================================
# 3. [인터페이스 2] 상단 KPI 요약 카드 (Metrics Cards)
# =========================================================
st.title("🎓 캠퍼스 라이프 설문조사 종합 대시보드")
st.markdown(
    f"**현재 조회 대상:** `{selected_dept}` | 학우들의 실제 생활 데이터를 바탕으로 실효성 있는 정책을 도출합니다."
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("총 응답 학우", f"{len(filtered_df)}명")
with kpi2:
    st.metric(
        "평균 만족도",
        f"{filtered_df['만족도'].mean():.1f}점",
        delta=f"{(filtered_df['만족도'].mean() - df_cleaned['만족도'].mean()):+.2f}점 vs 전체",
    )
with kpi3:
    st.metric(
        "평균 통학시간",
        f"{filtered_df['통학시간(분)'].mean():.0f}분",
        delta=f"{(filtered_df['통학시간(분)'].mean() - df_cleaned['통학시간(분)'].mean()):+.1f}분 vs 전체",
        delta_color="inverse",
    )
with kpi4:
    st.metric("일일 공부시간", f"{filtered_df['공부시간(시간)'].mean():.1f}시간")
with kpi5:
    st.metric("일일 수면시간", f"{filtered_df['수면시간(시간)'].mean():.1f}시간")

st.markdown("---")


# =========================================================
# 4. [인터페이스 3] 메인 분석 영역 (2단 분할 그리드)
#    - [좌측] 만족도 산점도
#    - [우측] 워드클라우드 & 상위 10개 단어 빈도 (+,-) 막대그래프
# =========================================================
grid_left, grid_right = st.columns([1, 1], gap="medium")

# ---------------------------------------------------------
# [출력 1] 여러 시간과 학교생활 만족도의 관계 (산점도)
# ---------------------------------------------------------
with grid_left:
    st.subheader(f"📊 [{selected_dept}] 생활 시간과 만족도 산점도")

    x_axis_choice = st.selectbox(
        "만족도와 비교할 생활 시간 지표를 선택하세요:",
        ["공부시간(시간)", "수면시간(시간)", "스마트폰(시간)", "통학시간(분)"],
        index=0,
    )

    fig_scatter, ax_sc = plt.subplots(figsize=(6.5, 5.2))
    sns.scatterplot(
        data=filtered_df,
        x=x_axis_choice,
        y="만족도",
        hue="학년",
        size="수면시간(시간)" if x_axis_choice != "수면시간(시간)" else "공부시간(시간)",
        sizes=(50, 200),
        palette="Set2",
        alpha=0.85,
        ax=ax_sc,
    )

    if len(filtered_df) > 2:
        sns.regplot(
            data=filtered_df,
            x=x_axis_choice,
            y="만족도",
            scatter=False,
            ax=ax_sc,
            color="gray",
            line_kws={"linestyle": "--", "linewidth": 1.2},
        )

    ax_sc.set_title(f"{selected_dept}: {x_axis_choice} 대비 만족도 상관 분포", fontsize=12)
    ax_sc.set_ylabel("캠퍼스 생활 만족도 (1~10)", fontsize=10)
    ax_sc.set_ylim(1, 10.5)
    ax_sc.grid(True, linestyle=":", alpha=0.6)
    ax_sc.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    st.pyplot(fig_scatter)
    st.caption("💡 점의 색상은 학년, 크기는 보조 활동 시간이며, 점선은 경향성 추세선입니다.")

# ---------------------------------------------------------
# [출력 2] 학생들 의견 단어 빈도수 분석 (워드클라우드 + 막대그래프)
# ---------------------------------------------------------
with grid_right:
    st.subheader(f"💬 [{selected_dept}] 학우 의견 단어 빈도 분석")

    comments_list = filtered_df["한줄의견"].dropna().tolist()
    all_comments = " ".join(comments_list)

    # 1) 워드클라우드 시각화
    stopwords = set([
        "시간이", "너무", "기간에", "오래", "봐서", "돼요", "있어요", "대한"
    ])

    if all_comments.strip():
        try:
            wc = WordCloud(
                font_path=font_file_path if os.path.exists(font_file_path) else None,
                width=650,
                height=320,
                background_color="white",
                colormap="Dark2",
                stopwords=stopwords,
                min_font_size=10,
            ).generate(all_comments)

            fig_wc, ax_wc = plt.subplots(figsize=(6.5, 3.2))
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)
        except Exception as err:
            st.warning(f"워드클라우드 생성 중 오류: {err}")

        # -----------------------------------------------------
        # [신규 기능] 상위 10개 단어 빈도수 및 (+, -) 감성 막대그래프
        # -----------------------------------------------------
        # 단어 토큰화 및 빈도 계산
        tokens = re.findall(r"[가-힣]{2,}", all_comments)
        filtered_tokens = [w for w in tokens if w not in stopwords]
        word_counts = Counter(filtered_tokens).most_common(10)

        if word_counts:
            # 긍정(+) 및 부정(-) 감성 분류 사전
            pos_keywords = {"좋아요", "재미있어요", "맛있어요", "친절해요", "도움이", "예뻐서", "편리해요", "즐거워요"}
            neg_keywords = {"피곤해요", "부족해요", "힘들어요", "아쉬워요", "불편해요", "후회돼요", "비싸서", "멀어서", "길어서", "붐벼서"}

            words, freqs, colors, labels = [], [], [], []
            for word, count in word_counts:
                words.append(word)
                freqs.append(count)
                if word in pos_keywords:
                    colors.append("#2b7bba")  # 긍정: 푸른색
                    labels.append(f"{word} (+)")
                elif word in neg_keywords:
                    colors.append("#d9534f")  # 부정: 붉은색
                    labels.append(f"{word} (-)")
                else:
                    colors.append("#6c757d")  # 중립: 회색
                    labels.append(f"{word} (중립)")

            # 수평 막대그래프 그리기 (상위 항목이 위로 오도록 역순 정렬)
            fig_bar, ax_bar = plt.subplots(figsize=(6.5, 3.8))
            y_pos = range(len(words))[::-1]
            bars = ax_bar.barh(y_pos, freqs, color=colors, height=0.65, edgecolor="none")

            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(labels, fontsize=10)
            ax_bar.set_xlabel("등장 횟수 (건)", fontsize=10)
            ax_bar.set_title(f"[{selected_dept}] 한줄의견 빈출 단어 TOP 10 (긍정 / 부정 구분)", fontsize=11)
            ax_bar.grid(axis="x", linestyle="--", alpha=0.5)

            # 막대 우측 텍스트 표기
            for bar, count in zip(bars, freqs):
                ax_bar.text(
                    bar.get_width() + 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count}회",
                    ha="left", va="center",
                    fontsize=9, weight="bold", color="#333333"
                )

            # 범례 추가
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#2b7bba", label="긍정(+) 단어"),
                Patch(facecolor="#d9534f", label="부정(-) 단어"),
                Patch(facecolor="#6c757d", label="기타/중립 단어"),
            ]
            ax_bar.legend(handles=legend_elements, loc="lower right", fontsize=8.5)

            plt.tight_layout()
            st.pyplot(fig_bar)
            st.caption("💡 (+) 긍정적 피드백 / (-) 애로사항 및 불만 건의 단어")
    else:
        st.write("표시할 의견 텍스트가 없습니다.")

st.markdown("---")


# =========================================================
# 5. [인터페이스 4] 비교 및 종합 영역 (탭 인터페이스)
#    - [기능] 학과별 생활습관 차이 분석 (히트맵)
# =========================================================
st.subheader("📑 비교 및 종합 분석")
tab1, tab2 = st.tabs(["🔥 학과별 생활 습관 히트맵", "📈 주요 지표별 단과대 순위"])

metric_cols = ["통학시간(분)", "공부시간(시간)", "스마트폰(시간)", "수면시간(시간)", "만족도"]
dept_group_mean = df_cleaned.groupby("학과")[metric_cols].mean()

with tab1:
    st.markdown("#### 📌 단과대/학과별 라이프스타일 패턴 비교")
    view_type = st.radio(
        "히트맵 표시 방식 선택:",
        ["실제 평균값 히트맵", "표준화(Z-Score) 상대 편차 히트맵 (과별 특성 부각)"],
        horizontal=True,
    )

    fig_hm, ax_hm = plt.subplots(figsize=(10, 4.5))

    if view_type == "실제 평균값 히트맵":
        sns.heatmap(
            dept_group_mean,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            linewidths=1,
            cbar_kws={"label": "평균값"},
            ax=ax_hm,
        )
        ax_hm.set_title("학과별 항목별 평균 수치", fontsize=13)
    else:
        dept_group_zscore = (dept_group_mean - dept_group_mean.mean()) / dept_group_mean.std()
        sns.heatmap(
            dept_group_zscore,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            linewidths=1,
            cbar_kws={"label": "Z-Score (0: 전체 평균)"},
            ax=ax_hm,
        )
        ax_hm.set_title("학과별 상대적 생활 패턴 편차 (붉은색: 높음 / 푸른색: 낮음)", fontsize=13)

    ax_hm.set_ylabel("")
    st.pyplot(fig_hm)
    st.caption("💡 각 학과의 통학 부담, 학습량, 스마트폰 이용, 수면 부족 여부를 비교할 수 있습니다.")

with tab2:
    st.markdown("#### 🏆 주요 지표별 학과 순위 현황")
    col_rank_select = st.selectbox("정렬 기준 지표를 선택하세요:", metric_cols, index=0)

    ranked_df = dept_group_mean[[col_rank_select]].sort_values(by=col_rank_select, ascending=False)

    fig_rank_bar, ax_rk = plt.subplots(figsize=(9, 4))
    sns.barplot(
        x=ranked_df[col_rank_select],
        y=ranked_df.index,
        palette="crest",
        ax=ax_rk,
    )
    ax_rk.set_title(f"학과별 '{col_rank_select}' 순위", fontsize=12)
    ax_rk.set_xlabel(col_rank_select)
    ax_rk.set_ylabel("")

    for p in ax_rk.patches:
        ax_rk.annotate(
            f"{p.get_width():.1f}",
            (p.get_width(), p.get_y() + p.get_height() / 2),
            ha="left", va="center",
            xytext=(5, 0), textcoords="offset points",
            fontsize=10,
        )

    st.pyplot(fig_rank_bar)

st.markdown("---")


# =========================================================
# 6. [인터페이스 5] 하단 검증 영역 (접이식 Expander)
# =========================================================
with st.expander("📋 [데이터 무결성 검증] 정제 완료 데이터 테이블 및 원본 검토"):
    st.markdown("총학생회 집행부 및 학우 누구나 검증할 수 있도록 정제 완료된 설문 데이터를 공개합니다.")

    col_exp1, col_exp2 = st.columns([3, 1])
    with col_exp1:
        st.write(f"현재 필터 적용 데이터 행 수: **{len(filtered_df)}건** (전체 60건 중)")
    with col_exp2:
        csv_download = filtered_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 정제 데이터 CSV 다운로드",
            data=csv_download,
            file_name=f"campus_life_cleaned_{selected_dept}.csv",
            mime="text/csv",
        )

    st.dataframe(filtered_df, use_container_width=True)

    st.markdown("""
    **데이터 검증 확인 내역:**
    - `S017`: '만족도' 원본 문자열 `'모름'` ➡️ 사학과 중앙값(6.5점)으로 정상 치환 완료.
    - `S042`: '공부시간' 원본 결측치(`NaN`) ➡️ 사학과 평균(2.4시간)으로 정상 대치 완료.
    - 헤더 단위 정규화: `통학시간(분)`, `공부시간(시간)`, `스마트폰(시간)`, `수면시간(시간)`
    """)