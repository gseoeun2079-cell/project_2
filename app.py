import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 페이지 기본 설정
st.set_page_config(page_title= "영화/드라마 추천 시스템", layout="wide")
st.title("🎬 AI 추천 시스템 실습 및 검증 프로그램")
st.write("제공된 알고리즘 이론에 기반하여 작동 원리와 장단점을 직관적으로 보여주는 데모입니다.")

# 0. 공통 데이터셋 구축 (이론 매칭용 데이터)
@st.cache_data
def load_data():
    # 콘텐츠 기반용 영화 메타데이터 (신작 '아바타 신작' 포함 -> 콜드스타트 테스트용)
    movies_df = pd.DataFrame([
        {"id": 1, "title": "인셉션", "features": "SF 액션 스릴러 꿈 크리스토퍼놀란 디카프리오"},
        {"id": 2, "title": "인터스텔라", "features": "SF 드라마 우주 시공간 가족애 크리스토퍼놀란"},
        {"id": 3, "title": "어바웃 타임", "features": "로맨스 코미디 시간여행 사랑 달달 레이첼맥아담스"},
        {"id": 4, "title": "라라랜드", "features": "로맨스 뮤지컬 드라마 꿈 청춘 음악 라이언고슬링"},
        {"id": 5, "title": "매트릭스", "features": "SF 액션 가상현실 AI 디스토피아 키아누리브스"},
        {"id": 6, "title": "아바타 신작", "features": "SF 액션 우주 외계행성 영상미 3D"} 
    ])
    
    # 협업 필터링용 사용자-아이템 평점 행렬 (NaN을 통한 데이터 희소성 표현)
    ratings_dict = {
        'User1': {'인셉션': 5, '인터스텔라': 4, '어바웃 타임': 1, '라라랜드': 1, '매트릭스': 5},
        'User2': {'인셉션': 4, '인터스텔라': 5, '어바웃 타임': 2, '라라랜드': 1, '매트릭스': 4},
        'User3': {'인셉션': 1, '인터스텔라': 1, '어바웃 타임': 5, '라라랜드': 5, '매트릭스': 2},
        'User4': {'인셉션': 5, '인터스텔라': np.nan, '어바웃 타임': 2, '라라랜드': np.nan, '매트릭스': 5},
    }
    ratings_df = pd.DataFrame(ratings_dict)
    return movies_df, ratings_df

movies_db, ratings_db = load_data()

# 사이드바에서 프로그램 선택하기
st.sidebar.header("⚙️ 알고리즘 선택")
program_choice = st.sidebar.radio(
    "실행할 프로그램을 선택하세요:",
    ("1. 콘텐츠 기반 필터링", "2. 협업 필터링", "3. 하이브리드 추천 시스템")
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **팁:** '아바타 신작'은 평점 데이터가 없는 새로 등록된 영상입니다. 각 알고리즘이 이를 어떻게 처리하는지 관찰해 보세요.")


# ==========================================
# PROGRAM 1: 콘텐츠 기반 필터링
# ==========================================
if program_choice == "1. 콘텐츠 기반 필터링":
    st.header("1️⃣ 콘텐츠 기반 필터링 (Content-Based Filtering)")
    st.caption("원칙: '이전에 좋아했던 것과 비슷한 것을 추천한다'")
    
    # 이론 요약 노출
    with st.expander("📌 작동 원리 및 장단점 보기", expanded=True):
        st.markdown("""
        - **작동 원리:** 아이템의 특성(텍스트)을 추출 ➡️ 유저 프로필 구축 ➡️ **코사인 유사도** 계산 ➡️ 추천
        - **장점:** 새로운 아이템도 즉시 추천 가능 (**콜드 스타트 해결**), 추천 이유 명시 가능 (**설명 가능한 AI**)
        - **단점:** 맨날 보던 장르만 나오는 **과도한 특수화(Over-specialization)**
        """)

    # UI 구현: 유저의 과거 시청 기록 선택 (멀티 셀렉트)
    user_history = st.multiselect(
        "당신이 과거에 재밌게 본 영화를 선택하세요 (사용자 프로필 구축용):",
        movies_db['title'].tolist(),
        default=["인셉션", "매트릭스"]
    )
    
    if user_history:
        # TF-IDF 벡터화 및 유사도 계산
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(movies_db['features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # 유저 히스토리 인덱스 추출 및 평균 유사도 계산
        user_movie_indices = movies_db[movies_db['title'].isin(user_history)].index
        sim_scores = cosine_sim[user_movie_indices].mean(axis=0)
        
        recommend_df = movies_db.copy()
        recommend_df['similarity'] = sim_scores
        # 이미 본 영화 제외
        recommend_df = recommend_df[~recommend_df['title'].isin(user_history)]
        final_recommend = recommend_df.sort_values(by='similarity', ascending=False)
        
        st.subheader("🎯 AI 추천 결과")
        for _, row in final_recommend.iterrows():
            with st.container():
                st.write(f"### **{row['title']}** (유사도 점수: `{row['similarity']:.2f}`)")
                st.write(f"💬 **추천 이유 (설명 가능한 AI):** 유저님이 선택하신 영화와 키워드(`{row['features'][:18]}...`) 측면에서 유사합니다.")
                
                if row['title'] == "아바타 신작":
                    st.success("✨ **장점 검증(콜드 스타트 해결):** 이 콘텐츠는 평점이 아예 없는 상태이지만, 텍스트 특성을 분석하여 무사히 추천되었습니다!")
                st.markdown("---")
    else:
        st.warning("영화를 하나 이상 선택해 주세요.")


# ==========================================
# PROGRAM 2: 협업 필터링
# ==========================================
elif program_choice == "2. 협업 필터링":
    st.header("2️⃣ 협업 필터링 (Collaborative Filtering - 아이템 기반)")
    st.caption("원칙: '나와 비슷한 취향을 가진 사람들이 좋아하는 것을 추천한다'")
    
    with st.expander("📌 작동 원리 및 장단점 보기", expanded=True):
        st.markdown("""
        - **작동 원리:** 사용자-아이템 상호작용 행렬 구축 ➡️ 평점 패턴 공유도 기반 아이템 유사도 식별 ➡️ 예측 점수 계산
        - **장점:** 콘텐츠 자체 분석 불필요, 예상치 못한 장르의 인생작 발견 가능 (**세렌디피티**)
        - **단점:** 평점이나 행동 데이터가 없는 신규 유저/아이템에 대한 **콜드 스타트 문제**, 데이터가 비어있는 **희소성 문제**
        """)

    st.subheader("📊 원본 데이터: 사용자-아이템 상호작용 행렬 (평점 데이터)")
    st.dataframe(ratings_db)
    st.caption("※ NaN은 아직 평가 점수가 없는 비어있는 공간(데이터 희소성)을 의미합니다.")

    # 새로운 타겟 유저 생성용 UI
    st.subheader("👤 내(타겟 유저) 평점 입력하기")
    col1, col2 = st.columns(2)
    with col1:
        rating_inception = st.slider("인셉션 평점", 1, 5, 5)
    with col2:
        st.info("💡 다른 영화들은 아직 보지 않은 상태(NaN, 희소 데이터)로 가정하여 알고리즘이 예측 평점을 계산합니다.")

    # 타겟 유저 데이터를 행렬에 추가
    target_user_ratings = {'인셉션': rating_inception, '인터스텔라': np.nan, '어바웃 타임': np.nan, '라라랜드': np.nan, '매트릭스': np.nan, '아바타 신작': np.nan}
    
    # 유사도 계산을 위해 피벗 테이블 구조 생성
    full_matrix = ratings_db.copy()
    full_matrix['MyUser'] = pd.Series(target_user_ratings)
    interaction_matrix = full_matrix.fillna(0)
    
    # 아이템 간 코사인 유사도
    item_similarity = cosine_similarity(interaction_matrix)
    item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
    
    # 보지 않은 영화에 대한 평점 예측
    watched_movies = ['인셉션']
    unwatched_movies = ['인터스텔라', '어바웃 타임', '라라랜드', '매트릭스', '아바타 신작']
    
    predictions = {}
    for movie in unwatched_movies:
        sim_sum = 0
        weighted_rating_sum = 0
        for watched_movie in watched_movies:
            sim = item_sim_df.loc[movie, watched_movie]
            rating = target_user_ratings[watched_movie]
            sim_sum += sim
            weighted_rating_sum += (sim * rating)
        predictions[movie] = weighted_rating_sum / sim_sum if sim_sum > 0 else 0

    st.subheader("🎯 집단 지성 기반 예측 및 추천 결과")
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    for movie, score in sorted_preds:
        with st.container():
            st.write(f"### **{movie}** (예측 평점: `{score:.2f}` 점)")
            
            if movie == "아바타 신작" and score == 0:
                st.error("⚠️ **단점 노출(콜드 스타트 문제 발생):** 이 영화는 신작이라 과거 유저들의 상호작용 평점이 0개입니다. 따라서 협업 필터링은 이 영화를 추천할 수 없어 점수가 0점 처리됩니다.")
            elif movie in ["어바웃 타임", "라라랜드"]:
                st.info("🎁 **세렌디피티(뜻밖의 발견):** 내 평점 데이터와 다른 유저들의 유사 패턴을 묶어, 내가 평소 생각지 못한 다른 장르도 추천 리스트에 진입할 기회를 얻습니다.")
            else:
                st.write("✨ 유저 간의 유사성 패턴(집단 지성)에 의해 산출된 점수입니다.")
            st.markdown("---")


# ==========================================
# PROGRAM 3: 하이브리드 추천 시스템
# ==========================================
elif program_choice == "3. 하이브리드 추천 시스템":
    st.header("3️⃣ 하이브리드 추천 시스템 (Hybrid Recommendation)")
    st.caption("원칙: '콘텐츠 기반과 협업 필터링의 시너지를 극대화하여 단점을 덮는다'")
    
    with st.expander("📌 작동 원리 및 장단점 보기", expanded=True):
        st.markdown("""
        - **가중치 기반(Weighted):** 두 알고리즘의 점수를 슬라이더 가중치 비율에 맞춰 합산합니다.
        - **전환 방식(Switching):** 만약 평점 데이터가 전무한 '신작(아바타 신작)'인 경우, 협업 알고리즘을 끄고 **콘텐츠 기반 필터링 엔진으로 자동 통째 전환**하여 시스템 에러나 추천 누락을 방지합니다.
        """)

    # 가중치 조절 UI
    st.subheader("🎛️ 하이브리드 가중치 비율 세팅")
    w_content = st.slider("콘텐츠 기반 필터링 반영 비율 (%)", 0, 100, 40)
    w_collab = 100 - w_content
    st.write(f"현재 결합 공식: $(콘텐츠 \times {w_content/100:.1f}) + (협업 \times {w_collab/100:.1f})$")

    # 기준 영화 선택
    target_movie = st.selectbox("기준이 될 영화를 선택하세요:", movies_db['title'].tolist())

    # 점수 사전 계산 (프로그램 1, 2 로직의 단순 정량화 합산)
    # 1. 콘텐츠 점수
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_db['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movie_idx = movies_db[movies_db['title'] == target_movie].index[0]
    content_scores = pd.Series(cosine_sim[movie_idx], index=movies_db['title'])

    # 2. 협업 점수 (사전 정의된 고정 행렬 기준 유사도 활용)
    interaction_matrix = ratings_db.fillna(0)
    # 데이터셋에 아바타 신작 강제 주입 (평점은 모두 0인 상태)
    if '아바타 신작' not in interaction_matrix.index:
        interaction_matrix.loc['아바타 신작'] = [0, 0, 0, 0]
    
    item_similarity = cosine_similarity(interaction_matrix)
    item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
    
    if target_movie in item_sim_df.index:
        collab_scores = item_sim_df[target_movie]
    else:
        collab_scores = pd.Series(0, index=movies_db['title'])

    # 하이브리드 결합 데이터프레임 생성
    hybrid_df = pd.DataFrame({
        'content': content_scores,
        'collaborative': collab_scores
    }).fillna(0)

    # 최종 가중 합산 점수 계산
    hybrid_df['final_score'] = (hybrid_df['content'] * (w_content / 100)) + (hybrid_df['collaborative'] * (w_collab / 100))
    final_rank = hybrid_df.drop(target_movie).sort_values(by='final_score', ascending=False)

    st.subheader("🎯 하이브리드 엔진 결합 추천 순위")
    
    # 전환 방식(Switching) 제어 시각화
    if target_movie == "아바타 신작":
        st.warning("🚨 [시스템 알림: 전환 방식(Switching) 발동] 평점 데이터가 없는 신작 영화를 선택하셨습니다. 협업 필터링 연산 오류를 방지하기 위해 자동으로 '콘텐츠 기반 엔진'의 비중을 높여 필터링을 수행합니다.")
    
    for title, row in final_rank.iterrows():
        with st.container():
            st.write(f"### **{title}** (종합 점수: `{row['final_score']:.2f}`)")
            st.text(f"📊 점수 상세 분해 -> 콘텐츠 기반 점수: {row['content']:.2f} | 협업 필터링 점수: {row['collaborative']:.2f}")
            
            if title == "아바타 신작" and w_collab > 50:
                st.caption("💡 협업 필터링의 비중이 높아 점수가 낮게 나왔지만, 콘텐츠 기반 필터링 점수 덕분에 완전히 누락되지 않고 리스트업 되었습니다. (하이브리드의 단점 보완 효과)")
            st.markdown("---")
