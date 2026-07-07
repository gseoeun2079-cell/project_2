import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. 페이지 설정 및 세션 상태 초기화
st.set_page_config(page_title="3가지의 알고리즘 영화 추천 시스템", layout="wide")

TMDB_API_KEY = "71f533a402be87b54aea626f2b1ef567" 

# 세션 상태 변수 초기화 (페이지 이동 및 동적 데이터 저장용)
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'main'
if 'custom_movies' not in st.session_state:
    st.session_state['custom_movies'] = pd.DataFrame(columns=["id", "title", "features", "poster"])
if 'custom_ratings' not in st.session_state:
    st.session_state['custom_ratings'] = {'기존유저A': {}, '기존유저B': {}, '나(타겟유저)': {}}
if 'search_result' not in st.session_state:
    st.session_state['search_result'] = None

# TMDB API 호출 함수
def search_movie_tmdb(query):
    if TMDB_API_KEY == "여기에_발급받은_API키를_넣으세요" or not TMDB_API_KEY:
        st.error("코드 상단의 TMDB_API_KEY 변수에 실제 API 키를 입력해야 작동합니다!")
        return None
    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=ko-KR"
    try:
        response = requests.get(url).json()
        if response.get('results'):
            movie_data = response['results'][0] # 가장 검색 연관성 높은 첫 번째 영화 선택
            movie_id = movie_data['id']
            title = movie_data['title']
            plot = movie_data['overview'] if movie_data['overview'] else "줄거리 정보 없음"
            poster_path = movie_data['poster_path']
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else ""
            
            # 상세 장르 가져오기
            genre_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=ko-KR"
            genre_res = requests.get(genre_url).json()
            genres = " ".join([g['name'] for g in genre_res.get('genres', [])])
            
            return {"title": title, "features": f"{genres} {plot}", "poster": poster_url}
    except Exception as e:
        st.error(f"API 호출 중 오류 발생: {e}")
    return None

#화면 0: 메인 페이지 (API 영화 검색 및 등록)
if st.session_state['current_page'] == 'main':
    st.title("🎬 3가지 알고리즘으로 만든 영화 추천 프로그램")
    st.subheader("콘텐츠 기반 · 협업 필터링 · 하이브리드 모델을 활용한 영화 플랫폼")
    st.write("실제 TMDB 영화 API 데이터와 사용자의 동적 입력 데이터를 기반으로 3가지 알고리즘을 실시간 작동 및 비교 검증합니다.")
    st.markdown("---")
    
    st.subheader("📥 API 연동 실시간 영화 등록")
    col_in1, col_in2 = st.columns([1, 2])
    
    with col_in1:
        search_query = st.text_input("영화 제목 검색 (한국어/영어 둘 다 가능):", placeholder="예: 토이스토리 또는 주토피아")
        
        if st.button("🔍 영화 검색하기", use_container_width=True):
            if search_query:
                with st.spinner("TMDB API에서 영화 데이터를 찾는 중..."):
                    result = search_movie_tmdb(search_query)
                    if result:
                        st.session_state['search_result'] = result
                    else:
                        st.error("영화 데이터를 찾지 못했습니다. 제목을 정확히 입력해 주세요.")
                        st.session_state['search_result'] = None
            else:
                st.error("검색할 영화 제목을 입력하세요.")
        
        if st.session_state['search_result']:
            res = st.session_state['search_result']
            st.markdown("---")
            st.write("### 🔍 검색된 영화 확인")
            
            preview_col1, preview_col2 = st.columns([1, 2])
            with preview_col1:
                if res['poster']:
                    st.image(res['poster'], width=120)
            with preview_col2:
                st.write(f"#### **{res['title']}**")
                st.caption(f"**데이터 내용:** {res['features'][:120]}...")
            
            if st.button(f"➕ [{res['title']}]을 영화 풀에 등록하기", type="primary", use_container_width=True):
                if res['title'] not in st.session_state['custom_movies']['title'].tolist():
                    new_id = len(st.session_state['custom_movies']) + 1
                    new_row = pd.DataFrame([{"id": new_id, "title": res['title'], "features": res['features'], "poster": res['poster']}])
                    st.session_state['custom_movies'] = pd.concat([st.session_state['custom_movies'], new_row], ignore_index=True)
                    
                    for user in st.session_state['custom_ratings']:
                        if res['title'] not in st.session_state['custom_ratings'][user]:
                            st.session_state['custom_ratings'][user][res['title']] = np.nan
                            
                    st.success(f"🎉 [{res['title']}] 등록 성공!")
                    st.session_state['search_result'] = None
                    st.rerun()
                else:
                    st.warning("⚠️ 이미 등록된 영화입니다.")
                
    with col_in2:
        st.write("**현재 API로 구축된 내 영화 풀 (Pool):**")
        if len(st.session_state['custom_movies']) == 0:
            st.info("아직 등록된 영화가 없습니다. 왼쪽에 실제 영화를 검색해 채워보세요! (서로 다른 장르로 4개 이상 등록 권장)")
        else:
            movies_list = st.session_state['custom_movies']
            cols = st.columns(min(len(movies_list), 4))
            
            for idx, row in movies_list.iterrows():
                with cols[idx % 4]:
                    if row['poster']:
                        st.image(row['poster'], width=100)
                    st.caption(f"**{row['title']}**")
                    
                    # 🌟 [추가] 영화 개별 삭제 버튼 구현
                    if st.button("❌ 삭제", key=f"del_{row['title']}_{idx}", use_container_width=True, type="secondary"):
                        # 1. 영화 데이터프레임에서 제외
                        st.session_state['custom_movies'] = st.session_state['custom_movies'][st.session_state['custom_movies']['title'] != row['title']].reset_index(drop=True)
                        
                        # 2. 유저별 평점 사전 파일에서도 연쇄적으로 해당 영화 명단 삭제
                        for user in st.session_state['custom_ratings']:
                            if row['title'] in st.session_state['custom_ratings'][user]:
                                del st.session_state['custom_ratings'][user][row['title']]
                                
                        st.rerun()
            
            st.write("")
            if st.button("🗑️ 영화 데이터 전체 초기화", use_container_width=True):
                st.session_state['custom_movies'] = pd.DataFrame(columns=["id", "title", "features", "poster"])
                st.session_state['custom_ratings'] = {'기존유저A': {}, '기존유저B': {}, '나(타겟유저)': {}}
                st.session_state['search_result'] = None
                st.rerun()

    st.markdown("---")
    st.subheader("🚀 2단계: 알고리즘 프로그램 실행")
    
    disabled_btn = len(st.session_state['custom_movies']) < 2
    if disabled_btn:
        st.warning("⚠️ 영화가 최소 2개 이상 등록되어야 알고리즘 페이지가 활성화됩니다.")
        
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 1️⃣ 콘텐츠 기반 필터링")
        st.write("API가 긁어온 실제 영화 줄거리 및 장르 텍스트 데이터를 분석해 추천합니다.")
        if st.button("콘텐츠 기반 프로그램 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_content'
            st.rerun()
    with col2:
        st.success("### 2️⃣ 협업 필터링")
        st.write("등록한 실제 영화들에 나만의 평점을 부여해 가상의 집단지성 행렬을 구성합니다.")
        if st.button("협업 필터링 프로그램 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_collaborative'
            st.rerun()
    with col3:
        st.warning("### 3️⃣ 하이브리드 시스템")
        st.write("콘텐츠 텍스트 스코어와 협업 집단 평점 스코어를 가중 결합하여 하이브리드 추천을 진행합니다.")
        if st.button("하이브리드 프로그램 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_hybrid'
            st.rerun()


# ==========================================
# 📄 화면 1: 콘텐츠 기반 필터링 페이지
# ==========================================
elif st.session_state['current_page'] == 'page_content':
    if st.button("⬅️ 메인 페이지로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("1️⃣ 콘텐츠 기반 필터링 (Content-Based Filtering)")
    movies_db = st.session_state['custom_movies']
    
    user_history = st.multiselect("당신이 과거에 극장에서 재밌게 본 영화를 선택하세요 (복수 선택 가능):", movies_db['title'].tolist())
    
    if user_history:
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(movies_db['features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        user_movie_indices = movies_db[movies_db['title'].isin(user_history)].index
        sim_scores = cosine_sim[user_movie_indices].mean(axis=0)
        
        recommend_df = movies_db.copy()
        recommend_df['similarity'] = sim_scores
        recommend_df = recommend_df[~recommend_df['title'].isin(user_history)]
        final_recommend = recommend_df.sort_values(by='similarity', ascending=False)
        
        st.subheader("🎯 API 텍스트 분석 기반 개인화 추천 결과")
        if final_recommend.empty:
            st.info("모든 영화를 선택하셨습니다! 추천할 다른 영화가 없습니다.")
        else:
            for _, row in final_recommend.iterrows():
                c1, c2 = st.columns([1, 5])
                with c1:
                    if row['poster']: st.image(row['poster'], width=120)
                with c2:
                    st.write(f"### **{row['title']}** (유사도 매칭 점수: `{row['similarity']:.2f}`)")
                    st.write(f"💬 **[설명 가능한 AI 요약]:** 이 영화의 API 데이터베이스 요약본(`{row['features'][:70]}...`)이 유저님의 과거 선호 장르/소재 패턴과 일치하여 추천되었습니다.")
                st.markdown("---")


# ==========================================
# 📄 화면 2: 협업 필터링 페이지
# ==========================================
elif st.session_state['current_page'] == 'page_collaborative':
    if st.button("⬅️ 메인 페이지로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("2️⃣ 협업 필터링 (Collaborative Filtering)")
    movies_db = st.session_state['custom_movies']
    
    st.write("### 🎲 1단계: 기존 가상 인구 집단의 평점 매핑")
    if st.button("타 유저 평점 데이터 랜덤 제너레이트"):
        for user in ['기존유저A', '기존유저B']:
            st.session_state['custom_ratings'][user] = {title: np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]) for title in movies_db['title'].tolist()}
        st.rerun()
            
    st.write("### 👤 2단계: 내 실제 영화 관람 평점 입력")
    my_ratings = st.session_state['custom_ratings'].get('나(타겟유저)', {})
    
    for idx, row in movies_db.iterrows():
        current_val = my_ratings.get(row['title'], np.nan)
        default_idx = 0 if pd.isna(current_val) else int(current_val)
        
        score = st.selectbox(f"[{row['title']}] 영화에 내 평점은?", ["안봄(NaN)", "1", "2", "3", "4", "5"], index=default_idx, key=f"collab_rat_{row['title']}")
        my_ratings[row['title']] = np.nan if score == "안봄(NaN)" else float(score)
        
    st.session_state['custom_ratings']['나(타겟유저)'] = my_ratings
    
    ratings_df = pd.DataFrame(st.session_state['custom_ratings']).reindex(movies_db['title'].tolist())
    st.write("#### 📊 생성된 실시간 실물 영화 사용자-아이템 행렬 (User-Item Matrix)")
    st.dataframe(ratings_df)
    
    interaction_matrix = ratings_df.fillna(0)
    item_similarity = cosine_similarity(interaction_matrix)
    item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
    
    watched_movies = [m for m, r in my_ratings.items() if not pd.isna(r)]
    unwatched_movies = [m for m, r in my_ratings.items() if pd.isna(r)]
    
    if watched_movies and unwatched_movies:
        predictions = {}
        for movie in unwatched_movies:
            sim_sum = 0
            weighted_rating_sum = 0
            for watched_movie in watched_movies:
                if movie in item_sim_df.index and watched_movie in item_sim_df.columns:
                    sim = item_sim_df.loc[movie, watched_movie]
                    rating = my_ratings[watched_movie]
                    sim_sum += sim
                    weighted_rating_sum += (sim * rating)
            predictions[movie] = weighted_rating_sum / sim_sum if sim_sum > 0 else 0
            
        st.subheader("🎯 집단 행동 패턴 분석 기반 예상 별점 결과")
        for movie, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            target_rows = movies_db[movies_db['title'] == movie]
            if target_rows.empty: continue
            target_row = target_rows.iloc[0]
            
            c1, c2 = st.columns([1, 5])
            with c1:
                if target_row['poster']: st.image(target_row['poster'], width=100)
            with c2:
                st.write(f"### **{movie}** (예측 평점 점수: `{score:.2f}` 점)")
                if score == 0:
                    st.error("⚠️ 데이터 희소성으로 이 영화를 평가한 다른 유저 세트가 없어 연산이 제한됩니다 (콜드 스타트).")
                else:
                    st.info("🎁 **[세렌디피티 발동]:** 이 영화의 시놉시스는 알지 못하지만 다른 관객들과의 평점 주파수가 일치하여 뜻밖에 도출된 개인화 영화입니다.")
            st.markdown("---")
    else:
        st.warning("추천을 보려면 최소 1개의 영화는 평점을 주고, 1개 이상의 영화는 '안봄(NaN)' 상태로 두세요.")


# ==========================================
# 📄 화면 3: 하이브리드 추천 시스템 페이지
# ==========================================
elif st.session_state['current_page'] == 'page_hybrid':
    if st.button("⬅️ 메인 페이지로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("3️⃣ 하이브리드 추천 시스템 (Hybrid Recommendation)")
    movies_db = st.session_state['custom_movies']
    
    st.subheader("🎛️ 하이브리드 합성 계수 조절")
    w_content = st.slider("콘텐츠(줄거리/장르) 가중치 설정 (%)", 0, 100, 50)
    w_collab = 100 - w_content
    
    target_movie = st.selectbox("추천 알고리즘의 기준점이 될 메인 영화 선택:", movies_db['title'].tolist())
    
    # 1. 콘텐츠 점수 연산
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(movies_db['features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    movie_idx = movies_db[movies_db['title'] == target_movie].index[0]
    
    content_scores = pd.Series(cosine_sim[movie_idx], index=movies_db['title'].tolist())
    
    # 2. 협업 점수 연산
    ratings_df = pd.DataFrame(st.session_state['custom_ratings']).reindex(movies_db['title'].tolist())
    interaction_matrix = ratings_df.fillna(0)
    
    if target_movie in interaction_matrix.index and interaction_matrix.sum().sum() > 0:
        item_similarity = cosine_similarity(interaction_matrix)
        item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
        collab_scores = item_sim_df[target_movie]
    else:
        collab_scores = pd.Series(0.0, index=movies_db['title'].tolist())
        
    # 3. 종합 하이브리드 스코어링
    hybrid_df = pd.DataFrame({'content': content_scores, 'collaborative': collab_scores}).fillna(0)
    hybrid_df['final_score'] = (hybrid_df['content'] * (w_content / 100)) + (hybrid_df['collaborative'] * (w_collab / 100))
    
    final_rank = hybrid_df.drop(target_movie, errors='ignore').sort_values(by='final_score', ascending=False)
    
    if collab_scores.sum() == 0:
        st.warning("🚨 **[전환 방식(Switching) 제어 가동]:** 협업 필터링 행렬 데이터 소스가 비어있거나 평점 정보가 없어 콘텐츠 기반 메커니즘이 안전장치로 전면 대체 구동됩니다.")

    st.subheader("🎯 하이브리드 엔진 종합 스코어 랭킹")
    if final_rank.empty:
        st.info("비교 분석할 다른 영화가 풀에 존재하지 않습니다.")
    else:
        for title, row in final_rank.iterrows():
            target_rows = movies_db[movies_db['title'] == title]
            if target_rows.empty: continue
            target_row = target_rows.iloc[0]
            
            c1, c2 = st.columns([1, 5])
            with c1:
                if target_row['poster']: st.image(target_row['poster'], width=100)
            with c2:
                st.write(f"### **{title}** (종합 가중 점수: `{row['final_score']:.2f}`)")
                st.caption(f"🧬 [API 메타 데이터 점수]: {row['content']:.2f}  |  [인구 집단 데이터 점수]: {row['collaborative']:.2f}")
            st.markdown("---")
