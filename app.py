import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STORAGE_FILE = "movie_storage.json"

def load_local_data():
    """로컬 파일에서 영화 보관함 데이터 불러오기"""
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state['custom_movies'] = pd.DataFrame(data.get('movies', []))
                return
        except Exception:
            pass
            
    st.session_state['custom_movies'] = pd.DataFrame(columns=["id", "title", "genre", "overview", "features", "poster"])

def save_local_data():
    """영화 보관함 데이터 저장"""
    data = {
        'movies': st.session_state['custom_movies'].to_dict(orient='records')
    }
    with open(STORAGE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# 1. 페이지 설정 및 세션 초기화
st.set_page_config(page_title="3가지 알고리즘 영화 추천 시스템", layout="wide")

TMDB_API_KEY = "71f533a402be87b54aea626f2b1ef567" 

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'main'
if 'search_result' not in st.session_state:
    st.session_state['search_result'] = None

if 'custom_movies' not in st.session_state:
    load_local_data()

if 'custom_ratings' not in st.session_state:
    st.session_state['custom_ratings'] = {'다른 관객1': {}, '다른 관객2': {}, '나': {}}

for title in st.session_state['custom_movies']['title'].tolist():
    for user in st.session_state['custom_ratings']:
        if title not in st.session_state['custom_ratings'][user]:
            st.session_state['custom_ratings'][user][title] = np.nan

# TMDB API 검색 함수 (장르/줄거리 분리 수집)
def search_movie_tmdb(query):
    if not TMDB_API_KEY or TMDB_API_KEY == "여기에_발급받은_API키를_넣으세요":
        st.error("API 키를 등록해주세요.")
        return None
    
    url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}&language=ko-KR"
    try:
        response = requests.get(url).json()
        if response.get('results'):
            movie_data = response['results'][0]
            movie_id = movie_data['id']
            title = movie_data['title']
            plot = movie_data['overview'] if movie_data['overview'] else "줄거리 정보 없음"
            poster_path = movie_data['poster_path']
            poster_url = f"https://image.tmdb.org/t/p/w200{poster_path}" if poster_path else ""
            
            genre_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=ko-KR"
            genre_res = requests.get(genre_url).json()
            genres = " ".join([g['name'] for g in genre_res.get('genres', [])])
            if not genres:
                genres = "장르 정보 없음"
            
            return {
                "title": title, 
                "genre": genres, 
                "overview": plot, 
                "features": f"{genres} {plot}", 
                "poster": poster_url
            }
    except Exception as e:
        st.error(f"API 호출 오류: {e}")
    return None

# ==========================================
# 📄 메인 페이지
# ==========================================
if st.session_state['current_page'] == 'main':
    st.title("🎬 3가지 알고리즘 영화 추천 시스템")
    st.subheader("콘텐츠 기반 · 협업 필터링 · 하이브리드 모델")
    st.markdown("---")
    
    st.subheader("📥 API 연동 실시간 영화 등록")
    col_in1, col_in2 = st.columns([1, 2])
    
    with col_in1:
        search_query = st.text_input("영화 제목 검색:", placeholder="예: 토이 스토리")
        
        if st.button("🔍 영화 검색하기", use_container_width=True):
            if search_query:
                with st.spinner("TMDB API 검색 중..."):
                    result = search_movie_tmdb(search_query)
                    st.session_state['search_result'] = result
            else:
                st.error("검색어를 입력해 주세요.")
        
        if st.session_state['search_result']:
            res = st.session_state['search_result']
            st.markdown("---")
            st.write("### 🔍 검색 결과")
            
            preview_col1, preview_col2 = st.columns([1, 2])
            with preview_col1:
                if res['poster']:
                    st.image(res['poster'], width=120)
            with preview_col2:
                st.write(f"#### **{res['title']}**")
                st.caption(f"🏷️ **장르:** {res['genre']}")
                st.caption(f"📖 **줄거리:** {res['overview'][:80]}...")
            
            if st.button(f"➕ [{res['title']}] 보관함에 등록", type="primary", use_container_width=True):
                if res['title'] not in st.session_state['custom_movies']['title'].tolist():
                    new_id = len(st.session_state['custom_movies']) + 1
                    new_row = pd.DataFrame([{
                        "id": new_id, 
                        "title": res['title'], 
                        "genre": res['genre'],
                        "overview": res['overview'],
                        "features": res['features'], 
                        "poster": res['poster']
                    }])
                    st.session_state['custom_movies'] = pd.concat([st.session_state['custom_movies'], new_row], ignore_index=True)
                    
                    for user in st.session_state['custom_ratings']:
                        st.session_state['custom_ratings'][user][res['title']] = np.nan
                    
                    save_local_data()
                    st.success(f"🎉 [{res['title']}] 등록 완료!")
                    st.session_state['search_result'] = None
                    st.rerun()
                else:
                    st.warning("⚠️ 이미 등록된 영화입니다.")
                
    with col_in2:
        st.write("**현재 영화 보관함:**")
        if len(st.session_state['custom_movies']) == 0:
            st.info("등록된 영화가 없습니다.")
        else:
            movies_list = st.session_state['custom_movies']
            cols = st.columns(min(len(movies_list), 4))
            
            for idx, row in movies_list.iterrows():
                with cols[idx % 4]:
                    if row['poster']:
                        st.image(row['poster'], width=100)
                    st.caption(f"**{row['title']}**")
                    
                    if st.button("❌ 삭제", key=f"del_{row['title']}_{idx}", use_container_width=True):
                        st.session_state['custom_movies'] = st.session_state['custom_movies'][st.session_state['custom_movies']['title'] != row['title']].reset_index(drop=True)
                        for user in st.session_state['custom_ratings']:
                            if row['title'] in st.session_state['custom_ratings'][user]:
                                del st.session_state['custom_ratings'][user][row['title']]
                        save_local_data()
                        st.rerun()
            
            st.write("")
            if st.button("🗑️ 보관함 초기화", use_container_width=True):
                st.session_state['custom_movies'] = pd.DataFrame(columns=["id", "title", "genre", "overview", "features", "poster"])
                st.session_state['custom_ratings'] = {'다른 관객1': {}, '다른 관객2': {}, '나': {}}
                st.session_state['search_result'] = None
                if os.path.exists(STORAGE_FILE):
                    os.remove(STORAGE_FILE)
                st.rerun()

    st.markdown("---")
    st.subheader("🚀 알고리즘 모드 선택")
    
    disabled_btn = len(st.session_state['custom_movies']) < 2
    if disabled_btn:
        st.warning("⚠️ 영화가 최소 2개 이상 등록되어야 활성화됩니다.")
        
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("1️⃣ 콘텐츠 기반 페이지 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_content'
            st.rerun()
    with col2:
        if st.button("2️⃣ 협업 필터링 페이지 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_collaborative'
            st.rerun()
    with col3:
        if st.button("3️⃣ 하이브리드 페이지 열기", use_container_width=True, disabled=disabled_btn):
            st.session_state['current_page'] = 'page_hybrid'
            st.rerun()

# ==========================================
# 📄 화면 1: 콘텐츠 기반 필터링
# ==========================================
elif st.session_state['current_page'] == 'page_content':
    if st.button("⬅️ 메인으로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("1️⃣ 콘텐츠 기반 필터링")
    movies_db = st.session_state['custom_movies']
    
    user_history = st.multiselect("재밌게 본 영화를 선택하세요:", movies_db['title'].tolist())
    
    if user_history:
        try:
            tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            tfidf_matrix = tfidf.fit_transform(movies_db['features'])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            user_movie_indices = movies_db[movies_db['title'].isin(user_history)].index
            sim_scores = cosine_sim[user_movie_indices].mean(axis=0)
            
            recommend_df = movies_db.copy()
            recommend_df['similarity'] = sim_scores
            recommend_df = recommend_df[~recommend_df['title'].isin(user_history)]
            final_recommend = recommend_df.sort_values(by='similarity', ascending=False)
            
            st.subheader("🎯 추천 결과")
            for _, row in final_recommend.iterrows():
                c1, c2 = st.columns([1, 5])
                with c1:
                    if row['poster']: st.image(row['poster'], width=120)
                with c2:
                    st.write(f"### **{row['title']}** (매칭 점수: `{row['similarity']:.2f}`)")
                    st.caption(f"🏷️ **장르:** {row.get('genre', '정보없음')} | 📖 **줄거리:** {row.get('overview', '')[:90]}...")
                st.markdown("---")
        except Exception:
            st.error("분석 중 오류가 발생했습니다.")

# ==========================================
# 📄 화면 2: 협업 필터링
# ==========================================
elif st.session_state['current_page'] == 'page_collaborative':
    if st.button("⬅️ 메인으로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("2️⃣ 협업 필터링")
    movies_db = st.session_state['custom_movies']
    
    if st.button("다른 유저 평점 랜덤 생성"):
        for user in ['다른 관객1', '다른 관객2']:
            st.session_state['custom_ratings'][user] = {title: np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]) for title in movies_db['title'].tolist()}
        st.rerun()
            
    my_ratings = st.session_state['custom_ratings'].get('나', {})
    changed = False
    for idx, row in movies_db.iterrows():
        current_val = my_ratings.get(row['title'], np.nan)
        default_idx = 0 if pd.isna(current_val) else int(current_val)
        score = st.selectbox(f"[{row['title']}] 내 평점:", ["안봄(NaN)", "1", "2", "3", "4", "5"], index=default_idx, key=f"collab_rat_{row['title']}")
        new_val = np.nan if score == "안봄(NaN)" else float(score)
        if str(current_val) != str(new_val):
            my_ratings[row['title']] = new_val
            changed = True
        
    if changed:
        st.session_state['custom_ratings']['나'] = my_ratings
    
    ratings_df = pd.DataFrame(st.session_state['custom_ratings']).reindex(movies_db['title'].tolist())
    st.write("#### 📊 평점 현황판")
    st.dataframe(ratings_df)
    
    interaction_matrix = ratings_df.fillna(0)
    item_similarity = cosine_similarity(interaction_matrix)
    item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
    
    watched_movies = [m for m, r in my_ratings.items() if not pd.isna(r)]
    unwatched_movies = [m for m, r in my_ratings.items() if pd.isna(r)]
    
    if watched_movies and unwatched_movies:
        predictions = {}
        for movie in unwatched_movies:
            sim_sum, weighted_rating_sum = 0, 0
            for watched_movie in watched_movies:
                if movie in item_sim_df.index and watched_movie in item_sim_df.columns:
                    sim = item_sim_df.loc[movie, watched_movie]
                    rating = my_ratings[watched_movie]
                    sim_sum += sim
                    weighted_rating_sum += (sim * rating)
            predictions[movie] = weighted_rating_sum / sim_sum if sim_sum > 0 else 0
            
        st.subheader("🎯 예상 별점 결과")
        for movie, score in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            target_row = movies_db[movies_db['title'] == movie].iloc[0]
            c1, c2 = st.columns([1, 5])
            with c1:
                if target_row['poster']: st.image(target_row['poster'], width=100)
            with c2:
                st.write(f"### **{movie}** (예측 평점: `{score:.2f}`점)")
            st.markdown("---")

# ==========================================
# 📄 화면 3: 하이브리드 추천 (장르/줄거리 API 상세 점수 추가)
# ==========================================
elif st.session_state['current_page'] == 'page_hybrid':
    if st.button("⬅️ 메인으로 돌아가기"):
        st.session_state['current_page'] = 'main'
        st.rerun()
        
    st.header("3️⃣ 하이브리드 추천 시스템")
    movies_db = st.session_state['custom_movies']
    
    st.subheader("🎛️ 추천 비율 균형 조절")
    w_content = st.slider("콘텐츠(장르/줄거리) 가중치 (%)", 0, 100, 50)
    w_collab = 100 - w_content
    
    target_movie = st.selectbox("기준 영화 선택:", movies_db['title'].tolist())
    
    try:
        # 1. 콘텐츠 전체 유사도 연산
        tfidf_full = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        full_matrix = tfidf_full.fit_transform(movies_db['features'])
        full_sim = cosine_similarity(full_matrix, full_matrix)
        
        # 2. 장르 세부 유사도 연산 (API 데이터)
        tfidf_genre = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        genre_matrix = tfidf_genre.fit_transform(movies_db['genre'].fillna(''))
        genre_sim = cosine_similarity(genre_matrix, genre_matrix)
        
        # 3. 줄거리 세부 유사도 연산 (API 데이터)
        tfidf_plot = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        plot_matrix = tfidf_plot.fit_transform(movies_db['overview'].fillna(''))
        plot_sim = cosine_similarity(plot_matrix, plot_matrix)
        
        movie_idx = movies_db[movies_db['title'] == target_movie].index[0]
        
        content_scores = pd.Series(full_sim[movie_idx], index=movies_db['title'].tolist())
        genre_scores = pd.Series(genre_sim[movie_idx], index=movies_db['title'].tolist())
        plot_scores = pd.Series(plot_sim[movie_idx], index=movies_db['title'].tolist())
        
        # 4. 협업 필터링 유사도 연산
        ratings_df = pd.DataFrame(st.session_state['custom_ratings']).reindex(movies_db['title'].tolist())
        interaction_matrix = ratings_df.fillna(0)
        
        if target_movie in interaction_matrix.index and interaction_matrix.sum().sum() > 0:
            item_similarity = cosine_similarity(interaction_matrix)
            item_sim_df = pd.DataFrame(item_similarity, index=interaction_matrix.index, columns=interaction_matrix.index)
            collab_scores = item_sim_df[target_movie]
        else:
            collab_scores = pd.Series(0.0, index=movies_db['title'].tolist())
            
        # 5. 종합 점수 계산
        hybrid_df = pd.DataFrame({
            'content': content_scores,
            'genre': genre_scores,
            'plot': plot_scores,
            'collaborative': collab_scores
        }).fillna(0)
        
        hybrid_df['final_score'] = (hybrid_df['content'] * (w_content / 100)) + (hybrid_df['collaborative'] * (w_collab / 100))
        final_rank = hybrid_df.drop(target_movie, errors='ignore').sort_values(by='final_score', ascending=False)

        st.subheader("🎯 하이브리드 종합 스코어 및 세부 점수 분석")
        if final_rank.empty:
            st.info("비교할 다른 영화가 없습니다.")
        else:
            for title, row in final_rank.iterrows():
                target_row = movies_db[movies_db['title'] == title].iloc[0]
                
                c1, c2 = st.columns([1, 5])
                with c1:
                    if target_row['poster']: st.image(target_row['poster'], width=100)
                with c2:
                    st.write(f"### **{title}** (종합 점수: `{row['final_score']:.2f}`)")
                    
                    # 지표 분해 표시 (장르/줄거리/유저평점)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("🏷️ 장르 일치도", f"{row['genre']:.2f}")
                    m2.metric("📖 줄거리 매칭점수", f"{row['plot']:.2f}")
                    m3.metric("🧬 콘텐츠 통합점수", f"{row['content']:.2f}")
                    m4.metric("👥 유저평점 유사도", f"{row['collaborative']:.2f}")
                    
                    st.caption(f"📌 **[API 텍스트 요약]:** 장르({target_row['genre']}) | 줄거리({target_row['overview'][:60]}...)")
                st.markdown("---")
    except Exception as e:
        st.error(f"하이브리드 연산 실패: {e}")
