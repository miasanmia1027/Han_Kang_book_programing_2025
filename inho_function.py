import os # 기본으로 설치 되어있음.
import re # 기본으로 설치 되어있음.
from pykospacing import Spacing
from sentence_transformers import SentenceTransformer, util
import pandas as pd 
import numpy as np
# import ace_tools as tools
import ace_tools_open as tools
import pickle
import seaborn as sns
import matplotlib.pyplot as plt 
from itertools import combinations # 기본으로 설치 되어있음.
from collections import defaultdict # 기본으로 설치 되어있음.
from konlpy.tag import Okt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
#----------------------------------------------------------------
def rawdata_to_cleandata(data_file_path):
    with open(f'{data_file_path}.txt', 'r', encoding='utf-8') as file:
        content = file.read()

    # 공백 제거 및 문장 분리
    content = "".join(content.split())
    sentences = re.split(r'[.!?]', content)
    # 빈 문장 제거 및 큰따옴표 제거
    clean_sentences = [re.sub(r'"', '', sentence) for sentence in sentences if sentence]
    # 공백 복원
    spacing = Spacing()
    spaced_sentences = [spacing(sentence.strip()) for sentence in tqdm(clean_sentences)]
    
    with open(f'{data_file_path}.txt', 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(spaced_sentences))
    return 
#----------------------------------------------------------------
def txt_to_pkl_tokens(data_file):
    with open(f'{data_file}', "r", encoding="utf-8") as f:
        book = f.readlines()

    okt = Okt()
    tokenized_sentences = []
    
    for index, line in tqdm(enumerate(book)):
        # 명사 추출
        nouns = okt.nouns(line)

        # 불용어 제거 + 길이 필터링
        stopwords = {}
        filtered_tokens = [
            word for word in nouns 
            if word not in stopwords and len(word) > 1
        ]

        filtered_tokens = list(set(filtered_tokens))  # 중복 제거
        tokenized_sentences.append({
            "index": index,
            "num_tokens": len(filtered_tokens),# 문장 단위로 짤라야 하기 때문에
            "tokens": filtered_tokens
        })
    # 결과를 .pkl 파일로 저장
    with open(f"{data_file}.pkl", "wb") as f:
        pickle.dump(tokenized_sentences, f)
    return 
#----------------------------------------------------------------
from gensim.models import KeyedVectors
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
def pkl_token_to_vectoredata(file_path,name):
        # 1. FastText 모델 로드
    # ft_model = fasttext.load_model("no_han_book/cc.ko.300.bin")

    # # 2. 단어 리스트 가져오기 (에러 무시)
    # words = ft_model.get_words(on_unicode_error='ignore')

    # # 3. Gensim KeyedVectors 생성
    # kv_model = KeyedVectors(vector_size=ft_model.get_dimension())
    # kv_model.add_vectors(words, [ft_model.get_word_vector(w) for w in tqdm(words)])

    # # 4. 저장
    # kv_model.save("no_han_book/cc.ko.300.kv")
    # 여기는 그냥 코드를 가져옴


    model = KeyedVectors.load("no_han_book/cc.ko.300.kv")
    with open(file_path, "rb") as f:
        pkl_data = pickle.load(f)
    # 3. pkl 파일에서 모든 단어 추출
    # 여기서 문제는 문장 단위로 분석이 진행이 안된다 // 약간 아쉬운디
    all_tokens = []
    for entry in pkl_data:
        tokens = entry.get('tokens', [])
        all_tokens.extend(tokens)
    # 4. 중복 제거 및 유효 단어 필터링 (모델에 포함된 단어만 사용)
    unique_tokens = list(set(all_tokens))
    valid_tokens = [word for word in unique_tokens if word in model] #이게 어떤 변수를 만들지 모르겠다
    # 5. 단어 벡터화
    token_vectors = {}
    for token in tqdm(valid_tokens, desc="단어 벡터화 중"):# desc는 tqdm에서 글을 쓰는 방식
        token_vectors[token] = model[token]
        # 여기서 임베딩되어 있는 수치를 기록중인 과정이다

    # 6. 유사도 매트릭스 생성
    valid_word_list = list(token_vectors.keys())# 이게 목록의 제목을 가져오는디 이게 뭐지?
    vectors = [token_vectors[word] for word in valid_word_list]
    similarity_matrix = cosine_similarity(vectors)
    # 코사인 유사도 계산
    
    similarity_df = pd.DataFrame(similarity_matrix, index=valid_word_list, columns=valid_word_list)
    similarity_df.to_csv(f"no_han_book/{name}_similer.csv", encoding="UTF-8")
    return 
#----------------------------------------------------------------
#----------------------------------------------------------------
#----------------------------------------------------------------







