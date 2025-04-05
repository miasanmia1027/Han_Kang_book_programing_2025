import os
import re
from pykospacing import Spacing
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
# import ace_tools as tools
import ace_tools_open as tools
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from konlpy.tag import Okt
#------------------------------------------------------------------------------
def rawdata_to_cleandata(data_file):
    with open(f'{data_file}', 'r', encoding='utf-8') as file:
        content = file.read()

    content = "".join(content.split())
    sentences = re.split(r'[.!?]', content)

    # 빈 문장 제거 및 큰따옴표 제거
    clean_sentences = [re.sub(r'"', '', sentence) for sentence in sentences if sentence]

    spacing = Spacing()
    
    sentences = clean_sentences
    # 공백 복원
    spaced_sentences = [spacing(sentence.strip()) for sentence in sentences]
    with open('clean_data.txt', 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(spaced_sentences))
#------------------------------------------------------------------------------
def book_to_pkl_nouns(data_file):
    file_path = f"{data_file}"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 줄 단위로 읽기

    stopwords = {"그","그녀","것","때"}

    okt = Okt()
    tokenized_sentences = []
    filtered_words = []  # 불용어 제거된 단어만 저장할 리스트

    for idx, line in enumerate(lines):
        nouns = okt.nouns(line)  # 명사만 추출

        # 🔹 불용어 제거 (컴프리헨션 없이 작성)
        filtered_nouns = []
        for word in nouns:
            if word not in stopwords:
                filtered_nouns.append(word)

        filtered_nouns = list(set(filtered_nouns))  # 중복 제거-> 한 문장에 여러개의 것이 있더라도 무시
        #나는 이게 굳이 있어야 하나 싶기는 함

        tokenized_sentences.append({"index": idx, "num_nouns": len(filtered_nouns), "nouns": filtered_nouns})
        
        # 모든 단어를 리스트에 추가
        filtered_words.extend(filtered_nouns)

    # 결과 저장 (불용어 처리된 데이터 저장)
    with open("nouns_lines.pkl", "wb") as f:
        pickle.dump(tokenized_sentences, f)
#------------------------------------------------------------------------------
def nouns_신뢰도(pkl_file):
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)

   # 명사 등장 횟수 및 조합 빈도 계산
    word_pairs = defaultdict(int)
    word_counts = defaultdict(int)
    # 이거는 처음 등장하는 쌍(pairs)도 안전하게 카운트를 할 수 있게 하는 코드
    # -> 기존의 dict를 쓰면 없는거에 대해 0으로 처리를 안해주기 때문에 문제가 생기는 것이다. 
    total_sentences = len(sentences)

    for entry in sentences:
        nouns = entry["nouns"]  # 명사 리스트 가져오기

        # 개별 명사 등장 횟수 계산 -> 이거는 하나 즉 A,B 이런거구나
        for word in nouns:
            word_counts[word] += 1


        # 명사 쌍 등장 횟수 계산 -> 그러면 이게 전체구나 즉 A,B둘다 함께 등장한 수
        for pair in combinations(nouns, 2):
            word_pairs[pair] += 1

    # 신뢰도 계산
    confidence_values = []
    for pair, support_AB in word_pairs.items():
        A, B = pair
        # 이거는 왜 한거야?

        support_A = word_counts.get(A, 0)   # A의 지지도(등장한 횟수)-> 키 값으로 값을 가져오는 방식
        support_B = word_counts.get(B, 0)   # B의 지지도(등장한 횟수)


        confidence_A_to_B = support_AB / support_A if support_A > 0 else 0# 이거는 혹시 0으로 나눌까봐 만든 예비 장치
        confidence_B_to_A = support_AB / support_B if support_B > 0 else 0# 이거는 혹시 0으로 나눌까봐 만든 예비 장치

        confidence_values.append((pair, confidence_A_to_B, confidence_B_to_A))

    confidence_df = pd.DataFrame(confidence_values, columns=["단어쌍", "신뢰도(A→B)", "신뢰도(B→A)"])

    # CSV 저장
    confidence_df.to_csv("명사_신뢰도.csv", index=False, encoding="utf-8-sig")
#------------------------------------------------------------------------------
def predict_top_five(user_input):
    data = pd.read_csv("명사_신뢰도.csv")
    data_A_B = []
    data_A_B_index = []
    data_B_A = []
    data_B_A_index = []
    # 이렇게 2개로 나눈 이유는
    # "('소리', '너')",0.04672897196261682,0.02262443438914027
    # "('너', '소리')",0.01809954751131222,0.037383177570093455
    # 이상하게 다르네 이러면 무시를 못하지

    for i in range(len(data["단어쌍"])):
        # 이상하게 1.0이 넘는게 있다 썅
        # if data["신뢰도(A→B)"][i] != 1 and data["신뢰도(A→B)"][i] >= 0.7 and data["신뢰도(A→B)"][i] != 2.0:
            data_A_B_index.append(data["신뢰도(A→B)"][i])
            data_A_B.append(data["단어쌍"][i].split("'")) 
        # if data["신뢰도(B→A)"][i] != 1 and data["신뢰도(B→A)"][i] >= 0.7 and data["신뢰도(B→A)"][i] != 2.0:
            data_B_A_index.append(data["신뢰도(B→A)"][i])
            data_B_A.append(data["단어쌍"][i].split("'"))
    # filtered_nouns = list(set(filtered_nouns)
    # 내가 이거를 잘 처리를 안해서 숫자가 1.0을 넘어가는게 있는건가?
    # 썅 이거 때문이구만

    data_list = []
    data_index= []
    temp_list = []

    for i in range(len(data_A_B)):
        if data_A_B[i][1] == user_input:
            for i_2 in np.arange(0.99, 0, -0.01):  # 0.5 이상만
                if data_A_B_index[i] >= i_2 and data_A_B_index[i] != 1.0:
                    temp_list.append((data_A_B[i][3], data_A_B_index[i]))
                    break

    for i in range(len(data_B_A)):
        if data_B_A[i][1] == user_input:
            for i_2 in np.arange(0.99, 0, -0.01):
                if data_B_A_index[i] >= i_2 and data_B_A_index[i] != 1.0:
                    temp_list.append((data_B_A[i][3], data_B_A_index[i]))
                    break


    # 1. 신뢰도 기준으로 내림차순 정렬
    sorted_temp = sorted(temp_list, key=lambda x: x[1], reverse=True)

    # 2. 상위 5개만 추출
    top_5_temp = sorted_temp[:5]

    # 3. 다시 temp_list에 저장
    temp_list = top_5_temp

    # 분리해서 저장
    for item in temp_list:
        data_list.append(item[0])
        data_index.append(item[1])

    for word, score in zip(data_list, data_index):
        print(word, score)

    