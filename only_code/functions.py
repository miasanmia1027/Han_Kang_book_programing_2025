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

    stopwords = {}

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

        # filtered_nouns = list(set(filtered_nouns))  # 중복 제거-> 한 문장에 여러개의 것이 있더라도 무시
             #나는 이게 굳이 있어야 하나 싶기는 함

        tokenized_sentences.append({"index": idx, "num_nouns": len(filtered_nouns), "nouns": filtered_nouns})
        
        # 모든 단어를 리스트에 추가
        filtered_words.extend(filtered_nouns)

    # 결과 저장 (불용어 처리된 데이터 저장)
    with open("nouns_lines.pkl", "wb") as f:
        pickle.dump(tokenized_sentences, f)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def nouns_지지도(pkl_file):
    with open("nouns_lines.pkl", "rb") as f:
        sentences = pickle.load(f)

    word_pairs = defaultdict(int)# 처음 등장하는 단어 쌍(pair)도 안전하게 카운트할 수 있음
    # 이따가  word_pairs[pair] += 1 이거 떄문
    # 일단 여기에 쌍으로 저장을 함

    total_sentences = len(sentences)


    for entry in sentences:
        nouns = entry["nouns"]
        for pair in combinations(nouns, 2):  # 단어 쌍 생성
            word_pairs[pair] += 1

    # 지지도 계산
    support_values = {}  # 빈 딕셔너리 생성

    for pair, count in word_pairs.items():
        support_values[pair] = count / total_sentences  # 지지도 계산 후 딕셔너리에 추가

    # 여기서 pair은 단어 쌍이고
    # 여기서 count는 등장 횟수이다.
    # 즉 특정 단어 쌍 == 등장횟수/전체문장수

    df = pd.DataFrame(support_values.items(), columns=["단어쌍", "지지도"])
    df = df.sort_values(by="지지도", ascending=False)

    df.to_csv("명사_지지도.csv", index=False, encoding="utf-8-sig")
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
def nouns_향상도(pkl_file):
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)

    # 명사 등장 횟수 및 조합 빈도 계산
    word_counts = defaultdict(int)
    total_sentences = len(sentences)

    for entry in sentences:
        nouns = entry["nouns"]
        for word in nouns:
            word_counts[word] += 1

    # 신뢰도 불러오기
    confidence_df = pd.read_csv("명사_신뢰도.csv")
    lift_values = []

    for index, row in confidence_df.iterrows():
        pair = eval(row["단어쌍"])  # 문자열을 튜플로 변환
        confidence_A_to_B = row["신뢰도(A→B)"]# 이 신뢰도는 왜 가져왔을까?
        confidence_B_to_A = row["신뢰도(B→A)"]# 이 신뢰도는 왜 가져왔을까?

        A, B = pair

        support_B = word_counts.get(B, 0) / total_sentences  # B의 지지도
        support_A = word_counts.get(A, 0) / total_sentences  # A의 지지도
    # 이거를 역으로 봐야지 이해가 되는구나

        lift_A_to_B = confidence_A_to_B / support_B if support_B > 0 else 0
        lift_B_to_A = confidence_B_to_A / support_A if support_A > 0 else 0

        lift_values.append((pair, lift_A_to_B, lift_B_to_A))

    lift_df = pd.DataFrame(lift_values, columns=["단어쌍", "향상도(A→B)", "향상도(B→A)"])

    # CSV 저장
    lift_df.to_csv("명사_향상도.csv", index=False, encoding="utf-8-sig")
#------------------------------------------------------------------------------
def combin_three_nouns():
    # 🔹 CSV 파일 불러오기
    support_df = pd.read_csv("명사_지지도.csv")
    confidence_df = pd.read_csv("명사_신뢰도.csv")
    lift_df = pd.read_csv("명사_향상도.csv")

    # 🔹 가중치 설정 (추천: Lift와 Confidence를 더 강조)
    weight_support = 0.2
    weight_confidence = 0.4
    weight_lift = 0.4
    def predict_next_words(target_word, top_n=10):
        # 🔹 단어 쌍이 포함된 데이터 필터링
        filtered_confidence = confidence_df[confidence_df["단어쌍"].str.contains(target_word)]
        filtered_lift = lift_df[lift_df["단어쌍"].str.contains(target_word)]
        filtered_support = support_df[support_df["단어쌍"].str.contains(target_word)]

        scores = {}

        # 🔹 필터링된 데이터에서 연관 단어 찾기
        for _, row in filtered_confidence.iterrows():  
            pair = eval(row["단어쌍"])  # 문자열을 튜플로 변환
            confidence_A_to_B = row["신뢰도(A→B)"]
            confidence_B_to_A = row["신뢰도(B→A)"]

            # 🔹 대상 단어와 짝이 된 단어 찾기
            other_word = pair[1] if pair[0] == target_word else pair[0]

            # 🔹 특정 단어 쌍에 해당하는 향상도 값 찾기
            lift_row = filtered_lift[filtered_lift["단어쌍"] == str(pair)]
            if not lift_row.empty:
                lift_A_to_B = lift_row["향상도(A→B)"].values[0]
                lift_B_to_A = lift_row["향상도(B→A)"].values[0]
            else:
                lift_A_to_B = lift_B_to_A = 0  # 데이터가 없으면 기본값 0

            # 🔹 특정 단어 쌍에 해당하는 지지도 값 찾기
            support_row = filtered_support[filtered_support["단어쌍"] == str(pair)]
            support_AB = support_row["지지도"].values[0] if not support_row.empty else 0

            # 🔹 최종 점수 계산
            score = (weight_support * support_AB) + \
                    (weight_confidence * max(confidence_A_to_B, confidence_B_to_A)) + \
                    (weight_lift * max(lift_A_to_B, lift_B_to_A))

            scores[other_word] = score

        # 🔹 상위 N개 단어 정렬하여 반환
        sorted_words = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
        
        return sorted_words
    # 🔹 사용자 입력 받기
    def combin_three():
        target_word = input("예측할 단어를 입력하세요: ")
        top_predictions = predict_next_words(target_word)

        # 🔹 결과 출력
        print(f"\n'{target_word}' 다음에 나올 가능성이 높은 단어 Top 10:")
        for word, score in top_predictions:
            print(f"{word}: {score:.4f}")

    # 실행
    combin_three()
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
def book_to_pkl_morphs(data_file):
    file_path = f"{data_file}"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 줄 단위로 읽기

    stopwords = {}

    okt = Okt()
    tokenized_sentences = []
    filtered_words = []  # 불용어 제거된 단어만 저장할 리스트

    for idx, line in enumerate(lines):
        morphs = okt.morphs(line)  # 모든 형태소 추출

        # 🔹 불용어 제거 (컴프리헨션 없이 작성)
        filtered_morphs = []
        for word in morphs:
            if word not in stopwords:
                filtered_morphs.append(word)


        tokenized_sentences.append({
            "index": idx,
            "num_morphs": len(filtered_morphs),
            "morphs": filtered_morphs
        })
        
        # 모든 단어를 리스트에 추가
        filtered_words.extend(filtered_morphs)

    # 결과 저장 (불용어 처리된 데이터 저장)
    with open("morphs_lines.pkl", "wb") as f:
        pickle.dump(tokenized_sentences, f)
#------------------------------------------------------------------------------
def morphs_지지도(pkl_file):
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)  # 형태소 분석된 문장 데이터 로드

    word_pairs = defaultdict(int)  # 단어 쌍 카운트 딕셔너리
    total_sentences = len(sentences)  # 전체 문장 수

    for entry in sentences:
        morphs = entry["morphs"]  # 형태소 리스트 가져오기
        for pair in combinations(morphs, 2):  # 형태소 쌍 생성
            word_pairs[pair] += 1

    # 🔹 지지도 계산
    support_values = {pair: count / total_sentences for pair, count in word_pairs.items()}

    # 🔹 데이터프레임 생성 및 저장
    df = pd.DataFrame(support_values.items(), columns=["형태소쌍", "지지도"])
    df = df.sort_values(by="지지도", ascending=False)

    df.to_csv("형태소_지지도.csv", index=False, encoding="utf-8-sig")
#------------------------------------------------------------------------------
def morphs_신뢰도(pkl_file):
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)  # 형태소 분석된 문장 데이터 로드

    # 🔹 형태소 등장 횟수 및 조합 빈도 계산
    word_pairs = defaultdict(int)  # 형태소 쌍 등장 횟수 저장
    word_counts = defaultdict(int)  # 개별 형태소 등장 횟수 저장
    total_sentences = len(sentences)  # 전체 문장 수

    for entry in sentences:
        morphs = entry["morphs"]  # 형태소 리스트 가져오기

        # 🔹 개별 형태소 등장 횟수 계산
        for word in morphs:
            word_counts[word] += 1

        # 🔹 형태소 쌍 등장 횟수 계산
        for pair in combinations(morphs, 2):
            word_pairs[pair] += 1

    # 🔹 신뢰도 계산
    confidence_values = []
    for pair, support_AB in word_pairs.items():
        A, B = pair

        support_A = word_counts.get(A, 0)  # A의 등장 횟수
        support_B = word_counts.get(B, 0)  # B의 등장 횟수

        # A → B 신뢰도
        confidence_A_to_B = support_AB / support_A if support_A > 0 else 0
        # B → A 신뢰도
        confidence_B_to_A = support_AB / support_B if support_B > 0 else 0

        confidence_values.append((pair, confidence_A_to_B, confidence_B_to_A))

    # 🔹 데이터프레임 생성 및 저장
    confidence_df = pd.DataFrame(confidence_values, columns=["형태소쌍", "신뢰도(A→B)", "신뢰도(B→A)"])
    confidence_df.to_csv("형태소_신뢰도.csv", index=False, encoding="utf-8-sig")
#------------------------------------------------------------------------------
def morphs_향상도(pkl_file):
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)  # 형태소 분석된 문장 데이터 로드

    # 🔹 형태소 등장 횟수 계산
    word_counts = defaultdict(int)  # 개별 형태소 등장 횟수 저장
    total_sentences = len(sentences)  # 전체 문장 수

    for entry in sentences:
        morphs = entry["morphs"]  # 형태소 리스트 가져오기
        for word in morphs:
            word_counts[word] += 1  # 개별 형태소 등장 횟수 증가

    # 🔹 신뢰도 데이터 불러오기 (형태소_신뢰도.csv)
    confidence_df = pd.read_csv("형태소_신뢰도.csv")
    lift_values = []

    for index, row in confidence_df.iterrows():
        pair = eval(row["형태소쌍"])  # 문자열을 튜플로 변환
        confidence_A_to_B = row["신뢰도(A→B)"]
        confidence_B_to_A = row["신뢰도(B→A)"]

        A, B = pair

        # 🔹 개별 형태소(A, B)의 지지도 계산
        support_B = word_counts.get(B, 0) / total_sentences  # B의 지지도
        support_A = word_counts.get(A, 0) / total_sentences  # A의 지지도

        # 🔹 향상도(A → B, B → A) 계산
        lift_A_to_B = confidence_A_to_B / support_B if support_B > 0 else 0
        lift_B_to_A = confidence_B_to_A / support_A if support_A > 0 else 0

        lift_values.append((pair, lift_A_to_B, lift_B_to_A))

    # 🔹 데이터프레임 생성 및 저장
    lift_df = pd.DataFrame(lift_values, columns=["형태소쌍", "향상도(A→B)", "향상도(B→A)"])
    lift_df.to_csv("형태소_향상도.csv", index=False, encoding="utf-8-sig")
#------------------------------------------------------------------------------
def combin_three_morphs():
    # 🔹 CSV 파일 불러오기 (형태소 기반 데이터)
    support_df = pd.read_csv("형태소_지지도.csv")
    confidence_df = pd.read_csv("형태소_신뢰도.csv")
    lift_df = pd.read_csv("형태소_향상도.csv")

    # 🔹 가중치 설정 (Lift와 Confidence를 더 강조)
    weight_support = 0.2
    weight_confidence = 0.4
    weight_lift = 0.4

    def predict_next_words(target_word, top_n=10):
        # 🔹 특정 단어가 포함된 데이터 필터링
        filtered_confidence = confidence_df[confidence_df["형태소쌍"].str.contains(target_word)]
        filtered_lift = lift_df[lift_df["형태소쌍"].str.contains(target_word)]
        filtered_support = support_df[support_df["형태소쌍"].str.contains(target_word)]

        scores = {}

        # 🔹 필터링된 데이터에서 연관된 형태소 찾기
        for _, row in filtered_confidence.iterrows():
            pair = eval(row["형태소쌍"])  # 문자열을 튜플로 변환
            confidence_A_to_B = row["신뢰도(A→B)"]
            confidence_B_to_A = row["신뢰도(B→A)"]

            # 🔹 대상 단어와 짝이 된 형태소 찾기
            other_word = pair[1] if pair[0] == target_word else pair[0]

            # 🔹 특정 형태소 쌍에 대한 향상도 값 찾기
            lift_row = filtered_lift[filtered_lift["형태소쌍"] == str(pair)]
            if not lift_row.empty:
                lift_A_to_B = lift_row["향상도(A→B)"].values[0]
                lift_B_to_A = lift_row["향상도(B→A)"].values[0]
            else:
                lift_A_to_B = lift_B_to_A = 0  # 데이터가 없으면 기본값 0

            # 🔹 특정 형태소 쌍에 대한 지지도 값 찾기
            support_row = filtered_support[filtered_support["형태소쌍"] == str(pair)]
            support_AB = support_row["지지도"].values[0] if not support_row.empty else 0

            # 🔹 최종 점수 계산
            score = (weight_support * support_AB) + \
                    (weight_confidence * max(confidence_A_to_B, confidence_B_to_A)) + \
                    (weight_lift * max(lift_A_to_B, lift_B_to_A))

            scores[other_word] = score

        # 🔹 상위 N개 단어 정렬하여 반환
        sorted_words = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
        
        return sorted_words

    # 🔹 사용자 입력 받기
    target_word = input("예측할 형태소를 입력하세요: ")
    top_predictions = predict_next_words(target_word)

    # 🔹 결과 출력
    print(f"\n'{target_word}' 다음에 나올 가능성이 높은 형태소 Top 10:")
    for word, score in top_predictions:
        print(f"{word}: {score:.4f}")