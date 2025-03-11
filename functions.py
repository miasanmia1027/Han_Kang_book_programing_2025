#-----------------------------------------------------------------
# rawdata_to_cleandata(data_file) // 원본 데이터를 공백 삭제후 살리기 과정 // data_file == 원문(공백 정제 과정을 거치지 않은 상태)
# book_to_pkl_nouns(data_file) // 지지도 계산이 가능하게 일단 문장 단위 요소수와 문장을 요소로 나누어 pkl파일로 저장 // data_file == 문장을 지지도 분석 전에 원활하게 분석을 가능케 한다.
# make_nouns_pkl_to_지지도(pkl_file) // pkl_file ==  명사 pkl파일을 받아서 지지도를 보여주는 코드


# def filtering(csv_file,amount) // csv_file ==  이거는 지지도 분석을 완료한 상태의 csv파일 amount == 기준을 정해주는것
#-----------------------------------------------------------------

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

def book_to_pkl_nouns(data_file):
    file_path = f"{data_file}"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 줄 단위로 읽기

    # 형태소 분석 (줄 단위 명사 추출)
    okt = Okt()
    tokenized_sentences = []
    for idx, line in enumerate(lines):
        nouns = okt.nouns(line)  # 명사만 추출
        nouns = list(set(nouns))  # 중복 제거 -> 특정 언어가 과도하게 되지 않게  -> 근데 나는 굳이 필요 없다고 생각을 한다.
        tokenized_sentences.append({"index": idx, "num_nouns": len(nouns), "nouns": nouns})  # 줄 정보 저장 -> 문장 단위 분의 뭐시기가 가능하기 떄문에

    with open("nouns_lines.pkl", "wb") as f:
        pickle.dump(tokenized_sentences, f)


def nouns_지지도(pkl_file):
    with open(f"{pkl_file}", "rb") as f:
        sentences = pickle.load(f)

    word_pairs = defaultdict(int)
    total_sentences = len(sentences)

    for entry in sentences:
        nouns = entry["nouns"]
        for pair in combinations(nouns, 2):  # 단어 쌍 생성
            word_pairs[pair] += 1

    # 지지도 계산
    support_values = {pair: count / total_sentences for pair, count in word_pairs.items()}

    df = pd.DataFrame(support_values.items(), columns=["단어쌍", "지지도"])
    df = df.sort_values(by="지지도", ascending=False)

    df.to_csv("명사_지지도.csv", index=False, encoding="utf-8-sig")



def nouns_filtering(csv_file,amount):
    ahn = amount

    # 저장된 CSV 파일 불러오기
    file_path = f"{csv_file}"  # 기존 연관성 분석 결과 파일 경로
    df = pd.read_csv(file_path)

    # 지지도 0.5 이상 필터링
    df_filtered = df[df["지지도"] >= ahn]

    # 필터링된 결과를 새로운 CSV 파일로 저장
    filtered_file_path = f"명사_지지도{ahn}이상.csv"
    df_filtered.to_csv(filtered_file_path, index=False, encoding="utf-8-sig")


def book_to_pkl_all(data_file):
    # 파일 로드
    file_path = f"{data_file}"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 줄 단위로 읽기

    okt = Okt()
    tokenized_sentences = []
    for idx, line in enumerate(lines):
        morphs = okt.morphs(line)  # 형태소 추출
        morphs = list(set(morphs))  # 중복 제거
        tokenized_sentences.append({"index": idx, "num_morphs": len(morphs), "morphs": morphs})  # 줄 정보 저장

    with open("all_lines.pkl", "wb") as f:
        pickle.dump(tokenized_sentences, f)

def all_지지도(pkl_file):
    with open(f"{pkl_file}", "rb") as f:
        sentences = pickle.load(f)

    # 연관성 분석 (형태소 쌍의 등장 횟수)
    word_pairs = defaultdict(int)
    total_sentences = len(sentences)

    for entry in sentences:
        morphs = entry["morphs"]  # 형태소 리스트 가져오기
        morphs = [m for m in morphs if m.strip() and m not in [",", " "]]  # 공백 및 쉼표 제거
        for pair in combinations(morphs, 2):  # 형태소 쌍 생성
            word_pairs[pair] += 1

    # 지지도 계산
    support_values = {pair: count / total_sentences for pair, count in word_pairs.items()}

    # 데이터 정리
    df = pd.DataFrame(support_values.items(), columns=["형태소쌍", "지지도"])           
    df = df.sort_values(by="지지도", ascending=False)

    # CSV 파일로 저장
    df.to_csv("all_지지도.csv", index=False, encoding="utf-8-sig")

def all_filtering(csv_file,amount):
    ahn = amount

    # 저장된 CSV 파일 불러오기
    file_path = f"{csv_file}"  # 기존 연관성 분석 결과 파일 경로
    df = pd.read_csv(file_path)

    # 지지도 0.5 이상 필터링
    df_filtered = df[df["지지도"] >= ahn]

    # 필터링된 결과를 새로운 CSV 파일로 저장
    filtered_file_path = f"all_지지도{ahn}이상.csv"
    df_filtered.to_csv(filtered_file_path, index=False, encoding="utf-8-sig")













def nouns_신뢰도(pkl_file):
    """
    명사 지지도 PKL 파일에서 신뢰도를 계산하여 저장하는 함수.

    :param pkl_file: 명사 지지도 PKL 파일 경로
    """
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)

    # 명사 등장 횟수 및 조합 빈도 계산
    word_pairs = defaultdict(int)
    word_counts = defaultdict(int)
    total_sentences = len(sentences)

    for entry in sentences:
        nouns = entry["nouns"]  # 명사 리스트 가져오기

        # 개별 명사 등장 횟수 계산
        for word in nouns:
            word_counts[word] += 1

        # 명사 쌍 등장 횟수 계산
        for pair in combinations(nouns, 2):
            word_pairs[pair] += 1

    # 신뢰도 계산
    confidence_values = []
    for pair, support_AB in word_pairs.items():
        A, B = pair

        support_A = word_counts.get(A, 0) / total_sentences  # A의 지지도
        support_B = word_counts.get(B, 0) / total_sentences  # B의 지지도

        confidence_A_to_B = support_AB / word_counts[A] if word_counts[A] > 0 else 0
        confidence_B_to_A = support_AB / word_counts[B] if word_counts[B] > 0 else 0

        confidence_values.append((pair, confidence_A_to_B, confidence_B_to_A))

    confidence_df = pd.DataFrame(confidence_values, columns=["단어쌍", "신뢰도(A→B)", "신뢰도(B→A)"])
    
    # CSV 저장
    confidence_df.to_csv("명사_신뢰도.csv", index=False, encoding="utf-8-sig")


def nouns_향상도(pkl_file):
    """
    명사 신뢰도 PKL 파일에서 향상도를 계산하여 저장하는 함수.

    :param pkl_file: 명사 지지도 PKL 파일 경로
    """
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
        confidence_A_to_B = row["신뢰도(A→B)"]
        confidence_B_to_A = row["신뢰도(B→A)"]

        A, B = pair

        support_B = word_counts.get(B, 0) / total_sentences  # B의 지지도
        support_A = word_counts.get(A, 0) / total_sentences  # A의 지지도

        lift_A_to_B = confidence_A_to_B / support_B if support_B > 0 else 0
        lift_B_to_A = confidence_B_to_A / support_A if support_A > 0 else 0

        lift_values.append((pair, lift_A_to_B, lift_B_to_A))

    lift_df = pd.DataFrame(lift_values, columns=["단어쌍", "향상도(A→B)", "향상도(B→A)"])
    
    # CSV 저장
    lift_df.to_csv("명사_향상도.csv", index=False, encoding="utf-8-sig")



















def all_신뢰도(pkl_file):
    """
    형태소 지지도 PKL 파일에서 신뢰도를 계산하여 저장하는 함수.

    :param pkl_file: 형태소 지지도 PKL 파일 경로
    """
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)

    # 형태소 등장 횟수 및 조합 빈도 계산
    word_pairs = defaultdict(int)
    word_counts = defaultdict(int)
    total_sentences = len(sentences)

    for entry in sentences:
        morphs = entry["morphs"]  # 형태소 리스트 가져오기
        morphs = [m for m in morphs if m.strip() and m not in [",", " "]]  # 공백 및 쉼표 제거

        # 개별 형태소 등장 횟수 계산
        for word in morphs:
            word_counts[word] += 1

        # 형태소 쌍 등장 횟수 계산
        for pair in combinations(morphs, 2):
            word_pairs[pair] += 1

    # 신뢰도 계산
    confidence_values = []
    for pair, support_AB in word_pairs.items():
        A, B = pair

        support_A = word_counts.get(A, 0) / total_sentences  # A의 지지도
        support_B = word_counts.get(B, 0) / total_sentences  # B의 지지도

        confidence_A_to_B = support_AB / word_counts[A] if word_counts[A] > 0 else 0
        confidence_B_to_A = support_AB / word_counts[B] if word_counts[B] > 0 else 0

        confidence_values.append((pair, confidence_A_to_B, confidence_B_to_A))

    confidence_df = pd.DataFrame(confidence_values, columns=["형태소쌍", "신뢰도(A→B)", "신뢰도(B→A)"])
    
    # CSV 저장
    confidence_df.to_csv("형태소_신뢰도.csv", index=False, encoding="utf-8-sig")


def all_향상도(pkl_file):
    """
    형태소 신뢰도 PKL 파일에서 향상도를 계산하여 저장하는 함수.

    :param pkl_file: 형태소 지지도 PKL 파일 경로
    """
    with open(pkl_file, "rb") as f:
        sentences = pickle.load(f)

    # 형태소 등장 횟수 및 조합 빈도 계산
    word_counts = defaultdict(int)
    total_sentences = len(sentences)

    for entry in sentences:
        morphs = entry["morphs"]
        morphs = [m for m in morphs if m.strip() and m not in [",", " "]]  # 공백 및 쉼표 제거
        for word in morphs:
            word_counts[word] += 1

    # 신뢰도 불러오기
    confidence_df = pd.read_csv("형태소_신뢰도.csv")
    lift_values = []

    for index, row in confidence_df.iterrows():
        pair = eval(row["형태소쌍"])  # 문자열을 튜플로 변환
        confidence_A_to_B = row["신뢰도(A→B)"]
        confidence_B_to_A = row["신뢰도(B→A)"]

        A, B = pair

        support_B = word_counts.get(B, 0) / total_sentences  # B의 지지도
        support_A = word_counts.get(A, 0) / total_sentences  # A의 지지도

        lift_A_to_B = confidence_A_to_B / support_B if support_B > 0 else 0
        lift_B_to_A = confidence_B_to_A / support_A if support_A > 0 else 0

        lift_values.append((pair, lift_A_to_B, lift_B_to_A))

    lift_df = pd.DataFrame(lift_values, columns=["형태소쌍", "향상도(A→B)", "향상도(B→A)"])
    
    # CSV 저장
    lift_df.to_csv("형태소_향상도.csv", index=False, encoding="utf-8-sig")