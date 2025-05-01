# 텍스트 마이닝에서 배운 방식을 사용하여
- 한강 작가 끼리의 분석
- 한강작가와 다른 작가의 구분을 하는 방식을 채택을 해야 겠다

# 내가 해야 하는 작업들
2. 현재 있는 코드가 어떤건지 이해하기
- 가장 많이 사용한 단어를 기준으로 해라
- 내가 한 임베딩을 사용하는 방법 단어간의 간격을 비교하는 방식
3. 택스트 마이닝에서 배운거 적용하기
- 일단 이정도

# 현재 데이터를 사용하여 어떻게든 백터화를 진행을 했다
- 그러면 이제 다음에 어떤 작업을 수행을 해야할까?
- 상위 단어들을 추출을 해서 이 단어를 기준으로 쌍끼리의 거리를 측정을 해서 
-> 어떤게 많이 차이가 나는지 확인을 하는 코드를 짜볼까?
- P = 현재 단어가 단어 쌍 끼리의 토큰이 아니다
### 솔류션
```
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
```
- 이렇게 쌍끼리로 바꾸고 그 다음에 백터화를 진행을 할까?
1. 쌍으로 만든다
2. 이미 학습된 모델로 임베딩을 한다
- 이 임베딩 과정이 중요할거 같은데
- 이미 학습이 되어 있는 것으로 임베딩을 하면 서로 다른 간격이 음
3. 서로 같이 나오면 연관성의 거리를 줄이는 코드를 만들어야하나?






# 목표 ==책을 통해 작가를 알아간다














## 나중에
- 이게 내가 지금 
```
model = KeyedVectors.load("no_han_book/cc.ko.300.kv")
```
이미 학습되어 있는 모델을 사용중인데 이거를 쓰면 문장의 의미가 없어지고
학습이 되어 있는 모델만이 사용이 되니 이게 약간 아쉽다
- 나중에는 없는거로 시도를 해봐야겠다



```
방법 3: 임베딩 공간에서의 단어 분포 비교 (고급)

아이디어: FastText 같은 모델이 단어를 벡터 공간에 배치하는데, 작가 A가 주로 사용하는 단어들의 벡터 분포와 작가 B 그룹이 사용하는 단어들의 벡터 분포가 어떻게 다른지 시각화하거나 통계적으로 비교합니다.
과정:
각 그룹에서 자주 사용되는 명사들의 벡터를 추출합니다.
PCA나 t-SNE 같은 차원 축소 기법을 사용하여 2D 또는 3D 공간에 시각화합니다.
작가 A의 단어들이 특정 영역에 밀집되어 있는지, 작가 B 그룹과 다른 군집을 형성하는지 등을 관찰합니다.
장점: 전체적인 의미론적 사용 패턴 차이를 직관적으로 볼 수 있습니다.
단점: 해석이 주관적일 수 있고, 구현이 다소 복잡합니다.
```