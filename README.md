# MatSciBERT
개인 재구현 코드 공유

과제진행 당시 짰던(썼던) 코드입니다. 

## 수집
원 저자 코드에는 해당 부분이 없어, 수집(abstract만 진행)은 [elapsy api](https://github.com/ElsevierDev/elsapy/blob/master/exampleProg.py) 기반으로 직접 진행했습니다. 사용 코드는 `./script/get_abs_split.py`에 있습니다. 수집한 텍스트는 총 80Mb 분량입니다.

#### 특이사항
- [elapsy api](https://github.com/ElsevierDev/elsapy/blob/master/exampleProg.py)에는 key당 수집 제한이 있습니다. 여러 키를 생성해 동시에 사용하도록 하여 우회적으로 접근하기는 하였지만, 계정당 key는 10개로 정해져 있어 완전한 방법은 아닙니다.
- 특수문자 처리를 위해, 원 코드에서는 추가적인 전처리를 진행합니다. 코드는 `./script/normalize_txt.py`에 해당합니다.

## 학습
학습 코드는 원 저자 코드를 참조했습니다. HuggingFace 기반이라 편하게 보실 듯합니다. 여기에서는 `./script/pretrain.py`에 해당합니다.
