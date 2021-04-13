
# 3D공간상의 rigid변환을 이용한 자동 위치 복원<BR>(Automatic position recovery using rigid transform<BR> on  3D space)

2020년도 초반에 차량안의 DSM(Driver status monitoring)개발하고 있는 제품의 Head position accuracy를 검증하는 장비를 개발하였다. DSM 제품은 이 차량안에 사람의 얼굴을 모니터링하여, 졸음운전을 하는지, 전방을 주시하지 않는지 등의 알림을 주는 기능이다. 
검증 장비를 개발시에 디지털 카메라를 사용한 3D 측정장비(AICON의 DPA)를 이용하여, 3차원 모델의 위치를 추출할 수 있었다. 주로 제품의 위치와 마네킹의 위치에 대한 Ground Truth를 추출하기 위해, rigid transform을 사용하였다.
하지만 이 GT를 생성하는 과정에서 12가지 얼굴의 위치와 여러 제품들의 위치, DPA장비로 누락되는 마커와 장비 내부 변경으로 인한 마커 손상으로 인해, 많은 시간과 휴먼 계산 오류 등으로 정확도가 감소할 가능성이 존재하였다.
이에 이 문제를 해결하기 위해, 단편적인 데이터를 재사용하고, 휴먼 계산 오류를 막기위해, rigid변환을 이용한 자동위치 복원 기능을 구현해 보았다. 






## 참고문헌
1. http://nghiaho.com/?page_id=671
2.  [https://en.wikipedia.org/wiki/Kabsch_algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm)
3.  “Least-Squares Fitting of Two 3-D Point Sets”, Arun, K. S. and Huang, T. S. and Blostein, S. D, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987


## Rigid Transform

You can export the current file by clicking **Export to disk** in the menu. You can choose to export the file as plain Markdown, as HTML using a Handlebars template or as a PDF.





$$
error = \displaystyle\sum\limits_{i=1}^N \left|\left|RP_A^i + T -P_B^i \right|\right|^2
$$

$$
\mathbf P_A = R * P_B + T
$$
