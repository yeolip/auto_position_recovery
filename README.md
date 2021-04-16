
# 3D공간상의 rigid변환을 이용한 자동 위치 복원<BR>(Automatic position recovery using rigid transform<BR> on  3D space)

2020년도 초반에 차량안의 DSM(Driver status monitoring)개발하고 있는 제품의 Head position accuracy를 검증하는 장비를 개발하였다. 
DSM 제품은 이 차량안에 사람의 얼굴을 모니터링하여, 졸음운전을 하는지, 전방을 주시하지 않는지 등의 알림을 주는 기능이다. 
이에 사람 얼굴의 위치 결과를 전달해주는데, 마네킹의 위치 결과와 GroundTruth 데이터의 차이로 알고리즘 spec 및 accuracy를 판단하고 있다.
이 GT데이터를 계산하기위해, 3D 측정장비(AICON의 DPA)에 마커를 붙이고 3차원 모델의 위치를 추출이 필요하다.
12가지 마네킹 위치와 눈의 위치를 측정해야하고 눈에 마커를 붙이고, DSM 제품에도 마커를 붙이는 등등의 작업이 필요했었다.
하지만 장비 내부의 변경 및 위치 이동으로 마커가 사라지거나 인식되지 않는 경우가 발생하여 한번에 데이터를 다 취득하기 어려운 경우가 많았다.
결국에는 누락부분이나 별도의 위치 추정을 계산하는 부분을 담당자가 수동으로 계산해 주어야 했었다.
이에 Human error나 계산 과정의 실수가 발생가능하며, 이상 유무도 판단하기 어려웠었기에, 
이에 이 문제를 해결하기 위해, 단편적인 데이터를 재사용하고, 휴먼 계산 오류를 막기위해, 
Rigid Transform변환을 이용한 자동위치 복원 기능을 구현해 보았다. 



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
