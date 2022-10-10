1. 라이브러리 경로를 찾지 못할때
2022.10.10
1)sys라이브러리사용(매번 실행할때마다 추가 필요)
import sys
print(sys.path)
sys.path.append("지정하고자 하는 경로")
2)라이브러리 경로 변경
__name__=="__main__"은 항상 절대경로로 하기
나머지는 절대경로 안되면 상대경로로 변경.

========================================
1. When the library paths are not found

1) Use the sys module(each time you run main.py, you have to add the following codes)
import sys
print(sys.path)
sys.path.append("path you want to specify")
2.edit the module path
libraries must be imported as absolute paths in main modules (__name__=="__main__"), not relative paths.
both are available when it is not the case.
However,if these types of modul errors occur, then convert a relative path to an absolute path.