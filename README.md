
<H2>심층 신경망 구현 + 임의의 데이터 (새,포유류,기타) 분류</H2>  
<b>LogisticReg-2017111299.py</b><br/>
<br/><br/>

<H2>심층 신경망 구현 with MNIST Dataset</H2>  
<b>MNIST_LogisticReg.py</b><br/>
기능 : MNIST에서 제공하는 Dataset으로 심층신경망 구현<br/>
<br/><br/>

<H2>합성곱 신경망 구현 with MNIST Dataset</H2>  
<b>MNIST_CNN.py</b><br/>
기능 : MNIST에서 제공하는 Dataset으로 합성곱신경망 구현<br/>
목표 : 가로 픽셀(28) x 세로 픽셀(28) x 흑백채널(1) 의 3차원 데이터(숫자 손글씨 이미지)를 0~9의 10가지 숫자로 분류합니다.<br/>
특징 :<br/>
* 심층신경망은 Fully connected Layer로만 이루어져 데이터의 입체 정보가 무시되는 단점이 있었는데 이를 해결할 수 있습니다.<br/>
* 은닉층은 Convolution Layer 2층, Fully connected Layer 2층으로 구성했습니다.<br/>
* 두가지 최적화함수(Adam, RMSProb)를 적용해보고 비교합니다.<br/><br/>
<img src="https://user-images.githubusercontent.com/48902155/84871741-dfe63380-b0bb-11ea-970a-453b79225d01.png" width="70%"></img>
<br/><br/>

