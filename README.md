# DB-ECA-and-LSTR
DB-EAC and LSTR: DBnet based seal text detection and Lightweight Seal Text Recognition<br/>
# Introduction
This project mainly serves the detection and recognition of seals. DB-ECA can detect the position of seals and LSTR is used to recognize the text of seals.<br/>
The code relies on the Paddle platform for implementation, and detailed deployment scenarios can be viewed at：https://github.com/PaddlePaddle/PaddleOCR<br/>
The fastest way to start is to replace our file based on PaddleOCR and run it<br/>
## Installation
> requirement<br/>
>> +Python3.7<br/>
>> +CUDA11.6<br/>
>> +Cudnn 8.4.0<br/>
>_setup_<br/>
>>git clone https://github.com/PaddlePaddle/PaddleOCR.git<br/>
>> pip install paddlepaddle-gpu<br/>
>> pip install -r requirements.txt<br/>

Replace the neck part with DB-ECA.py/LSTR in the config file and run it
