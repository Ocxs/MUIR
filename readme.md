This is the implementation for our paper "Learning and Fusing Multiple User Interest Representations for Micro-Video and Movie Recommendations."



## Environment
Python 3.6, Tensorflow 1.12

## Dataset
Since the limited space in GitHub, we upload the [Movie data](https://pan.baidu.com/s/1tbocMGlpzqsRE3p_TejOHA)(password: ah6i) and [Micro-video data](https://pan.baidu.com/s/1hQIXiDSeStuP27fHFZM8Cg )(password: jp82) on BaiduYun for running our code. You could run `mkdir Files`, download the dataset, and then put them into `Files/`.

## Usage
- For Micro-video Recommendations, run 
```
cd Video/LIN/LIN_Video;python launcher.py;
cd Video/IIN/IIN_Video;python launcher.py;
cd Video/CIN/CIN_Video;python launcher.py;
cd Video/NIN/NIN_Video;python launcher.py;
```
Finally, you can perform late fusion to get final prediction of MUIR for micro-video recommendations.

- For Movie Recommedation, run 
```
cd Movie/LIN/LIN_Movie;python launcher.py;
cd Movie/IIN/IIN_Movie;python launcher.py;
cd Movie/CIN/CIN_Movie;python launcher.py;
cd Movie/NIN/NIN_Movie;python launcher.py;
```
Finally, you can perform late fusion to get final prediction of MUIR for movie recommendations.