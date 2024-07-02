# ISAAC GYM ROBOMASTER

训练：
```
python train.py task=RobomasterFull capture_video=True
```

推理：
```
python train.py task=RobomasterFull capture_video=True test=True checkpoint='./runs/RobomasterFull_19-14-58-10/nn/RobomasterFull.pth'
```