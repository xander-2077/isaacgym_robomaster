# ISAAC GYM ROBOMASTER

训练：
```
python train.py task=Robomaster capture_video=True
```

推理：
```
python train.py task=Robomaster capture_video=True test=True checkpoint='./runs/Robomaster_19-14-58-10/nn/Robomaster.pth'
```