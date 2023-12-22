[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13253847&assignment_repo_type=AssignmentRepo)
# MiniTorch Module 1

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1/module1/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py tests/test_module.py tests/test_operators.py project/run_manual.py

```python
PTS = 150
DATASET = minitorch.datasets["Simple"](PTS)
HIDDEN = 2
RATE = 0.5
```
```
Epoch: 10/500, loss: 86.3386013074784, correct: 123
Epoch: 20/500, loss: 66.20298984633504, correct: 137
Epoch: 30/500, loss: 45.93576259318546, correct: 145
Epoch: 40/500, loss: 33.07676464914028, correct: 147
Epoch: 50/500, loss: 28.979624446083754, correct: 142
Epoch: 60/500, loss: 31.193433221947537, correct: 135
Epoch: 70/500, loss: 22.978822035970303, correct: 142
Epoch: 80/500, loss: 17.64104337270025, correct: 148
Epoch: 90/500, loss: 15.704665491309196, correct: 150
Epoch: 100/500, loss: 15.298927009229322, correct: 144
Epoch: 110/500, loss: 19.009427789739277, correct: 142
Epoch: 120/500, loss: 17.697507829872333, correct: 142
Epoch: 130/500, loss: 12.719561100884022, correct: 146
Epoch: 140/500, loss: 10.86231292780951, correct: 150
Epoch: 150/500, loss: 10.050652658243429, correct: 150
Epoch: 160/500, loss: 9.88804905708161, correct: 150
Epoch: 170/500, loss: 12.524362758094625, correct: 144
Epoch: 180/500, loss: 16.320639544881317, correct: 142
Epoch: 190/500, loss: 11.338648569477716, correct: 144
Epoch: 200/500, loss: 9.146856182778814, correct: 149
Epoch: 210/500, loss: 8.112748938884199, correct: 150
Epoch: 220/500, loss: 8.008188984952003, correct: 150
Epoch: 230/500, loss: 8.709542072871884, correct: 148
Epoch: 240/500, loss: 11.189966835328331, correct: 144
Epoch: 250/500, loss: 11.75439512246964, correct: 144
Epoch: 260/500, loss: 9.136015694802053, correct: 144
Epoch: 270/500, loss: 7.501043081766024, correct: 150
Epoch: 280/500, loss: 6.834223082472769, correct: 150
Epoch: 290/500, loss: 6.555535810614546, correct: 150
Epoch: 300/500, loss: 6.775297652496304, correct: 150
Epoch: 310/500, loss: 8.053759505348326, correct: 144
Epoch: 320/500, loss: 10.525199609854184, correct: 144
Epoch: 330/500, loss: 9.765352931495517, correct: 144
Epoch: 340/500, loss: 7.185067716276942, correct: 148
Epoch: 350/500, loss: 5.794189449732173, correct: 150
Epoch: 360/500, loss: 5.200417180831331, correct: 150
Epoch: 370/500, loss: 4.898490908357622, correct: 149
Epoch: 380/500, loss: 4.752190591783464, correct: 149
Epoch: 390/500, loss: 4.690877676454174, correct: 150
Epoch: 400/500, loss: 4.885893522718088, correct: 150
Epoch: 410/500, loss: 5.91829026278255, correct: 149
Epoch: 420/500, loss: 14.067274336177457, correct: 143
Epoch: 430/500, loss: 12.699705938254942, correct: 143
Epoch: 440/500, loss: 7.496539323841025, correct: 144
Epoch: 450/500, loss: 4.4823471915339805, correct: 150
Epoch: 460/500, loss: 4.0426527472143485, correct: 149
Epoch: 470/500, loss: 3.9212585329202523, correct: 149
Epoch: 480/500, loss: 3.8334668822380857, correct: 149
Epoch: 490/500, loss: 3.752466535254265, correct: 149
Epoch: 500/500, loss: 3.6769046112732537, correct: 149
```
![simple.png](/simple.png)

```python
PTS = 150
DATASET = minitorch.datasets["Diag"](PTS)
HIDDEN = 2
RATE = 0.5
```
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 64.17549157499944, correct: 125
Epoch: 20/500, loss: 59.56480131117531, correct: 125
Epoch: 30/500, loss: 54.39726079132309, correct: 125
Epoch: 40/500, loss: 47.56936355157473, correct: 125
Epoch: 50/500, loss: 38.42481224188839, correct: 129
Epoch: 60/500, loss: 29.39295841475026, correct: 135
Epoch: 70/500, loss: 23.067430295573, correct: 141
Epoch: 80/500, loss: 19.036322628545516, correct: 146
Epoch: 90/500, loss: 16.44399987829961, correct: 149
Epoch: 100/500, loss: 14.54329954878951, correct: 150
Epoch: 110/500, loss: 13.116482501985256, correct: 150
Epoch: 120/500, loss: 11.97989599935435, correct: 150
Epoch: 130/500, loss: 11.09418605893836, correct: 150
Epoch: 140/500, loss: 10.313723408653765, correct: 150
Epoch: 150/500, loss: 9.650359867483733, correct: 150
Epoch: 160/500, loss: 9.082869101687448, correct: 150
Epoch: 170/500, loss: 8.589123665926929, correct: 150
Epoch: 180/500, loss: 8.151603965179872, correct: 150
Epoch: 190/500, loss: 7.759110834200025, correct: 150
Epoch: 200/500, loss: 7.404635357224622, correct: 150
Epoch: 210/500, loss: 7.082571733278812, correct: 150
Epoch: 220/500, loss: 6.788385069303262, correct: 150
Epoch: 230/500, loss: 6.525391707006546, correct: 150
Epoch: 240/500, loss: 6.280551465710706, correct: 150
Epoch: 250/500, loss: 6.049708394611583, correct: 150
Epoch: 260/500, loss: 5.840466725428141, correct: 150
Epoch: 270/500, loss: 5.64552863286596, correct: 150
Epoch: 280/500, loss: 5.463251326942767, correct: 150
Epoch: 290/500, loss: 5.292293083340569, correct: 150
Epoch: 300/500, loss: 5.131524285902103, correct: 150
Epoch: 310/500, loss: 4.979976489970409, correct: 150
Epoch: 320/500, loss: 4.836809888870857, correct: 150
Epoch: 330/500, loss: 4.701290484001355, correct: 150
Epoch: 340/500, loss: 4.572772905991196, correct: 150
Epoch: 350/500, loss: 4.450686887193309, correct: 150
Epoch: 360/500, loss: 4.334526310341363, correct: 150
Epoch: 370/500, loss: 4.2238401920401945, correct: 150
Epoch: 380/500, loss: 4.118225178078129, correct: 150
Epoch: 390/500, loss: 4.024433014139849, correct: 150
Epoch: 400/500, loss: 3.9307697362999496, correct: 150
Epoch: 410/500, loss: 3.8392697587223177, correct: 150
Epoch: 420/500, loss: 3.751497672163953, correct: 150
Epoch: 430/500, loss: 3.667294405539263, correct: 150
Epoch: 440/500, loss: 3.5813604867153184, correct: 150
Epoch: 450/500, loss: 3.504299698641465, correct: 150
Epoch: 460/500, loss: 3.4303500543056913, correct: 150
Epoch: 470/500, loss: 3.359102806695578, correct: 150
Epoch: 480/500, loss: 3.2904006950420492, correct: 150
Epoch: 490/500, loss: 3.2241027438504637, correct: 150
Epoch: 500/500, loss: 3.1600798549288824, correct: 150
```
![diag.png](/diag.png)

```python
PTS = 150
DATASET = minitorch.datasets["Split"](PTS)
HIDDEN = 5
RATE = 0.5
```
```
Epoch: 0/600, loss: 0, correct: 0
Epoch: 10/600, loss: 101.97024649106129, correct: 86
Epoch: 20/600, loss: 101.60516412247918, correct: 86
Epoch: 30/600, loss: 101.00339408376999, correct: 86
Epoch: 40/600, loss: 99.61298623187332, correct: 91
Epoch: 50/600, loss: 97.43593394701507, correct: 96
Epoch: 60/600, loss: 94.10846891815257, correct: 109
Epoch: 70/600, loss: 88.06459730126019, correct: 111
Epoch: 80/600, loss: 78.8183936535218, correct: 120
Epoch: 90/600, loss: 65.93608071912745, correct: 139
Epoch: 100/600, loss: 76.01767679547807, correct: 97
Epoch: 110/600, loss: 65.7661356539468, correct: 109
Epoch: 120/600, loss: 71.71668275727727, correct: 105
Epoch: 130/600, loss: 52.973668272212095, correct: 125
Epoch: 140/600, loss: 48.748511126363596, correct: 126
Epoch: 150/600, loss: 56.186098612295915, correct: 121
Epoch: 160/600, loss: 48.745205540154835, correct: 123
Epoch: 170/600, loss: 40.56987868389373, correct: 130
Epoch: 180/600, loss: 38.32098742528798, correct: 132
Epoch: 190/600, loss: 39.07140376114709, correct: 132
Epoch: 200/600, loss: 35.14613861076087, correct: 132
Epoch: 210/600, loss: 32.35287464947154, correct: 133
Epoch: 220/600, loss: 29.53258710750423, correct: 135
Epoch: 230/600, loss: 29.615679264945964, correct: 135
Epoch: 240/600, loss: 33.13923023693618, correct: 132
Epoch: 250/600, loss: 28.664433653454434, correct: 136
Epoch: 260/600, loss: 23.1574393251824, correct: 139
Epoch: 270/600, loss: 25.37325260652408, correct: 136
Epoch: 280/600, loss: 30.942225391712363, correct: 133
Epoch: 290/600, loss: 26.501474384311923, correct: 136
Epoch: 300/600, loss: 20.288306822851208, correct: 139
Epoch: 310/600, loss: 20.276977242283007, correct: 139
Epoch: 320/600, loss: 25.915861399328676, correct: 136
Epoch: 330/600, loss: 27.925798738794708, correct: 136
Epoch: 340/600, loss: 20.83780980414153, correct: 139
Epoch: 350/600, loss: 17.801230395633606, correct: 141
Epoch: 360/600, loss: 19.844674300209164, correct: 139
Epoch: 370/600, loss: 24.08703074245059, correct: 137
Epoch: 380/600, loss: 23.561082727350033, correct: 138
Epoch: 390/600, loss: 18.917186764568065, correct: 140
Epoch: 400/600, loss: 17.746351777981015, correct: 141
Epoch: 410/600, loss: 19.23549359585491, correct: 139
Epoch: 420/600, loss: 20.712091726091597, correct: 139
Epoch: 430/600, loss: 20.048503626713053, correct: 139
Epoch: 440/600, loss: 18.651341503639973, correct: 141
Epoch: 450/600, loss: 18.199596781973206, correct: 141
Epoch: 460/600, loss: 18.288993153708017, correct: 141
Epoch: 470/600, loss: 18.4565338630854, correct: 141
Epoch: 480/600, loss: 18.24323778543614, correct: 141
Epoch: 490/600, loss: 17.84301329996074, correct: 141
Epoch: 500/600, loss: 17.608842393877172, correct: 141
Epoch: 510/600, loss: 17.480066789997302, correct: 141
Epoch: 520/600, loss: 17.30585434555929, correct: 141
Epoch: 530/600, loss: 17.03567666824771, correct: 141
Epoch: 540/600, loss: 16.78586995298984, correct: 141
Epoch: 550/600, loss: 16.634119486253606, correct: 141
Epoch: 560/600, loss: 16.59139410184603, correct: 141
Epoch: 570/600, loss: 16.465506688789418, correct: 141
Epoch: 580/600, loss: 16.342354084990397, correct: 142
Epoch: 590/600, loss: 16.170190697290998, correct: 142
Epoch: 600/600, loss: 16.044384781003533, correct: 142
```

![split.png](/split.png)

```python
PTS = 150
DATASET = minitorch.datasets["Xor"](PTS)
HIDDEN = 7
RATE = 0.5
```
```
Epoch: 0/600, loss: 0, correct: 0
Epoch: 10/600, loss: 92.13939638626606, correct: 112
Epoch: 20/600, loss: 81.60420037215643, correct: 123
Epoch: 30/600, loss: 70.48221888265279, correct: 125
Epoch: 40/600, loss: 59.813032051936744, correct: 125
Epoch: 50/600, loss: 52.394464632921135, correct: 127
Epoch: 60/600, loss: 49.48104678245807, correct: 129
Epoch: 70/600, loss: 61.372986578189995, correct: 118
Epoch: 80/600, loss: 50.2016664714091, correct: 130
Epoch: 90/600, loss: 48.02684836134698, correct: 131
Epoch: 100/600, loss: 47.97821594630533, correct: 130
Epoch: 110/600, loss: 50.047578437302434, correct: 130
Epoch: 120/600, loss: 37.04216092480403, correct: 134
Epoch: 130/600, loss: 35.870692075951716, correct: 136
Epoch: 140/600, loss: 42.73873100111043, correct: 132
Epoch: 150/600, loss: 37.573064958826336, correct: 136
Epoch: 160/600, loss: 31.14768477840266, correct: 140
Epoch: 170/600, loss: 28.94699919765263, correct: 140
Epoch: 180/600, loss: 26.077148564009285, correct: 142
Epoch: 190/600, loss: 25.86843485667582, correct: 142
Epoch: 200/600, loss: 34.54429315643297, correct: 131
Epoch: 210/600, loss: 21.55936303460329, correct: 143
Epoch: 220/600, loss: 22.216651308469, correct: 143
Epoch: 230/600, loss: 22.511194320966695, correct: 142
Epoch: 240/600, loss: 22.73534203500998, correct: 140
Epoch: 250/600, loss: 21.177918605614273, correct: 143
Epoch: 260/600, loss: 18.300759201759746, correct: 143
Epoch: 270/600, loss: 18.83523517077005, correct: 143
Epoch: 280/600, loss: 19.113005219863627, correct: 142
Epoch: 290/600, loss: 18.124919634978234, correct: 143
Epoch: 300/600, loss: 18.409730759026058, correct: 142
Epoch: 310/600, loss: 17.450148682436133, correct: 142
Epoch: 320/600, loss: 16.988493942140888, correct: 142
Epoch: 330/600, loss: 16.85912027729569, correct: 142
Epoch: 340/600, loss: 16.692357046241753, correct: 142
Epoch: 350/600, loss: 15.7725000309215, correct: 142
Epoch: 360/600, loss: 13.249361101544757, correct: 144
Epoch: 370/600, loss: 14.00047470395496, correct: 144
Epoch: 380/600, loss: 12.522499727331072, correct: 145
Epoch: 390/600, loss: 13.157680250389046, correct: 144
Epoch: 400/600, loss: 12.99571987910919, correct: 144
Epoch: 410/600, loss: 10.148154977336834, correct: 148
Epoch: 420/600, loss: 9.220269443252786, correct: 149
Epoch: 430/600, loss: 11.888376822263494, correct: 144
Epoch: 440/600, loss: 19.396851810788988, correct: 141
Epoch: 450/600, loss: 9.527018594422204, correct: 148
Epoch: 460/600, loss: 8.104488525790128, correct: 149
Epoch: 470/600, loss: 7.772169243035175, correct: 149
Epoch: 480/600, loss: 7.710522263368031, correct: 149
Epoch: 490/600, loss: 14.407456246723317, correct: 142
Epoch: 500/600, loss: 19.979483302925956, correct: 140
Epoch: 510/600, loss: 7.535112715526235, correct: 149
Epoch: 520/600, loss: 6.994946455559455, correct: 149
Epoch: 530/600, loss: 6.742955662239714, correct: 149
Epoch: 540/600, loss: 6.590552832203412, correct: 149
Epoch: 550/600, loss: 6.647246543440026, correct: 149
Epoch: 560/600, loss: 7.36108931400993, correct: 148
Epoch: 570/600, loss: 14.108895264354562, correct: 144
Epoch: 580/600, loss: 17.435999514256682, correct: 143
Epoch: 590/600, loss: 6.182787099651078, correct: 149
Epoch: 600/600, loss: 5.7498139474174295, correct: 149
```
![xor.png](/xor.png)

```python
PTS = 150
DATASET = minitorch.datasets["Circle"](PTS)
HIDDEN = 10
RATE = 0.1
```
```
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/700, loss: 112.84929193103905, correct: 39
Epoch: 20/700, loss: 100.09087114910953, correct: 104
Epoch: 30/700, loss: 98.23690675274278, correct: 104
Epoch: 40/700, loss: 96.33384717360676, correct: 104
Epoch: 50/700, loss: 94.76353594584056, correct: 104
Epoch: 60/700, loss: 93.56506387163839, correct: 104
Epoch: 70/700, loss: 92.9521810799226, correct: 104
Epoch: 0/700, loss: 0, correct: 0
Epoch: 10/700, loss: 90.29546798144715, correct: 104
Epoch: 20/700, loss: 86.5130269153618, correct: 104
Epoch: 30/700, loss: 86.01226442104442, correct: 104
Epoch: 40/700, loss: 85.51641143099057, correct: 104
Epoch: 50/700, loss: 84.95697038889406, correct: 104
Epoch: 60/700, loss: 84.33650441076884, correct: 104
Epoch: 70/700, loss: 83.52663816178132, correct: 104
Epoch: 80/700, loss: 82.64633021862772, correct: 104
Epoch: 90/700, loss: 81.79912803797856, correct: 104
Epoch: 100/700, loss: 81.01006836990993, correct: 104
Epoch: 110/700, loss: 80.10406845410529, correct: 104
Epoch: 120/700, loss: 79.15490489314351, correct: 104
Epoch: 130/700, loss: 78.17323736285124, correct: 104
Epoch: 140/700, loss: 77.33048151276061, correct: 104
Epoch: 150/700, loss: 76.51814462394894, correct: 104
Epoch: 160/700, loss: 75.67356985996715, correct: 104
Epoch: 170/700, loss: 74.80464807856266, correct: 104
Epoch: 180/700, loss: 73.90417149784517, correct: 104
Epoch: 190/700, loss: 72.96287710066923, correct: 104
Epoch: 200/700, loss: 71.974212395568, correct: 105
Epoch: 210/700, loss: 70.94919230336133, correct: 106
Epoch: 220/700, loss: 69.87731845264837, correct: 109
Epoch: 230/700, loss: 68.7512923863133, correct: 113
Epoch: 240/700, loss: 67.5620890337076, correct: 116
Epoch: 250/700, loss: 66.31591178564588, correct: 117
Epoch: 260/700, loss: 65.01635629696872, correct: 118
Epoch: 270/700, loss: 63.65351268392676, correct: 121
Epoch: 280/700, loss: 62.24413297241591, correct: 124
Epoch: 290/700, loss: 60.7826525384239, correct: 125
Epoch: 300/700, loss: 59.27435892867771, correct: 127
Epoch: 310/700, loss: 57.724572783417976, correct: 129
Epoch: 320/700, loss: 56.12378714752338, correct: 130
Epoch: 330/700, loss: 54.4697435985392, correct: 131
Epoch: 340/700, loss: 52.78980344448236, correct: 133
Epoch: 350/700, loss: 51.070528623822426, correct: 134
Epoch: 360/700, loss: 49.301544097833435, correct: 137
Epoch: 370/700, loss: 47.5476444105145, correct: 137
Epoch: 380/700, loss: 45.809729688398804, correct: 138
Epoch: 390/700, loss: 44.08934459003846, correct: 139
Epoch: 400/700, loss: 42.42962676592896, correct: 140
Epoch: 410/700, loss: 40.8351719279358, correct: 140
Epoch: 420/700, loss: 39.279793851083475, correct: 141
Epoch: 430/700, loss: 37.81150979074201, correct: 141
Epoch: 440/700, loss: 36.45162454906974, correct: 141
Epoch: 450/700, loss: 35.19437746380446, correct: 143
Epoch: 460/700, loss: 33.995029418467425, correct: 144
Epoch: 470/700, loss: 32.87419318554786, correct: 144
Epoch: 480/700, loss: 31.822194727168498, correct: 143
Epoch: 490/700, loss: 30.824935735689472, correct: 143
Epoch: 500/700, loss: 29.89844504278993, correct: 143
Epoch: 510/700, loss: 29.027434166759853, correct: 144
Epoch: 520/700, loss: 28.209194237076282, correct: 144
Epoch: 530/700, loss: 27.42561120528718, correct: 144
Epoch: 540/700, loss: 26.682159817243445, correct: 144
Epoch: 550/700, loss: 25.982218782596345, correct: 144
Epoch: 560/700, loss: 25.31028128334189, correct: 144
Epoch: 570/700, loss: 24.68055672809064, correct: 144
Epoch: 580/700, loss: 24.082218645595304, correct: 144
Epoch: 590/700, loss: 23.502624234495748, correct: 144
Epoch: 600/700, loss: 22.927218413580448, correct: 144
Epoch: 610/700, loss: 22.35518150539521, correct: 144
Epoch: 620/700, loss: 21.867176146936757, correct: 144
Epoch: 630/700, loss: 21.410504242071895, correct: 144
Epoch: 640/700, loss: 20.97541062062303, correct: 144
Epoch: 650/700, loss: 20.559396940593803, correct: 144
Epoch: 660/700, loss: 20.15843161037924, correct: 145
Epoch: 670/700, loss: 19.775079446253617, correct: 145
Epoch: 680/700, loss: 19.408394725974986, correct: 145
Epoch: 690/700, loss: 19.04839122328182, correct: 145
Epoch: 700/700, loss: 18.700624483840336, correct: 145
```

![circle.png](/circle.png)
