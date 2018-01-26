# -*- coding: utf-8 -*-

import random
from PIL import Image
import numpy as np

import keras
from keras.models import Input
from keras.models import Model
from keras.models import load_model
from keras.layers.merge import concatenate,add
from keras.layers.core import Dense, Dropout, Activation, Flatten,Reshape,Permute# モジュールのインポート
from keras.layers.convolutional import Conv2D,Convolution2D, MaxPooling2D,Cropping2D,Deconvolution2D,Conv2DTranspose# CNN層、Pooling層のインポート
from keras.optimizers import Adam
from keras.utils import plot_model
from collections import deque

import matplotlib.pyplot as plt

#　ぷよ色
RED     = 1
BLUE    = 2
GREEN   = 3
YELLOW  = 4

#　学習データの保存用クラス
class Memory:
    def __init__(self,max_size = 1000):
        self.buffer = deque(maxlen=max_size)

    def add(self,experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        index = np.random.choice(np.arange(len(self.buffer)),size=batch_size,replace=False)
        return [self.buffer[i] for i in index]

    def len(self):
        return len(self.buffer)

# ぷよクラス
class Puyo:

    NONE = 0
    RED = 1
    BLUE = 2
    GREEN = 3
    YELLOW = 4

    LEFT = 1
    RIGHT = 3
    UP = 2
    DOWN = 4

    def __init__(self,color1,color2,direction = UP):
        self.direct = direction
        self.color1 = color1
        self.color2 = color2

    def set_direct(self,direct):
        self.direct = direct

# 現在ぷよを落下させる
def puyo_fall(stage,puyo,x):
    line = stage[:,x]
    puyo1_fall = None
    puyo2_fall = None

    try:
        if puyo.direct == Puyo.UP:
            # 親ぷよ
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y, x] = puyo.color1
            puyo1_fall = (x, fall_y)
            # 従属ぷよ
            line = stage[:, x]
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y, x] = puyo.color2
            puyo2_fall = (x, fall_y)

        elif puyo.direct == Puyo.LEFT:
            #親ぷよ
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y, x] = puyo.color1
            puyo1_fall = (x, fall_y)
            #従属ぷよ
            line = stage[:, x-1]
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y,x-1] = puyo.color2
            puyo2_fall = (x, fall_y)

        elif puyo.direct == Puyo.RIGHT:
            # 親ぷよ
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y, x] = puyo.color1
            puyo1_fall = (x, fall_y)
            # 従属ぷよ
            line = stage[:, x + 1]
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y, x + 1] = puyo.color2
            puyo2_fall = (x, fall_y)

        elif puyo.direct == Puyo.DOWN:
            # 従属ぷよ
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y, x] = puyo.color2
            puyo2_fall = (x, fall_y)
            # 親ぷよ
            NONEIndex = np.where(line == Puyo.NONE)
            fall_y = np.max(NONEIndex)
            stage[fall_y, x] = puyo.color1
            puyo1_fall = (x, fall_y)
    except(ValueError):
        return stage,False

    return stage,True

# ぷよ消しのアルゴリズム
def erase(stage,newStage):
    diff = stage - newStage
    getScore = 0.0
    points = np.where(diff != 0)
    for i in range(0,len(points[0])):
        try:
            x = points[1][i]
            y = points[0][i]
        except IndexError:
            print('!?')
        color = newStage[y,x]
        cpStage,counter = erase_puyo(newStage.copy(),x,y,color)
        if counter > 3:
            getScore += counter*5
            newStage = cpStage
    return newStage,getScore

def erase_puyo(stage,x,y,color,counter = 0):
    if color == Puyo.NONE:
        return stage,counter
    if stage[y,x] != color:
        return stage,counter
    else :
        counter += 1
        stage[y,x] = Puyo.NONE

    if y - 1 != -1 :
        stage,counter = erase_puyo(stage,x,y-1,color,counter)
    if y + 1 != 13:
        stage,counter = erase_puyo(stage, x,y+1,color,counter)
    if x - 1 != -1 :
        stage,counter = erase_puyo(stage,x-1,y,color,counter)
    if x + 1 !=  6:
        stage,counter = erase_puyo(stage, x+1,y,color,counter)

    return stage,counter

#　ぷよを落下させる
def fall(stage):
    for i in range(6):
        while(True):
            line = stage[:, i]
            NONEIndex = np.where(line == Puyo.NONE)
            PUYOIndex = np.where(line != Puyo.NONE)

            if len(NONEIndex[0]) == 0 or len(PUYOIndex[0]) == 0:
                break

            try:
                noneMax = np.max(NONEIndex)
                puyoMin = np.min(PUYOIndex)
            except ValueError:
                print('!?')

            if (noneMax > puyoMin):
                line[puyoMin+1:noneMax+1] = line[puyoMin:noneMax]
                line[puyoMin] = Puyo.NONE
                #print(puyoMin,noneMax)
                stage[:,i] = line
            else:
                break

    return stage

#データ変形用
def stage2Binary(stage):
    index = stage
    stage_c = np.zeros([13,6,5])

    for i in range(0,13):
        for j in range(0,6):
            num = index[i,j]
            stage_c[i,j,int(index[i,j])] = 1

        tage_c = stage_c.reshape([1,13,6,5])
    return stage_c

def create_Qmodel(learning_rate = 0.1**(4)):

    puyo_input = Input(shape=(13,6,5),name='puyo_net')
    x = Conv2D(filters=1,kernel_size = (13,1),strides=(1,1),activation='relu',padding='valid')(puyo_input)
    x = Flatten()(x)

    y = Conv2D(filters=1,kernel_size = (1,6),strides=(1,1),activation='relu',padding='valid')(puyo_input)
    y = Flatten()(y)
    nowpuyo_input = Input(shape=(2, 5),name='nowpuyo_input')
    nextpuyo_input = Input(shape=(2, 5), name='nextpuyo_input')

    z = Conv2D(filters=16,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(puyo_input)
    z = Conv2D(filters=16,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(z)
    z = MaxPooling2D()(z)
    z = Conv2D(filters=32,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(z)
    z = Conv2D(filters=32,kernel_size = (2,2),strides=(1,1),activation='relu',padding='same')(z)
    z = MaxPooling2D()(z)
    z = Flatten()(z)

    a = Flatten()(nowpuyo_input)
    b = Flatten()(nextpuyo_input)

    x = keras.layers.concatenate([x,y,z,a,b],axis=1)
    x = Dense(1000,activation='relu')(x)
    x = Dense(400, activation='relu')(x)
    output = Dense(22,activation='linear',name='output')(x)
    optimizer = Adam(lr=learning_rate)
    model = Model(inputs=[puyo_input,nowpuyo_input,nextpuyo_input],outputs=output)
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    plot_model(model, to_file='model.png',show_shapes=True)

    return model


#行動を決定する関数
def get_action(state_puyo,model,turn,puyos,learning = True):
    epsilon = 0.001 + 0.9 / (1.0 + turn)
    state = stage2Binary(state_puyo)
    result = model.predict([map2batch(state),puyos[0].reshape((1,2,5)),puyos[1].reshape((1,2,5))])

    epsilon = 1


    if epsilon <= np.random.uniform(0,1):
        next_action = (np.argmax(result[0])) #最も値の高いインデックスを選択
    else:
        next_action = np.random.randint(0,22)
    return next_action,result

#盤面更新
def updata_state(gameMap_origin,action,puyo):
    gameMap = np.copy(gameMap_origin)

    counter = 0
    score = 0
    #向き・落下点決め
    if action < 3:
        x = 0
        if action == 0:
            puyo.direct = puyo.RIGHT
        if action == 1:
            puyo.direct = puyo.DOWN
        if action == 2:
            puyo.direct = puyo.UP
    elif action > 18:
        if action == 19:
            puyo.direct = puyo.LEFT
        if action == 20:
            puyo.direct = puyo.DOWN
        if action == 21:
            puyo.direct = puyo.UP
        x = 5
    else:
        puyo.direct = ((action+1)-4) % 4+1
        x = int((action-3)/4)+1
    new_GameMap,flag = puyo_fall(gameMap.copy(),puyo,x)
    noErase = np.copy(new_GameMap)

    if flag == False:
        return gameMap,-1,True,noErase
    reward = 0

    while (True):
        # 消す処理
        newStage, getScore = erase(gameMap.copy(), new_GameMap.copy())

        if getScore > 0:
             reward += 1
        score += getScore * (counter ** 2)

        if getScore == 0:
            return newStage,reward,False,noErase
        counter += 1

        # 落とす
        gameMap = fall(newStage.copy())
        new_GameMap = gameMap
        gameMap = newStage

        getScore = 0

    print(gameMap)

    return gameMap,reward,False,noErase

# バッチ形式に変形する
def map2batch(gameMap,batch_size = 1):
    return gameMap.reshape((batch_size,13,6,5))

#学習の関数
def learning(QmainModel,memory,batch_size,gamma,QtargetModel):

    inputs = np.zeros((batch_size,13,6,5))
    inputs_puyo0 = np.zeros([batch_size, 2, 5])
    inputs_puyo1 = np.zeros([batch_size, 2, 5])
    inputs_puyo2 = np.zeros([batch_size, 2, 5])
    targets = np.zeros((batch_size,MOVE_KIND))
    mini_batch = memory.sample(batch_size)

    for i,(state_b,puyos_b,action_b,reward_b,next_state_b,next_puyos_b) in enumerate(mini_batch):
        state_b = stage2Binary(next_state_b)
        inputs[i:i+1] = state_b #　盤面
        inputs_puyo0[i:i+1] = puyos_b[0]
        inputs_puyo1[i:i+1] = puyos_b[1]
        inputs_puyo2[i:i+1] = puyos_b[2]

        target = reward_b #　state_b盤面の時action_bを行って得た報酬
        cd = next_state_b == np.zeros(state_b.shape).all(axis=1)

        if not cd: # 次状態の盤面が全て0でないなら
            next_state_b = stage2Binary(next_state_b)
            neMap = map2batch(next_state_b)
            retMainQs = QmainModel.predict([neMap,next_puyos_b[0].reshape(1,2,5),next_puyos_b[1].reshape(1,2,5)])[0]
            next_action = np.argmax(retMainQs)
            target = reward_b + gamma * QtargetModel.predict([map2batch(next_state_b),next_puyos_b[0].reshape(1,2,5),next_puyos_b[1].reshape(1,2,5)])[0][next_action]
            if target < -1:
                target = -1

        targets[i] = QmainModel.predict([map2batch(state_b),puyos_b[0].reshape(1,2,5),puyos_b[1].reshape(1,2,5)])
        targets[i][action_b] = target
    QmainModel.fit([inputs,inputs_puyo0,inputs_puyo1], targets, epochs=1, verbose=0)

    return QmainModel

#盤面の画像化
def puyo_img(puyo_field,name = 'img',size = 30):
    field = np.zeros([13*size,6*size,3])
    for i in range(13):
        for j in range(6):
            if puyo_field[i,j] == 1:
                field[i*size:(i+1)*size,j*size:(j+1)*size,0] = 255
            if puyo_field[i,j] == 2:
                field[i*size:(i+1)*size,j*size:(j+1)*size,2] = 255
            if puyo_field[i,j] == 3:
                field[i * size:(i + 1) * size, j * size : (j + 1) * size, 1] = 255
                field[i * size:(i + 1) * size, j * size : (j + 1) * size, 2] = 255
            if puyo_field[i,j] == 4:
                field[i * size:(i + 1) * size, j * size : (j + 1) * size, 0] = 255
                field[i * size:(i + 1) * size, j * size : (j + 1) * size, 2] = 255

    Image.fromarray(field.astype('uint8')).save(name+'.png')

DQN_MODE = True         #DDQN学習にするかDQN学習にするか
EPISODE_NUMBER = 2000   #学習回数
MAXSTEP = 100           #ぷよの最大手番数
MOVE_KIND = 22          #行動の種類
WIN_TURN = 50           #生き残り手番数の目標値

#パラメータ等
batch_size = 10
gamma = 0.8
rewardList = np.zeros(EPISODE_NUMBER)
turnList = np.zeros(EPISODE_NUMBER)

#モデルの作成
QmainModel = create_Qmodel()
QtargetModel = create_Qmodel()

#モデルのロード
#QmainModel = load_model('learnedModel.h5')

#行動と報酬の記憶メモリ
memory = Memory()

### 学習開始 ###
for episode in range(EPISODE_NUMBER):

    puyo_list = list()
    for i in range(MAXSTEP):
        puyo_list.append(Puyo(random.randint(1,4),random.randint(1,4)))

    # 初期マップ生成
    gameMap = np.zeros([13,6])

    episode_reward = 0
    #action,Qpal = get_action(gameMap,QtargetModel,episode)

    #一つ前の試行のモデルと同じにする(DDQN用の処理)
    QtargetModel = QmainModel

    for t in range(MAXSTEP):
        puyos = np.zeros([3,2,5])
        puyos2 = np.zeros([3,2,5])
        for i in range(0,3):
            puyos[i,0,puyo_list[i].color1] = 1
            puyos[i, 1, puyo_list[i].color2] = 1

            puyos2[i,0,puyo_list[i+1].color1 ] = 1
            puyos2[i,1, puyo_list[i + 1].color2] = 1

        #　時刻tでの行動の決定
        action,Qpal = get_action(gameMap,QmainModel,episode,puyos)
        nextGameMap,reward,done,noErase = updata_state(gameMap,action,puyo_list[0])

        # ゲームオーバーだったら
        if done:
            nextGameMap = np.zeros(gameMap.shape)
            puyos2 = np.zeros([3,2,5])

        if t == WIN_TURN:
            nextGameMap = np.zeros(gameMap.shape)
            reward = 1
        puyo_list.pop(0)

        # 合計報酬の更新
        episode_reward += reward

        #状態を保存(学習に使う)
        memory.add((gameMap,puyos,action,reward,nextGameMap,puyos2))

        #モデルの学習
        if (memory.len() > batch_size):
            QmainModel = learning(QmainModel,memory,batch_size,gamma,QtargetModel)

        if DQN_MODE: # DQNの場合は毎回のステップごとにモデルを等しくする
            QtargetModel = QmainModel

        #状態の更新
        if done != True and t != WIN_TURN:
            gameMap = nextGameMap

        if done or t == MAXSTEP -1:
            print ('eposode:'+str(episode)+'  rewards:'+str(episode_reward) + ' turns:'+str(t) )
            rewardList[episode] = reward
            turnList[episode] = t
            #print(gameMap)
            break
        if t == WIN_TURN:
            print('eposode:' + str(episode) + '  rewards:' + str(episode_reward) + ' turns:' + str(t))
           # print(gameMap)
            rewardList[episode] = reward
            turnList[episode] = t
            break

#モデルの保存
QmainModel.save('learnedModel.h5')

plt.plot(rewardList)
plt.show()
plt.clf()

plt.plot(turnList)
plt.show()


### テスト開始 ###
print ('GameStart')

#初期マップ生成
gameMap = np.zeros([13,6])
gameMap = np.zeros([13,6])

score = 0

#ぷよの生成
puyo_list = list()
for i in range(MAXSTEP):
    puyo_list.append(Puyo(random.randint(1, 4), random.randint(1, 4)))

for j in range(0,MAXSTEP):

    puyos = np.zeros([3, 2, 5])
    puyos2 = np.zeros([3, 2, 5])
    for i in range(0, 3):
        puyos[i, 0, puyo_list[i].color1] = 1
        puyos[i, 1, puyo_list[i].color2] = 1

        puyos2[i, 0, puyo_list[i + 1].color1] = 1
        puyos2[i, 1, puyo_list[i + 1].color2] = 1

    action, Qpal = get_action(gameMap, QmainModel, 10**10,puyos,learning = False)
    gameMap,reward,done,noErase = updata_state(gameMap,action,puyo_list[0])
    puyo_list.pop(0)

    score += reward

    print (gameMap)
    puyo_img(gameMap,str(j))
    print('======')

    if j == WIN_TURN:
        break

    if done:
        score = -1
        break

print (score)

