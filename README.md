[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Soliton-Analytics-Team/Julius/blob/main/Julius.ipynb)

# 5分でできる音声認識　JuliusをColabのGPUで動かす

音声認識技術の向上には目を見張るものがあります。ただ、それを応用していろいろやろうとした時、クラウドベースのサービスだと認識結果の文字列のみの出力で物足りない場合があります。そのような時、特に日本語の場合、オープンソースの[Julius](https://julius.osdn.jp/index.php)が頼りになります。そこでJuliusのディクテーションをColabで動かしてみました。後でGPUを使うので、ここでcolabのランタイムのタイプをGPUにしてください。

まずは、[ディクテーションキット](https://julius.osdn.jp/index.php?q=dictation-kit.html)を[ここ](https://github.com/julius-speech/dictation-kit)から取得します。この時、ちょっとつまずきがありました。「git-lfs をインストール・設定してから clone して下さい」と書いてあるのですが、その方法が書いてありません。検索した所、以下の手順でした。


```shell
!apt-get install git-lfs
!git lfs install
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following NEW packages will be installed:
      git-lfs
    0 upgraded, 1 newly installed, 0 to remove and 37 not upgraded.
    Need to get 2,129 kB of archives.
    After this operation, 7,662 kB of additional disk space will be used.
    Get:1 http://archive.ubuntu.com/ubuntu bionic/universe amd64 git-lfs amd64 2.3.4-1 [2,129 kB]
    Fetched 2,129 kB in 4s (473 kB/s)
    Selecting previously unselected package git-lfs.
    (Reading database ... 155229 files and directories currently installed.)
    Preparing to unpack .../git-lfs_2.3.4-1_amd64.deb ...
    Unpacking git-lfs (2.3.4-1) ...
    Setting up git-lfs (2.3.4-1) ...
    Processing triggers for man-db (2.8.3-2ubuntu0.1) ...
    Error: Failed to call git rev-parse --git-dir --show-toplevel: "fatal: not a git repository (or any of the parent directories): .git\n"
    Git LFS initialized.


途中でエラーが出ていますが、大丈夫です。

それではディクテーションキットを取得します。取得サイズが2.1Gと表示されれば成功です。


```shell
%cd /content/
!git clone https://github.com/julius-speech/dictation-kit.git
!du -sh /content/dictation-kit
```

    /content
    Cloning into 'dictation-kit'...
    remote: Enumerating objects: 343, done.[K
    remote: Total 343 (delta 0), reused 0 (delta 0), pack-reused 343[K
    Receiving objects: 100% (343/343), 326.24 MiB | 38.19 MiB/s, done.
    Resolving deltas: 100% (94/94), done.
    Filtering content: 100% (28/28), 851.93 MiB | 62.09 MiB/s, done.
    2.1G	/content/dictation-kit


次に[こちら](https://soratobu.net/voicesample/)から音声サンプルを取得します。音声ファイルはサンプリング周波数16k、モノラル、１６ビットpcm形式に変換します。([再生](http://soratobu.net/wp3/wp-content/uploads/2020/01/fukuikazue_documentary.mp3))

---

```shell
%cd /content
!wget -q http://soratobu.net/wp3/wp-content/uploads/2020/01/fukuikazue_documentary.mp3
!ffmpeg -v 0 -y -i fukuikazue_documentary.mp3 -ar 16000 -ac 1 -acodec pcm_s16le sample.wav
```

    /content


ディクテーションキットに含まれるシェルスクリプトを利用して音声認識を実行します。まずは深層学習版。


```shell
%%time
%cd /content/dictation-kit/
!./run-linux-dnn.sh -input stdin -nolog -spsegment < /content/sample.wav
```

    /content/dictation-kit
    STAT: include config: main.jconf
    STAT: include config: am-dnn.jconf
    STAT: parsing option string: "-htkconf model/dnn/config.lmfb -cvn -cmnload model/dnn/norm -cmnstatic"
    Stat: para: parsing HTK Config file: model/dnn/config.lmfb
    Warning: para: "SOURCEFORMAT" ignored (not supported, or irrelevant)
    Warning: para: "SOURCEKIND" ignored (not supported, or irrelevant)
    Stat: para: SOURCERATE=625
    Warning: para: TARGETKIND skipped (will be determined by AM header)
    Stat: para: TARGETRATE=100000.0
    Warning: para: "SAVECOMPRESSED" ignored (not supported, or irrelevant)
    Warning: para: "SAVEWITHCRC" ignored (not supported, or irrelevant)
    Stat: para: WINDOWSIZE=250000.0
    Stat: para: USEHAMMING=T
    Stat: para: PREEMCOEF=0.97
    Stat: para: NUMCHANS=40
    Stat: para: ENORMALISE=F
    Warning: para: "BYTEORDER" ignored (not supported, or irrelevant)
    
    世界で次々に起こる災害や以上気性が、長期的な地球      
    温暖化の傾向と一致していると、世界気象機関が、
    異例の警笛を鳴らしている。統計開始以来、最近四         
    年間が、平均祇園の最高記録、一から読みを独占して
    いる。
    
    reached end of input on stdin
    CPU times: user 372 ms, sys: 47.5 ms, total: 419 ms
    Wall time: 34 s


次は統計モデル版。


```shell
%%time
%cd /content/dictation-kit/
!./run-linux-gmm.sh -input stdin -nolog -spsegment < /content/sample.wav
```

    /content/dictation-kit
    STAT: include config: main.jconf
    STAT: include config: am-gmm.jconf
    
    世界で次々に起こる災害や異常気象が、長期的な地球      
    温暖化の傾向と一致していると、世界気象機関が、         
    入江の警笛を鳴らしている。統計開始以来。最近四         
    年間が、平均気温の最高記録、一から四運用、特選し      
    ている。
    
    reached end of input on stdin
    CPU times: user 105 ms, sys: 22.6 ms, total: 127 ms
    Wall time: 7.56 s


どちらも惜しい感じです。

run-linux-dnn.shの中身を覗いて他の候補も出力するようにしてみましょう。


```shell
!cat run-linux-dnn.sh
```

    #! /bin/sh
    
    ./bin/linux/julius -C main.jconf -C am-dnn.jconf -demo -dnnconf julius.dnnconf $*



```shell
%%time
%cd /content/dictation-kit/
!./bin/linux/julius -C main.jconf -C am-dnn.jconf -dnnconf julius.dnnconf -input stdin -nolog -spsegment -quiet -confnet < /content/sample.wav
```

    /content/dictation-kit
    STAT: include config: main.jconf
    STAT: include config: am-dnn.jconf
    STAT: parsing option string: "-htkconf model/dnn/config.lmfb -cvn -cmnload model/dnn/norm -cmnstatic"
    Stat: para: parsing HTK Config file: model/dnn/config.lmfb
    Warning: para: "SOURCEFORMAT" ignored (not supported, or irrelevant)
    Warning: para: "SOURCEKIND" ignored (not supported, or irrelevant)
    Stat: para: SOURCERATE=625
    Warning: para: TARGETKIND skipped (will be determined by AM header)
    Stat: para: TARGETRATE=100000.0
    Warning: para: "SAVECOMPRESSED" ignored (not supported, or irrelevant)
    Warning: para: "SAVEWITHCRC" ignored (not supported, or irrelevant)
    Stat: para: WINDOWSIZE=250000.0
    Stat: para: USEHAMMING=T
    Stat: para: PREEMCOEF=0.97
    Stat: para: NUMCHANS=40
    Stat: para: ENORMALISE=F
    Warning: para: "BYTEORDER" ignored (not supported, or irrelevant)
    
    pass1_best:  世界 で 次々 に 起こる 災害 や 異常 気象 が 、 長期 的 な 地球 温暖 化 の 傾向 と 一致 し て いる と 、 世界 気象 機関 が 、 異例 の 警笛 を 鳴らし て いる 。
    sentence1:  世界 で 次々 に 起こる 災害 や 以上 気性 が 、 長期 的 な 地球 温暖 化 の 傾向 と 一致 し て いる と 、 世界 気象 機関 が 、 異例 の 警笛 を 鳴らし て いる 。
    ---- begin confusion network ---
    (:1.000)  
    (-:0.989)  (、:0.006)(ウ:0.004)
    (世界:0.951)  (せかい:0.022)(政界:0.014)(正解:0.010)(青海:0.001)(石灰:0.001)(切開:0.001)(せっかい:0.000)(節介:0.000)(制海:0.000)
    (で:0.982)  (って:0.009)(て:0.008)(出:0.000)(デ:0.000)(Ｄ:0.000)(に:0.000)
    (-:1.000)  (、:0.000)
    (つぎつぎ:0.514)  (次々:0.484)(次つぎ:0.002)
    (に:1.000)  
    (起こる:1.000)  
    (災害:1.000)  
    (や:1.000)  
    (以上:0.537)  (異常:0.463)(-:0.000)
    (気性:1.000)  
    (が:1.000)  
    (、:1.000)  
    (長期:1.000)  
    (的:0.988)  (敵:0.012)
    (な:0.988)  (の:0.012)
    (地球:1.000)  
    (温暖:1.000)  
    (化:1.000)  
    (の:1.000)  
    (傾向:1.000)  
    (と:1.000)  
    (一致:1.000)  
    (し:1.000)  
    (て:1.000)  
    (いる:1.000)  
    (と:1.000)  
    (、:1.000)  
    (世界:1.000)  
    (気象:1.000)  
    (機関:1.000)  
    (が:1.000)  
    (、:1.000)  
    (異例:1.000)  
    (の:1.000)  
    (警笛:1.000)  
    (を:1.000)  
    (鳴らし:1.000)  
    (て:1.000)  
    (いる:1.000)  
    (。:1.000)  
    ---- end confusion network ---
    pass1_best:  統計 開始 以来 、 最近 よ 年間 が 、 平均 気温 の 最高 記録 、 一 から 四 を 独占 し て いる 。
    sentence1:  統計 開始 以来 、 最近 四 年間 が 、 平均 祇園 の 最高 記録 、 一 から 読み を 独占 し て いる 。
    ---- begin confusion network ---
    (:1.000)  (-:0.000)
    (-:1.000)  (ああ:0.000)(を:0.000)(お:0.000)
    (-:0.978)  (、:0.011)(イ:0.008)(二:0.002)(ウ:0.000)
    (-:1.000)  (お:0.000)
    (-:0.514)  (統計:0.484)(東経:0.001)(闘鶏:0.001)(道警:0.000)(同型:0.000)(陶芸:0.000)(東欧:0.000)
    (-:1.000)  (系:0.000)
    (-:0.514)  (開始:0.486)
    (-:0.514)  (以来:0.458)(いらい:0.027)(依頼:0.001)
    (-:0.507)  (、:0.493)
    (-:0.514)  (最近:0.486)
    (-:0.514)  (四:0.481)(余:0.005)
    (-:0.514)  (年間:0.486)
    (-:0.514)  (が:0.486)
    (-:0.514)  (、:0.486)
    (-:0.514)  (平均:0.486)
    (-:0.514)  (祇園:0.486)
    (-:0.514)  (の:0.486)
    (-:0.514)  (最高:0.486)
    (-:0.514)  (記録:0.486)
    (-:0.514)  (、:0.486)
    (-:0.514)  (一:0.486)
    (-:0.514)  (から:0.486)
    (-:0.514)  (読み:0.486)
    (-:0.514)  (を:0.486)
    (-:0.514)  (独占:0.486)
    (-:0.514)  (し:0.486)
    (-:0.514)  (て:0.486)
    (-:0.514)  (いる:0.486)
    (-:0.514)  (。:0.486)
    ---- end confusion network ---
    
    reached end of input on stdin
    CPU times: user 226 ms, sys: 41.2 ms, total: 268 ms
    Wall time: 33.5 s


pass1_bestで「異常気象」や「平均気温」が正しく認識されているのに、最終的な認識結果の中では候補にもあがってません。いろいろパラメータを変えてやってみましたが、今のところ、最終候補に残るようにはできていません。

統計モデルはどうでしょうか。


```shell
%%time
%cd /content/dictation-kit/
!./bin/linux/julius -C main.jconf -C am-gmm.jconf -input stdin -nolog -spsegment -quiet -confnet < /content/sample.wav
```

    /content/dictation-kit
    STAT: include config: main.jconf
    STAT: include config: am-gmm.jconf
    
    pass1_best:  世界 で 次々 に 起こる 災害 や 異常 気象 が 、
    sentence1:  世界 で 次々 に 起こる 災害 や 異常 気象 が 、
    ---- begin confusion network ---
    (:1.000)  (-:0.000)
    (-:0.970)  (、:0.020)(千:0.007)(セ:0.004)
    (世界:0.641)  (政界:0.125)(正解:0.095)(せかい:0.033)(青海:0.026)(旋回:0.016)(前回:0.011)(全開:0.009)(石灰:0.008)(回:0.006)(浅海:0.005)(切開:0.005)(界:0.004)(水害:0.003)(制海:0.003)(世代:0.002)(世帯:0.002)(せっかい:0.002)(節介:0.002)(世凱:0.001)(街:0.001)
    (で:0.854)  (に:0.086)(て:0.040)(って:0.014)(-:0.004)(へ:0.002)
    (-:0.980)  (、:0.020)
    (次々:0.955)  (つぎつぎ:0.027)(次つぎ:0.018)
    (に:1.000)  
    (起こる:1.000)  
    (災害:1.000)  
    (や:1.000)  
    (異常:1.000)  
    (気象:1.000)  
    (が:1.000)  
    (、:1.000)  
    ---- end confusion network ---
    pass1_best: 、 長期 的 な 時期 温暖 化 の 結婚 と 一致 し て いる と 、
    sentence1: 、 長期 的 な 地球 温暖 化 の 傾向 と 一致 し て いる と 、
    ---- begin confusion network ---
    (、:1.000)  
    (-:1.000)  (ソー:0.000)(双:0.000)(ア:0.000)
    (長期:0.981)  (早期:0.009)(想起:0.003)(そう:0.002)(超:0.002)(相互:0.001)(そっち:0.001)(消費:0.001)(臓器:0.000)(その:0.000)(町:0.000)(キ:0.000)(装置:0.000)(朝:0.000)(喜:0.000)(帳:0.000)
    (-:0.998)  (期:0.001)(、:0.000)(き:0.000)(気:0.000)(記:0.000)(来:0.000)(木:0.000)
    (的:0.999)  (知的:0.001)(私的:0.000)(詩的:0.000)
    (な:1.000)  
    (地球:1.000)  
    (温暖:1.000)  
    (化:1.000)  
    (の:1.000)  
    (傾向:1.000)  
    (と:1.000)  
    (一致:1.000)  
    (し:1.000)  
    (て:1.000)  
    (いる:1.000)  
    (と:1.000)  
    (、:1.065)  
    ---- end confusion network ---
    pass1_best: 、 世界 諸 機関 が 、
    sentence1: 、 世界 気象 機関 が 、
    ---- begin confusion network ---
    (、:1.000)  
    (-:0.988)  (千:0.012)
    (世界:0.944)  (せかい:0.014)(旋回:0.012)(回:0.012)(政界:0.010)(正解:0.009)(-:0.000)
    (-:0.998)  (、:0.002)(を:0.001)(が:0.000)
    (-:1.000)  (、:0.000)
    (地:0.385)  (長:0.373)(気象:0.236)(-:0.003)(の:0.001)(町:0.001)(小:0.000)(市:0.000)(超:0.000)(ショー:0.000)(省:0.000)(少:0.000)(一:0.000)(商:0.000)(相:0.000)(症:0.000)(しょう:0.000)(勝:0.000)(庄:0.000)(証:0.000)(焼:0.000)(翔:0.000)(抄:0.000)(昭:0.000)(承:0.000)
    (-:0.611)  (の:0.386)(諸:0.003)(章:0.000)(賞:0.000)(塩:0.000)(将:0.000)(ショウ:0.000)(消:0.000)
    (機関:0.627)  (期間:0.373)(器官:0.000)(-:0.000)
    (が:1.000)  (-:0.000)
    (、:1.000)  (-:0.000)
    ---- end confusion network ---
    pass1_best: 、 入江 の 警笛 を 鳴らし て いる 。
    sentence1: 、 入江 の 警笛 を 鳴らし て いる 。
    ---- begin confusion network ---
    (、:1.000)  
    (入江:0.201)  (入り江:0.156)(異例:0.127)(一定:0.065)(家:0.062)(伊兵衛:0.055)(威厳:0.038)(いずれ:0.034)(家々:0.030)(一連:0.027)(入れ:0.022)(池:0.021)(畏敬:0.019)(慰霊:0.018)(一:0.017)(雪絵:0.014)(意見:0.014)(理系:0.011)(リレー:0.011)(友里恵:0.010)(遺伝:0.009)(湯:0.009)(幸枝:0.008)(以前:0.007)(ユリエ:0.007)(義兄:0.006)(維持:0.004)
    (-:0.979)  (Ａ:0.017)(へ:0.004)
    (の:0.993)  (と:0.007)
    (警笛:1.000)  
    (を:1.000)  
    (鳴らし:1.000)  
    (て:1.000)  
    (いる:1.000)  
    (。:1.000)  
    ---- end confusion network ---
    pass1_best:  と 警戒 し ない 。
    sentence1:  統計 開始 以来 。
    ---- begin confusion network ---
    (:1.000)  (-:0.000)
    (-:0.337)  (当:0.123)(唐:0.068)(党:0.068)(父:0.063)(とう:0.046)(トウ:0.043)(等:0.040)(陶:0.023)(島:0.023)(とー:0.016)(遠:0.016)(塔:0.016)(董:0.014)(鄧:0.014)(十:0.014)(籐:0.013)(問う:0.013)(統:0.013)(當:0.013)(倒:0.009)(糖:0.007)(と:0.007)(遠く:0.002)
    (-:0.925)  (を:0.067)(、:0.008)(お:0.000)
    (軽快:0.444)  (統計:0.332)(警戒:0.220)(当家:0.004)(-:0.000)
    (し:0.645)  (開始:0.337)(白井:0.018)(-:0.000)
    (ない:0.395)  (以来:0.266)(たい:0.250)(いらい:0.059)(-:0.018)(依頼:0.010)(ライン:0.002)
    (-:0.784)  (、:0.126)(の:0.042)(ね:0.030)(で:0.011)(な:0.006)
    (-:0.993)  (え:0.007)
    (。:1.000)  
    ---- end confusion network ---
    pass1_best:  最近 四 年 刊 が 、
    sentence1:  最近 四 年間 が 、
    ---- begin confusion network ---
    (:1.000)  
    (-:0.981)  (あ:0.004)(ああ:0.002)(う:0.002)(あー:0.002)(え:0.002)(お:0.001)(ええ:0.001)(を:0.001)(ウ:0.001)(えー:0.001)(ア:0.000)(イ:0.000)(Ａ:0.000)(絵:0.000)(って:0.000)
    (-:0.941)  (、:0.045)(最:0.006)(サイ:0.005)(イ:0.002)(ア:0.001)
    (最近:0.958)  (さいきん:0.015)(細菌:0.013)(近:0.006)(キン:0.005)(再:0.002)(債:0.001)(-:0.000)
    (-:0.996)  (銀:0.002)(に:0.001)(金:0.001)
    (四:0.993)  (幼年:0.003)(余念:0.003)(余:0.000)(十:0.000)(よ:0.000)(用:0.000)(よう:0.000)
    (年間:0.928)  (年:0.065)(-:0.006)(年鑑:0.000)
    (-:0.976)  (刊:0.014)(感:0.006)(官:0.005)
    (は:0.829)  (が:0.097)(官衙:0.048)(か:0.022)(画:0.004)(-:0.000)
    (、:1.028)  
    ---- end confusion network ---
    pass1_best: 、 平均 気温 の 最高 記録 、
    sentence1: 、 平均 気温 の 最高 記録 、
    ---- begin confusion network ---
    (、:1.000)  
    (-:1.000)  (Ｋ:0.000)(ケ:0.000)(毛:0.000)(軽:0.000)(け:0.000)(計:0.000)(景:0.000)(刑:0.000)
    (平均:0.992)  (北京:0.003)(現金:0.002)(欠勤:0.001)(返金:0.001)(経験:0.001)(献金:0.000)(厳禁:0.000)(金:0.000)
    (-:0.999)  (図:0.000)(、:0.000)(を:0.000)(機:0.000)(ず:0.000)(児:0.000)(時:0.000)
    (気温:0.978)  (祇園:0.010)(企業:0.007)(事業:0.002)(昨日:0.002)(四:0.000)(用:0.000)
    (-:1.000)  (を:0.000)
    (は:0.390)  (-:0.227)(の:0.225)(℃:0.152)(も:0.004)(と:0.001)(が:0.001)(五:0.000)(だ:0.000)
    (-:1.000)  (、:0.000)
    (最高:1.000)  (再興:0.000)(再考:0.000)(採光:0.000)
    (-:0.962)  (を:0.038)
    (記録:0.947)  (気温:0.032)(キロ:0.012)(帰路:0.007)(岐路:0.003)
    (-:0.984)  (を:0.010)(付:0.005)(へ:0.001)(も:0.001)
    (、:2.160)  
    ---- end confusion network ---
    pass1_best: 、 一 第 四 いよう と 苦戦 し て いる 。
    sentence1: 、 一 から 四 運用 、 特選 し て いる 。
    ---- begin confusion network ---
    (、:1.000)  
    (一:0.772)  (市:0.114)(Ⅰ:0.105)(道:0.006)(位置:0.003)(日:0.001)(ⅰ:0.000)(一致:0.000)(-:0.000)(意地:0.000)
    (-:0.999)  (に:0.001)(位:0.000)(二:0.000)
    (-:0.983)  (で:0.014)(、:0.002)(他:0.000)(以下:0.000)
    (-:0.962)  (の:0.022)(は:0.014)(が:0.003)
    (-:0.499)  (内:0.170)(対:0.108)(ない:0.084)(七:0.059)(から:0.034)(名:0.014)(回:0.007)(ナイ:0.006)(なら:0.006)(か:0.002)(たり:0.002)(第:0.002)(台:0.001)(長い:0.001)(な:0.001)(なる:0.001)(だ:0.001)(代:0.000)(三:0.000)(外:0.000)(大:0.000)(荒い:0.000)(Ｉ:0.000)
    (内容:0.415)  (四:0.363)(よう:0.106)(太陽:0.038)(よ:0.036)(大量:0.030)(の:0.008)(概要:0.003)(代表:0.002)(利用:0.001)
    (-:0.956)  (を:0.044)
    (運用:1.000)  (-:0.000)
    (-:0.961)  (を:0.039)
    (、:0.968)  (-:0.031)(夫:0.001)(挙:0.000)
    (-:0.887)  (と:0.113)
    (特選:0.336)  (曲線:0.328)(直線:0.220)(苦戦:0.116)(-:0.000)
    (し:1.000)  (-:0.000)
    (て:1.000)  (-:0.000)
    (いる:1.000)  (-:0.000)
    (。:1.000)  (-:0.000)
    ---- end confusion network ---
    
    1 files processed
    CPU times: user 80.9 ms, sys: 11.7 ms, total: 92.5 ms
    Wall time: 8.85 s


「入江」に対して「異例」も候補に入っています。「特選」に対しては残念ながら「独占」が候補に挙がっていません。

ところで、Julius 4.6ではGPU対応が謳われています。執筆時点でディクテーションキットに含まれているJuliusは4.5です。せっかくなので、colabのGPUを有効にして処理速度向上を確かめてみましょう。

Juliusのエンジンを取得してビルドします。


```shell
!apt-get install build-essential zlib1g-dev libsdl2-dev
!git clone https://github.com/julius-speech/julius.git
%cd /content/julius
!CC=/usr/local/cuda-10.1/bin/nvcc CFLAGS=-O3 ./configure
!make
!make install
```

ビルドしたバイナリを使って音声認識させます。


```shell
%%time
%cd /content/dictation-kit/
!julius -C main.jconf -C am-dnn.jconf -dnnconf julius.dnnconf -input stdin -nolog -spsegment -demo < /content/sample.wav
```

    /content/dictation-kit
    STAT: include config: main.jconf
    STAT: include config: am-dnn.jconf
    STAT: parsing option string: "-htkconf model/dnn/config.lmfb -cvn -cmnload model/dnn/norm -cmnstatic"
    Stat: para: parsing HTK Config file: model/dnn/config.lmfb
    Warning: para: "SOURCEFORMAT" ignored (not supported, or irrelevant)
    Warning: para: "SOURCEKIND" ignored (not supported, or irrelevant)
    Stat: para: SOURCERATE=625
    Warning: para: TARGETKIND skipped (will be determined by AM header)
    Stat: para: TARGETRATE=100000.0
    Warning: para: "SAVECOMPRESSED" ignored (not supported, or irrelevant)
    Warning: para: "SAVEWITHCRC" ignored (not supported, or irrelevant)
    Stat: para: WINDOWSIZE=250000.0
    Stat: para: USEHAMMING=T
    Stat: para: PREEMCOEF=0.97
    Stat: para: NUMCHANS=40
    Stat: para: ENORMALISE=F
    Warning: para: "BYTEORDER" ignored (not supported, or irrelevant)
    
    世界で次々に起こる災害や以上気性が、長期的な地球      
    温暖化の傾向と一致していると、世界気象機関が、
    異例の警笛を鳴らしている。統計開始以来、最近四         
    年間が、平均祇園の最高記録、一から読みを独占して
    いる。
    
    1 files processed
    CPU times: user 150 ms, sys: 24 ms, total: 174 ms
    Wall time: 11.9 s


さすがGPU、精度は変わりませんが処理速度は3倍になっています。

ちなみにGoogleの音声認識はどうでしょうか。


```shell
!pip install SpeechRecognition
```

    Collecting SpeechRecognition
      Downloading SpeechRecognition-3.8.1-py2.py3-none-any.whl (32.8 MB)
    [K     |████████████████████████████████| 32.8 MB 1.3 MB/s 
    [?25hInstalling collected packages: SpeechRecognition
    Successfully installed SpeechRecognition-3.8.1



```shell
%%time

import speech_recognition as sr

AUDIO_FILE = "/content/sample.wav"

# use the audio file as the audio source
r = sr.Recognizer()
with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)  # read the entire audio file

result=r.recognize_google(audio, language='ja-JP')
print(result)
```

    世界で次々に起こる災害や異常気象が長期的な地球温暖化の傾向と一致していると 世界気象機関が異例の警笛を鳴らしている統計開始以来最近 四年間が平均気温の最高記録 1位から4位を独占している
    CPU times: user 13 ms, sys: 8.98 ms, total: 22 ms
    Wall time: 1.47 s


認識結果が完璧な上に処理速度も速い、、、さすが、、、
