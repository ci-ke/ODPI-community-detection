# 环境配置
如果想要运行在其他机器上，苦于配置各种依赖环境的话  
1）可以直接将CommunityDetection/.local/lib/python2.7直接上传到对应的~/.local/lib/下，因为python运行的时候，会搜寻这个目录  
2）同理，将CommunityDetection/.local/bin/pip 上传到对应的位置  
3）那么我们就可以直接使用~/.local/bin/pip install/unstaill package啦，避免各种依赖包环境等配置  
  
# 代码运行说明(在algorithm目录下的)
## EADP.py --- 算法核心代码 ODPI.py

run_linux_generate_picture函数包含配置参数

## my_* -- 算法的辅助代码(如一些基础类的定义，以及一些对比算法等)
## data_util(各种可执行文件的集合)
    中的benchmark 生成合成网络的（上传之后需要修改相应的权限）
    中的onmi 计算onmi的代码（上传之后需要修改相应的权限）
    中的gce 对比算法gce的可执行文件
    中的gce_code 对比算法gce的源代码，以及Makefile,对build/MAKEFILE make编译之后得到可执行文件GCECommunityFinder(可修改为gce)
    中的omega 评估标准omega的源码和可执行文件(源码为java，编译成class之后直接调用即可)
    中的test*(因为多个窗口运行的时候，彼此之间的生产的文件不能相互影响), 主要是运行的时候存放 benchmark 生成的网络，以及解析之后的
    lfr_true.txt,lfr_code.txt 以及支持gce算法需要生成的graph_input.txt
## result_images(里面也是通过不同的test区分开多个不同的窗口)
    这个是存放结果图像的文件夹
## datasets
    这个主要就是存放的真实网络的数据集文件
## datasets_images
    这个主要是一些真实数据集的社区结构的图片(如果想看真实网络的结构，可以参考这个目录)
## reference_parper
    这个主要是参考的文件(之后所有的参考文献等都可以统一放在这个目录维护)
### 注意，参考论文中有一篇《Overlapping Community Detection in Networks: The State-of-the-Art and Comparative Study》网络中重叠的社区检测：最新技术和比较研究
    
# 代码运行注意
## my_util.py上path和run_platform说明
首先解释一下为什么需要这两个变量  
由于很多的依赖代码都c、c++或者其他语言编写的(如ONMI的计算等)，但是我一开始在windows上懒得配置各种gcc等的环境,  
然后就是将所有的依赖非python的代码经过编译成可执行文件后全部放在linux或者mac，然后在通过python的command调用  
(command的调用需要一定是绝对路径，可以理解为是单独启了一个窗口，相对路径找不到的，当时被这个坑了好久) 
所以在做代码迁移的时候，只需将data_util一起迁移过去，并且将里面的一些代码编译成可执行文件即可，最后在my_util上  
修改相应的路径即可。

然后对于一些真实的数据集在修改算法之后经常需要测试，所以EADP运行的时候，通过run_platform=windows指定，直接解析  
本地数据文件，然后运行输出相应的结果（但是这里对于windows上没有onmi的时候情况，需要将结果拷贝到具有onmi可以  
执行的机器目录，然后手动调用计算出相应的评估标准，这样的操作也比较麻烦，其实对于目前我使用的mac系统是可以直接调用，  
这里后期对于真实社区的运行得到的评估标准，我也会做一个相应的修改）

## 代码在跑实验的时候，想要并且运行
python2.7 EADP.py test1 test2 表示运行对应的test1和test2的用例，可以直接这样传入参数
如果想要同时运行test1(即在服务器上，想要运行两次的实验)等，那么一定需要复制一个data_util和一个result_images目录
并且在my_util.py中修改path和EADP.py中修改对应的./result_images代码


# 代码说明
EADP.py文件中的代码是经过不停的迭代修改之后的代码(里面也有EADP的代码，但是需要控制好多的分支逻辑，比较麻烦，所以单独拆出来了一个EADP_cankao.py文件)
然后我将EADP的论文实现的代码单独拆出来(EADP_cankao.py),该代码只是实现了EADP论文的主题思路，论文里面有很多
细节没有说明的，甚至有的都是错误的逻辑，我按照我的理解将其进行了修改。

# 社区发现算法总结(里面有很多的重叠社区以及非重叠社区发现算法的代码以及说明，但是很多都是运行编译不成功的)：
https://github.com/RapidsAtHKUST/CommunityDetectionCodes  

# 生成网络代码
https://github.com/eXascaleInfolab/LFR-Benchmark_UndirWeightOvp  

# 评估标准(和EADP的论文的评估标准是一样的，只不过它们叫xx Index，我们的论文叫Omega)
ONMI: 通用的  
OMEGA: https://github.com/amirubin87/OmegaIndex  
（Fuzzy overlapping communities in networks  ）
 （Overlapping Communities Explain Core–Periphery Organization of Networks  ）
（具体的公式:Omega: A General Formulation of the Rand Index of Cluster Recovery Suitable for Non-disjoint Solutions 这篇论文中有对其的介绍）
F-Score: https://github.com/GiulioRossetti/f1-communities
Modularity: https://github.com/RapidsAtHKUST/CommunityDetectionCodes/blob/master/Metrics/metrics/link_belong_modularity.py

# 对比算法
## EADP论文的采用的对比算法：
1）MOSES(可执行文件已有 http://sites.google.com/site/aaronmcdaid/moses，从这下载，编译通不过但是有直接编译好的可执行文件)  
2）SLPA 已经实现  
3）HOCTracker 未找到对应的代码  
4）OCDDP(代码已有，matlab编写，不会弄环境 https://github.com/XueyingBai/An-overlapping-community-detection-algorithm-based-on-density-peaks)  
5）SMEFRW 未找到对应的代码  

## 我们的论文目前使用的对比算法
CPM: 派系过滤算法，clique percolation method  
LFM_EX: 局部拓展，LFM的改进版  
SLPA：EADP论文上也有  
GCE：主要是由C语言实现，https://github.com/RapidsAtHKUST/CommunityDetectionCodes，但是由于效果比我们的好，估计不能使用  
DEMON: python实现 https://github.com/RapidsAtHKUST/CommunityDetectionCodes  
MOSES：EADP论文上也有    

## 对比算法存在的问题
OCDDP(EADP中已经有的)：  
  代码：https://github.com/XueyingBai/An-overlapping-community-detection-algorithm-based-on-density-peaks  
  论文：https://www.sciencedirect.com/science/article/abs/pii/S092523121631400X  
  由于代码是Matlab编写的，不会调试环境  

# 数据集
## EADP论文使用的数据集
数据集来源 https://github.com/XueyingBai/An-overlapping-community-detection-algorithm-based-on-density-peaks/tree/master/Real_World_Data  
数据集来源 https://github.com/RapidsAtHKUST/CommunityDetectionCodes/tree/master/Datasets  
数据集来源 https://blog.csdn.net/wzgang123/article/details/51089521  
### known  
Dolphin(ok)  
Football(ok)  
Karate(ok)  
Polbooks(ok)  
School1    
School2  
    
### unkonwn   
Power(ok)  
Netscience(ok)
Hep-th  
Astro-ph  
Cond-mat(ok 但是节点个数和EADP中的描述不同)  
Cond-mat-2003 
 
在论文 <Centrality in Complex Networks with Overlapping Community Structure> 中有对下面两个数据集的引用说明 
ego-Facebook(ok)  
ca-GrQc(ok)


# 依赖环境
algorithm/requirements.txt里面指定了运行该算法需要依赖的包，最好按照里面的版本进行安装，   
可能会出现一些版本不对而出现报错啥的，遇到就浪费时间，

# windows 下使用的FileZilla进行文件上传要求输入主密码啥的，以验证该服务啥的
是在编辑的设置里的界面密码里进行修改设置的。

# 后台运行程序
解决后台运行程序，xshell关闭，或者网络断开，程序也会自动断开的问题
nohup python2.7 -u EADP.py node_contrast > node_contrast.log 2>&1 &
nohup python -u ODPI.py > ../true_known.log 2>&1 &
nohup python -u ODPI.py muw_n_om > ../N_gen.log 2>&1 &
也可了解screen指令，我嫌麻烦就是用上面的就行
