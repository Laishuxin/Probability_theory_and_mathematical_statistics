
# 总体、样本

样本与总体的概念，在前面已经多次接触到，只是没有非常明确的给出定义。例如我们一般假设某个学校全体学生的身高，用随机变量X表示，近似服从正态分布，这里的正态分布就是"总体"的分布，字面意思就是所有待研究对象的集合。在实际的数据分析中，我们通过观察或其他测量方式得到的数据一般都只是待研究对象的一个子集，这个子集就是一个样本（可以包含多个个体）。例如通过某种方式，从全体学生中找出100名学生，这100名学生就是一个样本。


## 概念
1. 总体： 研究对象的全部个体
2. 个体： 总体中的一个
3. 总体容量： 总体中的个体的多少
4. 有限总体：容量有限的总体
5. 无限总体：容量无限(很大)的总体

## 样本、总体、随机变量
1. 样本：总体中的一些个体组成样本
2. 随机变量：研究总体某个指标X(如，身高)，对于不同的个体有不同的取值，这些取值构成一个分布。因此可以称$X$是一个随机变量
3. 有时候直接将$X$称为总体. 假设X的分布函数为$F(x)$,也称总体$X$具有分布$F(x)$.

## 简单随机样本
  (简单地说就是，独立同分布的样本)
  
![](images/6_1.png)

### 简单随机抽样
1. 简单随机抽样：获取简单随机样本
2. 如何进行抽样  
    + 放回抽样：样本容量有限(小)
    + 不放回抽样：样本容量无限(很大)

# 统计量
## 定义
>统计量：样本的不含有任何未知参数的**函数**(如，平均值函数等)

## 常用统计量
1. 样本均值：$$\bar{X} = \frac{1}{n} \sum_{i=1}^{n}X_i$$
2. 样本方差(注意分母是n-1): $$S^2 = \frac{1}{n-1} \sum_{i=1}^{n}(X_i - \bar{X})^2$$  
样本标准差：$$S = \sqrt{S^2}$$  
3. 样本矩：  
k阶矩：$$A_k = \frac{1}{n} \sum_{i=1}^{n}X_i^k$$  
k阶中心距：$$B_k = \frac{1}{n} \sum_{i=1}^{n}(X_i - \bar{X}^k)$$  
$$k = 1,2,\cdots$$

### 相关问题
#### $\bar{X} = \mu$？
![](images/6_2.png)
#### 样本方差分母为什么的n-1
##### 无偏估计
1. 简单定义：估计量的期望等于估计参数的真值
2. 理解
设想一下，想知道全体女性的身高均值$\mu$ ,但是我们没有办法对每个女性进行测量，只能抽一部分人来估计全体女性的身高，那么根据抽样的数据如何进行推断？什么样的推断方法才称得上‘准确’？
<br>
比如：我们得到的样本女性身高为：
<center>${x_1,x_2···,x_n}$</center>
那么，
<br>
<center>$\bar{x} = \frac{x_1+x_2+···+x_n}{n}$</center>
<br>
对于$\mu$来说是一个不错的估计，因为它是无偏估计。(即，$\bar{X}$可以近似的代替$\mu$)
<br>
3. 方差$\sigma^2$的无偏估计    
假设：$\mu$已知，而$\sigma^2$未知
<br>
由方差的定义有(对于单个样本)： <br> 
$E[(X_i - \mu)^2] = \sigma^2, i= 1,2,···,n$
<br>
$=>$
<br>
(对于所有样本)
<br>
$E[\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \mu)^2] = \frac{1}{n}\displaystyle \sum_{i=1}^{n}E(X_i-\mu)^2 = \frac{1}{n} \times n\sigma^2 = \sigma^2$
<br>
$=>$
<br>
$\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \mu)^2$ 是方差$\sigma^2$的一个无偏估计    

这个结果符合直觉，并且在数学上也是显而易见的。  
现在，我们考虑随机变量$X$的数学期望$\mu$是未知的情形。这时，我们会倾向于无脑直接用样本均值$\bar{X}$替换掉上面式子中的$\mu$。这样做有什么后果呢？后果就是，  
**如果直接使用$\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - 
\bar{X})^2$作为估计，那么你会倾向于低估方差！**  
这是因为：  
$\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \bar{X})^2 $  
$=\frac{1}{n} \displaystyle \sum_{i=1}^{n}[(X_i - \mu) + (\mu - \bar{X})]^2$   $=\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \mu)^2 + \frac{2}{n} \displaystyle \sum_{i=1}^{n}(X_i -\mu)(\mu-\bar{X})+\frac{1}{n} \displaystyle \sum_{i=1}^{n}(\mu-\bar{X})^2$  
$=\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i -\mu)^2 + 2(\bar{X}-\mu)(\mu-\bar{X})+(\mu-\bar{X})^2\\= \frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \mu)^2 - (\mu - \bar{X})^2$

换言之，除非正好${\bar{X} = \mu}$，否则我们一定有
$\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \bar{X})^2  < \frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \mu^2 $

这个不等式说明了为什么直接使用$\frac{1}{n} \displaystyle \sum_{i=1}^{n}(X_i - \bar{X})^2 $会导致结果出现偏差  
那么，在不知道随机变量真实数学期望的前提下，如何“正确”的估计方差呢？  
##### 样本方差
定义：  
设$X_1,…,X_n$是随机变量X的$n$个样本，则样本方差定义为
$s^2 = \frac{1}{n-1}\displaystyle \sum_{i=1}^{n}(X_i - n\bar{X})^2$  
其中，$\bar{X}为样本均值$  
根据定义可以得出：
$s^2 = \frac{1}{n-1}\displaystyle \sum_{i=1}^{n}(X_i - n\bar{X})^2 \\= \frac{1}{n-1}\displaystyle \sum_{i=1}^{n}(X_i^2 -  2n\bar{X}^2 + \bar{X}^2) \\= \frac{1}{n-1}(\displaystyle \sum_{i=1}^{n}X_i^2 - n\bar{X}^2)$   
无偏性：  
$$E(s^2) = \dfrac{1}{n-1}(n\sigma^2 + n\mu^2 - n(\dfrac{\sigma^2}{n} + \mu^2)) = \sigma^2$$  
其中：
$\displaystyle \sum_{i=1}^{n}E(X_i^2) = D(X) + [E(X)]^2 = (\sigma^2 + \mu^2)\times n$  
$E(\bar{X}^2) = D(\bar{X}) + [E(\bar{X})]^2 = D(\frac{X_1+X_2+…+X_n}{n}) + \mu^2 = \frac{1}{n^2}D(X_1+X_2+…+X_n) + \mu^2 = \frac{1}{n^2}n\sigma^2 + \mu^2 = \frac{\sigma^2}{n} + \mu^2\$



# 抽样分布
## 定义
>当总体X服从一般分布（如指数分布、均匀分布等），要得出统计量的分布是很困难的；当总体X服从正态分布时，统计量$\bar{X}$、$S^2$是可以计算的，且服从一定的分布。这些分布就是下面要介绍的三大抽样分布——$χ^2$分布，$t$分布，$F$分布。

## 分位数/分位点
分位数是一个非常重要的概念，一开始也有点难理解。首先要明确一点，分位数分的是**面积**。更准确的说，分位数分的是**某个特定分布的概率密度函数曲线下的面积**。每给定一个分位数，这个概率密度函数曲线就被该点一分为二。

### 四分位数（Quartile）
>四分位数（Quartile）也称四分位点，是指在统计学中把所有数值由小到大排列并分成四等份，处于三个分割点位置的数值。  
例如1， 3， 5， 7， 9， 11，其3个四分位点分别是3，6，9。分别叫做第一四分位数（$Q_1$或者$x_{0.25}$），第二四分位数（$Q_2$或者$x_{0.5}$），第三四分位数（$Q_3$或者$x_{0.75}$）。(认识最多的就是$Q_2$,也就是中位数)

对于概率密度函数来说，四分位点就是将概率密度曲线下的面积均分为4部分的点。

#### 利用pd.DataFrame().describe()实现统计数据显示(包括分位数)


```python
import pandas as pd
import numpy as np

# 获取1到100的10个随机整数
sample = np.random.randint(1, 100, 10)
# 显示统计数据：平均值，标准差，最小值，分位数
pd.DataFrame(sample).describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>46.700000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>23.127905</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>37.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>46.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>59.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 上$\alpha$分位数(Upper Percentile)
>上$α$分位数是概率密度函数定义域内的一个数值，这个数值将概率密度函数曲线下的面积沿x轴分成了两个部分，其中该点右侧部分概率密度函数曲线与x轴围成的面积等于$α$。  

![](images/6_4.png)
由于概率密度函数曲线下的面积就是概率，因此上α分位数中的α既是该点右侧区域的面积，也是在这个分布中取到所有大于该点的值的概率。  
即：$$p(x≥x_{\alpha}) = \alpha$$

此时有两个值，一个是$α$，另一个是$x_α$。这两个值中确定其中一个，另一个值也就确定了。因此我们可以通过一个给定的α值，求在某个特定分布中的上$α$分位数，即$x_α$，的值；也可以在某个特定分布中，任意给定一个定义域内的点x，求取到比该点的值更大的值的概率，即该点的$α$值。

## $\chi^2$分布
### 定义
![](images/6_3.png)

### 概率密度(了解)
![](images/6_5.png)

![](images/6_6.png)

由上图可以看出，随着自由度n的增加，卡方分布越近似正态分布

### 性质
![](images/6_8.png)

证明1：  
$E(\chi^2)\\=E(\sum_{i=1}^{n}X_i^2)\\=\sum_{i=1}^{n}E(X_i^2)\\=\sum_{i=1}^{n}[D(X) + [E(x)]^2 ]\\=\sum_{i=1}^{n}D(X)\\=n$

$D(\chi^2)\\=D(\sum_{i=1}^{n}X_i^2)\\=\sum_{i=1}^{n}D(X_i^2)\\=\sum_{i=1}^{n}[E(X_i^4) - [E(X_i^2)]^2]\\=\sum_{i=1}^{n}(3-1)\\=2n$

证明2：  
$$Y_1 = \sum_{i=1}^{n_1}X_i$$  
$$Y_2 = \sum_{i=1}^{n_2}X_i$$  
由于：$$X_i \sim N(0, 1)$$  
所以：  
$$Y_1 + Y_2 = \sum_{i=1}^{n_1 + n_2}X_i$$  
即：  
$$Y_1 + Y_2 \sim \chi^2(n_1 + n_2)$$

### 上$\alpha$分位数

![](images/69.png)

$$P{ [\chi^2>\chi^2_{\alpha}(n)] } = \int_{\chi^2_{\alpha}(n)}^{\infty}f(y)dy =\alpha$$

### 上$\alpha$分位数表以及如何查看

![](images/70.png)

1. 确定自由度(n)
2. 确定分位点$\alpha$
3. 读出数据  


![](images/6_9.png)

## t分布

### 定义

![](images/6_10.png)

![](images/6_11.png)

### 上$\alpha$分位数(上侧分位数)

![](images/6_12.png)

### 性质
1. t分布的概率密度函数为偶函数
2. 对于任意实数x，当$n->\infty$,t分布近似于标准正态分布$N(0, 1)$
$$\lim_{n->\infty}f_T(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}x^2}$$


![](images/6_17.png)

![](images/6_13.png)

![](images/6_14.png)

### 例题
![](images/6_15.png)


![](images/6_16.png)

## F分布

### 定义

![](images/6_18.png)

### 上$\alpha$分位数

![](images/6_19.png)

### 如何查表
![](images/6_20.png)

## 小结

![](images/6_21.png)

## 单个正态总体的抽样分布

![](images/611.png)
![](images/612.png)
![](images/613.png)
![](images/614.png)
![](images/615.png)
![](images/616.png)
![](images/617.png)
![](images/618.png)
![](images/619.png)

## 两个正态总体的抽样分布

![](images/620.png)
![](images/621.png)
![](images/622.png)
![](images/623.png)
![](images/624.png)
![](images/625.png)
![](images/626.png)
![](images/627.png)


# python实现


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
```

## 统计数据


```python
def get_stats(dist_sample):
    sample = pd.DataFrame(dist_sample, columns=['sample'])
    sample['mean'] = np.mean(sample['sample'])
    sample['X_i - X_mean'] = sample['sample'] - sample['mean']
    sample['(X_i - X_mean)^2'] = np.power(sample['X_i - X_mean'], 2)
    sample['S^2'] = np.std(dist_sample)
    sample['B_2'] = ((np.alen(dist_sample)-1) / np.alen(dist_sample)) * sample['S^2']
    return sample
```


```python
uniform_sample = np.random.rand(10)
print(get_stats(uniform_sample).head())
_ = pd.Series(uniform_sample)
np.alen(uniform_sample)
```

         sample      mean  X_i - X_mean  (X_i - X_mean)^2       S^2       B_2
    0  0.030305  0.589302     -0.558997          0.312478  0.321702  0.289532
    1  0.932017  0.589302      0.342714          0.117453  0.321702  0.289532
    2  0.691355  0.589302      0.102053          0.010415  0.321702  0.289532
    3  0.956281  0.589302      0.366979          0.134673  0.321702  0.289532
    4  0.571374  0.589302     -0.017928          0.000321  0.321702  0.289532
    




    10




```python
normal_sample = np.random.randn(10)
print(get_stats(normal_sample).head())
```

         sample      mean  X_i - X_mean  (X_i - X_mean)^2       S^2       B_2
    0 -1.386327 -0.499904     -0.886423          0.785746  0.621319  0.559187
    1  0.867403 -0.499904      1.367307          1.869528  0.621319  0.559187
    2 -0.207902 -0.499904      0.292002          0.085265  0.621319  0.559187
    3 -0.450259 -0.499904      0.049644          0.002465  0.621319  0.559187
    4 -0.065373 -0.499904      0.434531          0.188817  0.621319  0.559187
    

## $\chi^2$分布

![](images/6_22.png)

### 绘图


```python
def chi2_distribution(df_list=[1,2,3,4]):
    """
    实现不同自由度的卡方分布
    : df :degree freedom 自由度，也就是卡方分布的n
    """
    # 参见chi^2函数
    chi2_dis_1 = stats.chi2(df_list[0])
    chi2_dis_2 = stats.chi2(df_list[1])
    chi2_dis_3 = stats.chi2(df_list[2])
    chi2_dis_4 = stats.chi2(df_list[3])
    
    # 设置x轴
    x_1 = np.linspace(chi2_dis_1.ppf(0.65),
                      chi2_dis_1.ppf(0.999), 100)
    x_2 = np.linspace(chi2_dis_2.ppf(0.0001),
                      chi2_dis_2.ppf(0.9999), 100)
    x_3 = np.linspace(chi2_dis_3.ppf(0.001),
                      chi2_dis_3.ppf(0.999), 100)
    x_4 = np.linspace(chi2_dis_4.ppf(0.001),
                      chi2_dis_4.ppf(0.999), 100)
    
    # 计算概率密度函数
    pdf_1 = chi2_dis_1.pdf(x_1)
    pdf_2 = chi2_dis_2.pdf(x_2)
    pdf_3 = chi2_dis_3.pdf(x_3)
    pdf_4 = chi2_dis_4.pdf(x_4)
    
    # 画图
    fig, ax = plt.subplots(1,1)
    ax.plot(x_1, pdf_1, 'g-', lw=2, label='df={}'.format(df_list[0]))
    ax.plot(x_2, pdf_2, 'r-', lw=2, label='df={}'.format(df_list[1]))    
    ax.plot(x_3, pdf_3, 'b-', lw=2, label='df={}'.format(df_list[2]))    
    ax.plot(x_4, pdf_4, 'y-', lw=2, label='df={}'.format(df_list[3]))    
    
    plt.ylabel('Probablity')
    plt.title(r'PDF of $\chi^2 Dist.$')
    ax.legend(loc='best', frameon=False)
    plt.show()

chi2_distribution([1,4,10,20])
```


![png](output_63_0.png)


### 分位数计算
例如计算:  

$$P{ [\chi^2(10)>20.483] } = ?$$
即，$$\chi^2_\alpha(10)=20.483$$


```python
# 计算概率密度
def calculate_cdf(df, upper_percent):
    """
    准确度比较低
    : df :自由度
    : upper_percentile: 上α分位数值
    """
    return stats.chi2.cdf(df, upper_percent)

# 计算自由度为10， 分位点α为20.483的概率
print(calculate_cdf(10, 20.483))

def calculate_cdf_2(df, upper_percent):
    """
    准确度高
    : sf:survival function
    """
    return stats.chi2.sf(upper_percent, df)
print(calculate_cdf_2(10, 20.483))
```

    0.026155741012841877
    0.025001449464351087
    

例如计算：
$$P[\chi^2(10)>\chi^2_{\alpha}(10)] = 0.05$$
$$求\chi^2_{\alpha(10)} = ?$$


```python
def calculate_upper_percentile(df, probablity):
    """
    ：df：自由度
    ：probability：概率
    """
    return stats.chi2.isf(probablity, df)

print(calculate_upper_percentile(10, 0.05))
```

    18.30703805327515
    

## t分布

![](images/6_23.png)

### 绘图


```python
def t_distribution(df_list=[1,2,3,4]):
    """
    实现不同自由度的卡方分布
    : df :degree freedom 自由度，也就是卡方分布的n
    """
    # 创建t分布函数
    t_dis_1 = stats.t(df_list[0])
    t_dis_2 = stats.t(df_list[1])
    t_dis_3 = stats.t(df_list[2])
#     t_dis_4 = stats.t(df_list[3])
    
    # 创建标准正态分布函数
    norm_dist = stats.norm()
    
    # 设置x轴
#     x_1 = np.linspace(t_dis_1.ppf(0.004),
#                       t_dis_1.ppf(0.999), 1000)
#     x_2 = np.linspace(t_dis_2.ppf(0.001),
#                       t_dis_2.ppf(0.999), 1000)
#     x_3 = np.linspace(t_dis_3.ppf(0.001),
#                       t_dis_3.ppf(0.999), 1000)
#     x_4 = np.linspace(t_dis_4.ppf(0.001),
#                       t_dis_4.ppf(0.999), 1000)
    x = np.linspace(-4, 4, 100)
    
    # 计算概率密度函数
    pdf_1 = t_dis_1.pdf(x)
    pdf_2 = t_dis_2.pdf(x)
    pdf_3 = t_dis_3.pdf(x)
#     pdf_4 = t_dis_4.pdf(x)
    pdf_5 = norm_dist.pdf(x)
    
    # 画图
    fig, ax = plt.subplots(1,1)
    ax.plot(x, pdf_1, 'g-', lw=2, label=r'$X \sim t({})$'.format(df_list[0]))
    ax.plot(x, pdf_2, 'r-', lw=2, label=r'$X \sim t({})$'.format(df_list[1]))    
    ax.plot(x, pdf_3, 'b-', lw=2, label=r'$X \sim t({})$'.format(df_list[2]))    
#     ax.plot(x, pdf_4, 'y-', lw=2, label=r'X \sim \chi^2({})'.format(df_list[3]))    
    ax.plot(x, pdf_5, 'k-', lw=2, label=r'$X\sim N(0,1)$')    
    
    plt.ylabel('Probablity')
    plt.title(r'PDF of t Dist.')
    ax.legend(loc='best', frameon=False)
    plt.show()

t_distribution([1,5,10])
```


![png](output_71_0.png)


### 分位数计算
例如计算:  

$$P{ [t(10)>1.8125] } = ?$$
即，$$t_\alpha(10)=1.8125$$


```python
def calculate_cdf_2(df, upper_percent):
    """
    计算t分布的上侧分位数
    准确度高
    : sf:survival function
    """
    return stats.t.sf(upper_percent, df)
print(calculate_cdf_2(10, 1.8125))
```

    0.04999682852392339
    

例如计算：
$$P[t(10)>t_{\alpha}(10)] = 0.05$$
$$求t_{\alpha(10)} = ?$$


```python
def calculate_upper_percentile(df, probabilty):
    """
    计算t分布上侧分位数对应的值
    : df:自由度
    : probability:概率
    """
    return stats.t.isf(probabilty, df)
print(calculate_upper_percentile(10, 0.05))
```

    1.8124611228107341
    

### F分布

![](images/6_24.png)


```python
def t_distribution(df_list=[1,2,3,4], df_list_2=[1,2,3,4]):
    """
    实现不同自由度的F分布
    : df_list :degree freedom 自由度，也就是F分布的n1
    : df_list_2 :degree freedom 自由度，也就是F分布的n_2
    """
    # 创建t分布函数
    f_dis_1 = stats.f(df_list[0], df_list_2[0])
    f_dis_2 = stats.f(df_list[1], df_list_2[1])
    f_dis_3 = stats.f(df_list[2], df_list_2[2])
#     t_dis_4 = stats.t(df_list[3])
    
    # 设置x轴
#     x_1 = np.linspace(t_dis_1.ppf(0.004),
#                       t_dis_1.ppf(0.999), 1000)
#     x_2 = np.linspace(t_dis_2.ppf(0.001),
#                       t_dis_2.ppf(0.999), 1000)
#     x_3 = np.linspace(t_dis_3.ppf(0.001),
#                       t_dis_3.ppf(0.999), 1000)
#     x_4 = np.linspace(t_dis_4.ppf(0.001),
#                       t_dis_4.ppf(0.999), 1000)
    x = np.linspace(0.01, 10, 100)
    
    # 计算概率密度函数
    pdf_1 = f_dis_1.pdf(x)
    pdf_2 = f_dis_2.pdf(x)
    pdf_3 = f_dis_3.pdf(x)
#     pdf_4 = t_dis_4.pdf(x)
    
    # 画图
    fig, ax = plt.subplots(1,1)
    ax.plot(x, pdf_1, 'g-', lw=2, label=r'$X \sim F({}, {})$'.format(df_list[0], df_list_2[0]))
    ax.plot(x, pdf_2, 'r-', lw=2, label=r'$X \sim F({}, {})$'.format(df_list[1], df_list_2[1]))    
    ax.plot(x, pdf_3, 'b-', lw=2, label=r'$X \sim F({}, {})$'.format(df_list[2], df_list_2[2]))    
#     ax.plot(x, pdf_4, 'y-', lw=2, label=r'X \sim \chi^2({})'.format(df_list[3]))    
    
    plt.ylabel('Probablity')
    plt.title(r'PDF of F Dist.')
    ax.legend(loc='best', frameon=False)
    plt.show()

t_distribution([1,30,100], [30,5,100])
```


![png](output_78_0.png)

