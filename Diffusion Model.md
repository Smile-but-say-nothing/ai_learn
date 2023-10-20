# Diffusion Model

## 扩散模型基本原理[^1]

复习一下正态分布$N(\mu,\sigma^2)$的概率密度函数：
$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}e^{[-\frac{(x-\mu)^2}{2\sigma^2}]}.
$$

其中，均值$\mu=\frac{\sum^{N}_{i=1}{x_i}}{N}$，标准差$\sigma=\sqrt{\frac{\sum^{N}_{i=1}{(x_i-\mu)^2}}{N}}$，所谓标准正态分布即$N(0, 1)$。

高斯噪声是指满足标准正态分布的随机噪声，从标准正态分布中随机采样而来。

扩散现象是指物质粒子从高浓度区域向低浓度区域移动扩散的过程。例如，在一杯水中滴入一滴墨水。**扩散模型受其启发，通过逐步向图片中加入高斯噪声来模拟这种现象，并且通过逆向求解从随机噪声中生成图像**。

### 前向加噪

![img](https://github.com/wangjia184/diffusion_model/raw/main/chain.png)
$$
x_0 \overset{q(x_1 | x_0)}{\rightarrow} x_1 \overset{q(x_2 | x_1)}{\rightarrow} x_2 \rightarrow \dots  \rightarrow x_{T-1} \overset{q(x_{t} | x_{t-1})}{\rightarrow} x_T
$$
此过程本质是马尔可夫过程，$q(x_{t} | x_{t-1})$在时间步$t$为$x_{t-1}$加权添加高斯噪声得到$x_t$：
$$
x_t = \sqrt{1-β_t}\times x_{t-1} + \sqrt{β_t}\times ϵ_{t}.
$$
上式中，$ϵ_{t}$是高斯噪声，从标准正态分布采样得到，$β_t$可能不是常数，有可能随$t$有线性、样条、三角等关系。但 $0 < β_1 < β_2 < β_3 < \dots < β_T < 1 $，也就是图像会越来越被高斯噪声所淹没。

为了方便后续计算，定义$a_t = 1 - β_t$:
$$
x_t = \sqrt{a_{t}}\times x_{t-1} +  \sqrt{1-a_t} \times ϵ_{t}
$$
现在来考虑$C\epsilon_t(C\neq0)$的数学意义，将服从标准正态分布的随机变量$\times C$后：
$$
\mu_1=\frac{\sum^{N}_{i=1}{C \times x_i}}{N}=C \times \frac{\sum^{N}_{i=1}{x_i}}{N}=C\mu
\\
\sigma_1=\sqrt{\frac{\sum^{N}_{i=1}{(C \times x_i-C \times \mu)^2}}{N}}=\sqrt{\frac{\sum^{N}_{i=1}{C \times (x_i-\mu)^2}}{N}}=C\sigma
$$
其仍服从正态分布，只是均值和标准差也同样变为$C$倍，此时$x \sim N(0,C^2)$。

### 考察$x_{t}$和$x_{t-2}$的关系

注意到：
$$
x_{t-1} = \sqrt{a_{t-1}}\times x_{t-2} +  \sqrt{1-a_{t-1}} \times ϵ_{t-1}
$$
带入$x_{t}$：
$$
x_t = \sqrt{a_{t}} (\sqrt{a_{t-1}}\times x_{t-2} +  \sqrt{1-a_{t-1}} \times ϵ_{t-1}) +  \sqrt{1-a_t} \times ϵ_t
$$
即：
$$
x_t = \sqrt{a_{t}a_{t-1}}\times x_{t-2} +  \sqrt{a_{t}(1-a_{t-1})} \times ϵ_{t-1} +  \sqrt{1-a_t} \times ϵ_t
$$
考虑到正态分布的叠加仍为正态分布，即$N(\mu_{1},\sigma_{1}^{2}) + N(\mu_{2},\sigma_{2}^{2}) = N(\mu_{1}+\mu_{2},\sigma_{1}^{2} + \sigma_{2}^{2})$，上式化简为：
$$
x_t = \sqrt{a_{t}a_{t-1}}\times x_{t-2} +  \sqrt{1 - \alpha_{t}\alpha_{t-1}} \times ϵ
$$
此时，$\epsilon \sim N(0, 1 - \alpha_{t}\alpha_{t-1})$，这种方法为重参数化技巧。

现在考虑$x_{t-2}$和$x_{t-3}$：
$$
x_{t-2} = \sqrt{a_{t-2}}\times x_{t-3} +  \sqrt{1-a_{t-2}} \times ϵ_{t-2}
$$
代入$x_t$中，有：
$$
x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}\times x_{t-3}  +  \sqrt{a_{t}a_{t-1}(1-a_{t-2})} \times ϵ_{t-2} +  \sqrt{1-a_{t}a_{t-1}}\times ϵ
$$
同理，利用正态分布的叠加性质，化简得：
$$
x_t = \sqrt{a_{t}a_{t-1}a_{t-2}}\times x_{t-3} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}} \times ϵ
$$
利用数学归纳法，有：
$$
x_t = \sqrt{a_{t}a_{t-1}a_{t-2}a_{t-3}...a_{2}a_{1}}\times x_{0} +  \sqrt{1-a_{t}a_{t-1}a_{t-2}a_{t-3}...a_{2}a_{1}}\times ϵ
$$
也就是仅仅采样一次，就可以从$x_0$得到$x_t$。

令$\bar{a}_{t}=a_{t}a_{t-1}a_{t-2}a_{t-3}...a_{2}a_{1}$：
$$
x_{t} = \sqrt{\bar{a}_t}\times x_0+ \sqrt{1-\bar{a}_t}\times ϵ , ϵ \sim N(0,1-\bar{a}_t)
$$
那么：
$$
q(x_{t}|x_{0}) = \frac{1}{\sqrt{2\pi } \sqrt{1-\bar{a}_{t}}} e^{\left (  -\frac{1}{2}\frac{(x_{t}-\sqrt{\bar{a}_{t}}x_0)^2}{1-\bar{a}_{t}}   \right ) }
$$

### 反向求解

复习一下Bayes定理：
$$
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
$$
所谓反向求解就是给定$x_t$求解$x_{t-1}$，利用Bayes定理即：
$$
P(x_{t-1}|x_t,x_0)=\frac{P(x_t|x_{t-1},x_0)P(x_{t-1}|x_0)}{P(x_t|x_0)}
$$
对于$P(x_t|x_0)$：
$$
P(x_t|x_0) \sim N(\sqrt{\bar{a}_{t}}x_0,1-\bar{a}_t)
$$
对于$P(x_{t-1}|x_0)$：
$$
P(x_{t-1}|x_0) \sim N(\sqrt{\bar{a}_{t-1}}x_0,1-\bar{a}_{t-1})
$$
对于$P(x_t|x_{t-1},x_0)$：
$$
P(x_t|x_{t-1},x_0) \sim N(\sqrt{{a}_{t}}x_{t-1},1-{a}_t)
$$
那么各自带入正态分布的概率密度函数，并且由$x_o \rightarrow x_t$有$x_0 = \frac{x_t - \sqrt{1-\bar{a}_t}\times ϵ}{\sqrt{\bar{a}_t}}$，最终整理得到：
$$
P(x_{t-1}|x_{t}) \sim N\left( 
      {\color{Purple} \frac{\sqrt{a_t}(1-\bar{a}_{t-1})}{1-\bar{a}_t}x_t
      +
      \frac{\sqrt{\bar{a}_{t-1}}(1-a_t)}{1-\bar{a}_t}\times \frac{x_t - \sqrt{1-\bar{a}_t}\times ϵ}{\sqrt{\bar{a}_t}} } ,
       {\color{Red} \sqrt{\frac{ (1-{a}_{t}) (1-\bar{a}_{t-1}) } { 1-\bar{a}_{t}}}^2} 
 \right)
$$
在上面我们已经得到了$x_0$和$x_t$之间的关系：
$$
x_{t} = \sqrt{\bar{a}_t}\times x_0+ \sqrt{1-\bar{a}_t}\times ϵ , ϵ \sim N(0,1-\bar{a}_t)
$$
也就是，我们将$t$时刻的图像都认为是在$x_0$上直接加噪（不同的）得到的，那么只要我们求解出了这个噪音模型，即$\epsilon$，就可以根据利用Bayes定理，根据$x_t$得到$x_{t-1}$了。因此，训练一个神经网络模型，输入$x_t$图像，来预测此图像相对于$x_0$原图上加入的噪音。这样就能得到$x_{t-1}$图像的概率分布，接着以此概率分布进行随机采样，得到$x_{t-1}$图像，继续作为输入喂给神经网络，使其预测$x_{t-1}$相对于原图的噪音，不断重复此过程，直到得到$x_0$。

在$T$时刻，$x_T \approx \epsilon$，因此可以认为任何一张服从标准正态分布的噪音图片，都是某张$x_0$叠加噪音之后的图像，以输入给神经网络。优化目标即要求反向过程中预测的噪声分布与前向过程中施加的噪声分布之间的“距离”最小。另外，值得一提的是，在神经网络的输入中还有时间$t$这个参数，这是使得神经网络能够学习到该图像在整个加噪过程中的位置。

以上就是DDPM[^2]的diffusion模型生成图像的简单原理。

![](https://huggingface.co/blog/assets/78_annotated-diffusion/diffusion_figure.png)



[^1]: https://github.com/wangjia184/diffusion_model/tree/main
[^2]:https://arxiv.org/abs/2006.11239