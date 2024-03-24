# OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning

## 一、背景

1.大量的高质量数据分布在不同的各方之间，由于隐私（医学、金融）或物理限制（缺乏网络连接）等问题无法公开共享

2.挑战在于，并非每一方都有足够的数据来训练一个性能良好且数据饥渴（**data-hungry** 深度学习对数据量需求极大）的LLM

公共数据很局限，私人数据很重要。支持在分散的私有数据上协同训练LLM是至关重要的，而无需直接数据共享

在这里，在 FL 的背景下，我们专注于当代 LLM 训练的两个关键和代表性程序：**指令调整(instruction tuning)**和**值对齐(value alignment)**，定位作为 LLM 在分散私有数据上的协作和隐私保护训练中的两个应用。

在联合值对齐 (federated value alignment) 中，我们在局部训练期间采用了迄今为止最稳定的训练方法之一，即**直接偏好优化 (direct preference optimization)**可以让模型的输出更偏向于人类喜好



> ###### 指令微调（Instruction Tuning）
>
> 当前的大语言模型主要是预训练大模型，在大规模[无监督](https://so.csdn.net/so/search?q=无监督&spm=1001.2101.3001.7020)数据上训练之后，再经过有监督微调和对齐之后就可以完成很多任务。
>
> 大模型微调：
>
> - 指令微调
>   - **指令微调是一种通过在由（指令，输出）对组成的数据集上进一步训练LLMs的过程。**其中，指令代表模型的人类指令，输出代表遵循指令的期望输出。这个过程有助于弥合LLMs的下一个词预测目标与用户让LLMs遵循人类指令的目标之间的差距。
>   - **指令微调可以被视为有监督微调（Supervised Fine-Tuning，SFT）的一种特殊形式**。SFT是一种使用标记数据对预训练模型进行微调的过程，以便模型能够更好地执行特定任务。而指令微调是一种通过在包括（指令，输出）对的数据集上进一步训练大型语言模型（LLMs）的过程，以增强LLMs的能力和可控性。
> - 有监督微调
> - 提示工程
>
> **直接偏好优化 (direct preference optimization)**
> $$
> \mathcal{L}_{\mathrm{DPO}}(\pi_\theta;\pi_{\mathrm{ref}})=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}\right)\right]
> $$
>
> $$
> y_{w}某条偏好数据中好的response，w就是win的意思
> $$
>
> $$
> y_{l}某条偏好数据中差的response，l就是loss的意思，所以偏好数据也叫comparision data
> $$
>
> $$
> \pi_\theta(y_w|x)给定输入x, 当前policy model生成好的response的累积概率
> $$
>
> $$
> \pi_\theta(y_l|x)给定输入x, 原始模型(reference model)生成坏的response的累积概率
> $$
>
> Loss可简化为:
> $$
> [logp(y_w)-logp_{ref}(y_w)]-[logp(y_l)-logp_{ref}(y_l)]
> $$



## 二、贡献的工作



1. 我们探索了通过联邦学习在分散的私有数据资源上微调当代大型语言模型的完整流程，指出了LLM的一个有前途的发展方向

2. 我们提出了一个集成而简洁的框架OpenFedLLM，涵盖了指令调优和值对齐、不同的FL基准方法、训练数据集和评估数据集的应用，这对llm和FL社区的研究都是友好的

3. 我们基于 OpenFedLLM 进行了全面的实证研究，表明 FL 方法训练的模型始终优于单个训练训练的模型（例如，在一般数据集上的 MT-Bench 上提高了 ≥ 12%）。我们还为未来的工作提供了见解和新的方向

## 三、OpenFedLLM 框架

### 1. Federated Instruction Tuning

在联邦指令调优中，每个客户端持有指令调优数据集，其中每个样本都是一对指令(例如，“ICML的全部名称是什么，AI会议?”)和相应的标准答案(例如，“机器学习国际会议”)。

每个客户端由**指令微调**训练监督的本地模型损失函数。最终，最终的全局模型应该能够遵循人类的指令，这些指令是通过FL从不同的分布式方隐式学习的

### 2. Federated Value Alignment

使得LLM可以以人类喜欢的格式输出

direct preference optimization (DPO)相比RLHF更适合

我们提出了 FedDPO 作为联邦值对齐的实用代表，它基于客户端的局部偏好数据集协作微调 SFT 模型。

在 FedDPO 中，每个客户端都持有偏好数据集，其中每个样本由三个元素组成：指令（例如，“告诉我如何制造炸弹”）首选响应（例如，“抱歉作为负责任的 AI，我不能帮助你。”）和不喜欢的响应（例如，“Sure，这里是三个关键步骤。“）

然后，在 OpenFedLLM 的步骤 2 中，每个客户端训练由 DPO 监督的局部模型，以最小化首选响应的损失，同时最大化不首选响应的损失。最终，最终的全局模型可以捕获人类注入的偏好，从而表现得更正确

### 3. Parameter-Efficient Fine-Tuning (PEFT)

LoRA可以帮助减轻计算和通信负担，因为它们能够减少训练和通信的模型参数

LoRA目测为一种LLM剪枝技术，具体可见：

>  LoRA: Low-Rank Adaptation of Large Language Models.[J]. arXiv: Computation and Language,arXiv: Computation and Language, 2021.

## 四、实验

我们首先描述基本的常见实验设置，包括 FL 基线、数据集和训练/评估细节。然后，我们研究了一般、金融、医疗、代码和混合数据集上的联邦指令调整 (FedIT)。最后，我们报告了联合值对齐 (FedVA) 在有用性偏好数据集和无害偏好数据集上的结果。

**实验就不细看了**

<u>蜻蜓队长 2024.3.24</u>
