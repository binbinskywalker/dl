生成器网络（generator network）：接收一个随机向量（潜在空间的随机点）作为输入，并将其解码为一张合成图像
判别器网络（discriminator network）：接收图像作为输入（真实或者合成），并预测该图像是来自训练集还是生成网络
简要流程
1 generoator网络将盛装为（latent_dim,）的向量映射为形状（64,64,3）的图像
2 discriminator将形状(64,64,3)的图像映射为一个二进制分数，并判定其真假
3 gan网络将genrrator和discriminator连接在一起：gan(x) = discriminator(generator(x))
4 通过带有真/假标签的真假图像训练判别器
5 为了训练生成器，需要利用gan模型的损失相对于生成器权重的梯度

