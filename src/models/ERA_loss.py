from torch import sum, sqrt, Tensor, FloatTensor


def ERA_loss(rec: FloatTensor,
             real: FloatTensor,
             emb: FloatTensor) -> FloatTensor:
    """ Embadding Regularized loss function: MSE + сумма разниц расстояний
    между двумя объектами в исходном и скрытом пространствах

    :param rec: Восстановленный профиль после сжатия
    :param real: Профиль до сжатия
    :param emb: Профиль в скрытом пространстве
    :return: len_sum: Значение функции потерь
    """

    def euclid(x: FloatTensor) -> Tensor:
        """Евклидова норма для Torch.Tensor"""
        x = x**2
        l = sum(x)
        l = sqrt(l)
        return l

    len_sum=0
    for i in range(real.size(dim=0)):
        for j in range(i + 1, real.size(dim=0)):
            len_real = euclid(real[i, :(len(real[0]) - 1)]-real[j, :(len(real[0]) - 1)])
            len_emb = euclid(emb[i, :(len(real[0]) - 1)]-emb[j, :(len(real[0]) - 1)])
            len_sum = len_sum + (len_real - len_emb)**2
    len_sum=len_sum+F.mse_loss(rec, real)
    return len_sum
