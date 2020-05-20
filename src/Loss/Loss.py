import torch
from torch import ones, multinomial
from torch.distributions import MultivariateNormal


class LossFunction:
    def __init__(self):
        pass

    def loss(self, **kwargs):
        raise NotImplementedError


class WeightedICALossFunction(LossFunction):
    def __init__(self, power, number_of_gausses, cuda, z_dim=None):
        super(WeightedICALossFunction, self).__init__()
        self.power = power
        self.number_of_gausses = number_of_gausses
        self.z_dim = z_dim
        self.cuda = cuda
        self.reduction_type = "mean"

    def random_choice_full(self, input, n_samples):
        if n_samples * self.number_of_gausses < input.shape[0]:
            replacement = False
        else:
            replacement = True
        idx = multinomial(ones(input.shape[0]), n_samples * self.number_of_gausses, replacement=replacement)
        sampled = input[idx].reshape(self.number_of_gausses, n_samples, -1)
        return torch.mean(sampled, axis=1)

    def loss(self, z, latent_normalization=False):
        if latent_normalization:
            x = (z - z.mean(dim=1, keepdim=True)) / z.std(dim=1, keepdim=True)
        else:
            x = z
        dim = self.z_dim if self.z_dim is not None else x.shape[1]
        scale = (1 / dim) ** self.power
        sampled_points = self.random_choice_full(x, dim)

        cov_mat = (scale * torch.eye(dim)).repeat(self.number_of_gausses, 1, 1).type(torch.float64)
        if self.cuda:
            cov_mat = cov_mat.cuda()

        mvn = MultivariateNormal(loc=sampled_points,
                                 covariance_matrix=cov_mat)

        weight_vector = torch.exp(mvn.log_prob(x.reshape(-1, 1, dim)))

        sum_of_weights = torch.sum(weight_vector, axis=0)
        weight_sum = torch.sum(x * weight_vector.T.reshape(self.number_of_gausses, -1, 1), axis=1)
        weight_mean = weight_sum / sum_of_weights.reshape(-1, 1)

        xm = x - weight_mean.reshape(self.number_of_gausses, 1, -1)
        wxm = xm * weight_vector.T.reshape(self.number_of_gausses, -1, 1)

        wcov = (wxm.permute(0, 2, 1).matmul(xm)) / sum_of_weights.reshape(-1, 1, 1)

        diag = torch.diagonal(wcov ** 2, dim1=1, dim2=2)
        diag_pow_plus = diag.reshape(diag.shape[0], diag.shape[1], -1) + diag.reshape(diag.shape[0], -1, diag.shape[1])

        tmp = (2 * wcov ** 2 / diag_pow_plus)

        triu = torch.triu(tmp, diagonal=1)
        normalize = 2.0 / (dim * (dim - 1))
        cost = torch.sum(normalize * triu) / self.number_of_gausses
        return cost


def MINE(T_xy, T_x_y):
    # compute the negative loss (maximise loss == minimise -loss)
    neg_loss = -(torch.mean(T_xy) - torch.log(torch.mean(torch.exp(T_x_y))))
    return neg_loss
